"""offline exact GP 학습 — Type-II ML (.claude/rules/gp-residual.md).

- 딕셔너리 Z = 학습 궤적에서 균일 subsample 한 M개 점 (표준화 공간).
- exact GP: K = RBF(Z,Z) + sigma_n^2 I, alpha = K^-1 y_std.
- Type-II ML: NLML = 0.5 y^T alpha + sum(log diag(L)) + 0.5 n log(2pi),
  L = chol(K). **log-det 생략 금지** (Rasmussen & Williams 2006, Eq. 5.8).
- 채널 v_y, gamma 독립 학습. 입력·타깃 표준화, 통계 저장·재적용.

사후분산은 제어에 넣지 않되 반드시 계산·저장한다 (predict_var). mean-only 제어.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from src.config import GpConfig
from src.gp.dataset import ResidualDataset, Standardizer
from src.gp.kernels import ard_rbf_np


@dataclass
class GpChannel:
    """단일 채널 exact GP (표준화 공간). 예측 평균·분산 제공."""
    Z: np.ndarray            # (M, d) 표준화 딕셔너리 입력
    lengthscales: np.ndarray  # (d,)
    sigma_f: float
    sigma_n: float
    alpha: np.ndarray        # (M,) = (K+sig_n^2 I)^-1 y_std
    W: np.ndarray            # (M, M) = (K+sig_n^2 I)^-1 (분산용)

    def mean_std(self, z_std: np.ndarray) -> np.ndarray:
        """표준화 입력에서 표준화 평균 예측. z_std:(n,d) -> (n,)."""
        k = ard_rbf_np(z_std, self.Z, self.lengthscales, self.sigma_f)   # (n,M)
        return k @ self.alpha

    def var_std(self, z_std: np.ndarray) -> np.ndarray:
        """표준화 입력에서 표준화 사후분산. z_std:(n,d) -> (n,)."""
        k = ard_rbf_np(z_std, self.Z, self.lengthscales, self.sigma_f)   # (n,M)
        prior = self.sigma_f**2                                          # k(z,z)
        v = prior - np.einsum("ij,jk,ik->i", k, self.W, k)
        return np.maximum(v, 0.0)


def _nlml(theta_log: np.ndarray, Zs: np.ndarray, y: np.ndarray, jitter: float) -> float:
    """음의 로그 주변우도. theta_log = log([l_1..l_d, sigma_f, sigma_n])."""
    d = Zs.shape[1]
    ls = np.exp(theta_log[:d])
    sf = np.exp(theta_log[d])
    sn = np.exp(theta_log[d + 1])
    K = ard_rbf_np(Zs, Zs, ls, sf) + (sn**2 + jitter) * np.eye(len(y))
    try:
        L = np.linalg.cholesky(K)
    except np.linalg.LinAlgError:
        return 1e10
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
    return float(0.5 * y @ alpha + np.sum(np.log(np.diag(L))) + 0.5 * len(y) * np.log(2 * np.pi))


def _fit_channel(Zs: np.ndarray, y: np.ndarray, cfg: GpConfig) -> GpChannel:
    """단일 채널 Type-II ML 학습.

    sigma_n 에 하한(sigma_n_floor)을 둔다 — 없으면 sigma_n->0 으로 보간 과적합이
    일어나 학습 궤적 밖(배포 분포)에서 진동한다 (Phase 7 진단으로 확인).
    """
    d = Zs.shape[1]
    sn0 = max(cfg.init_sigma_n, cfg.sigma_n_floor)
    theta0 = np.log(np.concatenate([np.full(d, cfg.init_lengthscale), [cfg.init_sigma_f, sn0]]))
    # bounds: lengthscale·sigma_f 는 넉넉히, sigma_n 은 floor 이상.
    bounds = [(np.log(1e-3), np.log(1e3))] * d + [(np.log(1e-3), np.log(1e3))] \
        + [(np.log(cfg.sigma_n_floor), np.log(1e2))]
    res = minimize(_nlml, theta0, args=(Zs, y, cfg.jitter), method="L-BFGS-B", bounds=bounds)
    theta = res.x
    ls = np.exp(theta[:d]); sf = float(np.exp(theta[d])); sn = float(np.exp(theta[d + 1]))
    K = ard_rbf_np(Zs, Zs, ls, sf) + (sn**2 + cfg.jitter) * np.eye(len(y))
    W = np.linalg.inv(K)
    alpha = W @ y
    return GpChannel(Z=Zs, lengthscales=ls, sigma_f=sf, sigma_n=sn, alpha=alpha, W=W)


@dataclass
class TwoChannelGP:
    """v_y, gamma 두 채널 GP + 입력·타깃 표준화. 실단위 예측 제공."""
    z_scaler: Standardizer
    r_scaler: Standardizer
    channels: list[GpChannel]   # [v_y, gamma]

    def predict_mean(self, z: np.ndarray) -> np.ndarray:
        """실단위 잔차 평균 예측. z:(n,3) -> (n,2)."""
        zs = self.z_scaler.transform(z)
        mstd = np.column_stack([ch.mean_std(zs) for ch in self.channels])  # (n,2) 표준화
        return self.r_scaler.inverse(mstd)

    def predict_var(self, z: np.ndarray) -> np.ndarray:
        """실단위 사후분산. z:(n,3) -> (n,2). 분산은 스케일^2 로 역표준화."""
        zs = self.z_scaler.transform(z)
        vstd = np.column_stack([ch.var_std(zs) for ch in self.channels])   # (n,2)
        return vstd * (self.r_scaler.scale**2)


def train(dataset: ResidualDataset, cfg: GpConfig) -> TwoChannelGP:
    """데이터셋으로 2채널 exact GP 를 학습한다.

    딕셔너리는 M개 균일 subsample. 입력·타깃 표준화 후 채널별 Type-II ML.
    """
    dict_ds = dataset.subsample(cfg.M)
    z_scaler = Standardizer.fit(dict_ds.Z)
    r_scaler = Standardizer.fit(dict_ds.R)
    Zs = z_scaler.transform(dict_ds.Z)
    Rs = r_scaler.transform(dict_ds.R)
    channels = [_fit_channel(Zs, Rs[:, j], cfg) for j in range(Rs.shape[1])]
    return TwoChannelGP(z_scaler=z_scaler, r_scaler=r_scaler, channels=channels)


def save_gp(path: Path, gp: TwoChannelGP) -> Path:
    """학습된 GP 를 npz 로 저장 (표준화 통계 포함)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    d = {
        "z_mean": gp.z_scaler.mean, "z_scale": gp.z_scaler.scale,
        "r_mean": gp.r_scaler.mean, "r_scale": gp.r_scaler.scale,
    }
    for j, ch in enumerate(gp.channels):
        d[f"Z{j}"] = ch.Z; d[f"ls{j}"] = ch.lengthscales
        d[f"sf{j}"] = ch.sigma_f; d[f"sn{j}"] = ch.sigma_n
        d[f"alpha{j}"] = ch.alpha; d[f"W{j}"] = ch.W
    np.savez(path, n_channels=len(gp.channels), **d)
    return path


def load_gp(path: Path) -> TwoChannelGP:
    d = np.load(path)
    zc = Standardizer(d["z_mean"], d["z_scale"])
    rc = Standardizer(d["r_mean"], d["r_scale"])
    channels = []
    for j in range(int(d["n_channels"])):
        channels.append(GpChannel(Z=d[f"Z{j}"], lengthscales=d[f"ls{j}"],
                                  sigma_f=float(d[f"sf{j}"]), sigma_n=float(d[f"sn{j}"]),
                                  alpha=d[f"alpha{j}"], W=d[f"W{j}"]))
    return TwoChannelGP(z_scaler=zc, r_scaler=rc, channels=channels)
