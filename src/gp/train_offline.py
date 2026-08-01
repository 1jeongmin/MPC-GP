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
from src.gp.dataset import ResidualDataset, Standardizer, apply_lags
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
        """**잠재함수** 사후분산 Var[f*] (표준화). z_std:(n,d) -> (n,).

        관측잡음 sigma_n^2 은 **포함하지 않는다** — "이 입력 근처에 데이터가 있는가"
        (epistemic)를 재는 양이라 불확실성 지도에 쓴다. 관측된 잔차를 덮는지 보는
        캘리브레이션에는 `var_obs_std`(= 여기에 sigma_n^2 을 더한 것)를 써야 한다.
        (Rasmussen & Williams 2006, Eq. 2.19 vs 2.24 — 예측 대상이 f* 냐 y* 냐.)

        `sum((k@W)*k, axis=1)` 는 `einsum("ij,jk,ik->i", k, W, k)` 와 **수학적으로
        동일**하지만 행렬곱이 BLAS 로 내려가 M 이 커질수록 훨씬 빠르다. M 을 100 ->
        수천으로 키우면서(2026-07-31) 이 항이 스텝당 O(M^2) 병목이 되어 교체했다.
        """
        k = ard_rbf_np(z_std, self.Z, self.lengthscales, self.sigma_f)   # (n,M)
        prior = self.sigma_f**2                                          # k(z,z)
        v = prior - np.sum((k @ self.W) * k, axis=1)
        return np.maximum(v, 0.0)

    def var_obs_std(self, z_std: np.ndarray) -> np.ndarray:
        """**관측** 예측분산 Var[y*] = Var[f*] + sigma_n^2 (표준화). z_std:(n,d) -> (n,).

        캘리브레이션이 비교하는 대상은 실제로 관측된 1스텝 잔차이므로 이 쪽이 맞다.
        M 이 커지면 Var[f*] 가 데이터 근처에서 0 으로 붕괴해 sigma_n^2 이 지배항이
        된다 — M=1000 에서 Var[f*]=2.35e-8 vs sigma_n^2=3.66e-7 로 **15배**였고,
        이 항을 빼먹은 탓에 "M 을 키우면 캘리브레이션이 나빠진다"는 잘못된 결론이
        나왔다 (2026-07-31 진단).
        """
        return self.var_std(z_std) + self.sigma_n**2


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
    """단일 채널 Type-II ML 학습 (**다중 재시작** — NLML 이 가장 낮은 해를 채택).

    sigma_n 에 하한(sigma_n_floor)을 둔다 — 없으면 sigma_n->0 으로 보간 과적합이
    일어나 학습 궤적 밖(배포 분포)에서 진동한다 (Phase 7 진단으로 확인).

    ## 왜 재시작이 필요한가 (2026-08-01 측정으로 확인)

    NLML 은 비볼록이고, 이 문제에는 **"전부 잡음"으로 붕괴하는 국소최적**이 있다:
    lengthscale 이 전부 하한(1e-3), sigma_f -> 0, sigma_n -> 1(표준화 단위에서 타깃
    산포 전체) 로 가서 "예측할 수 없다"고 답하는 해다. 단일 시작점(init_lengthscale
    = 1.0, 종전 기본값)은 gamma 채널에서 실제로 여기 빠졌다:

        init_ls 0.1 / 0.3 -> NLML  1376.5  (sigma_f 5.71, sigma_n 0.452, ls~[1.9,0.9,0.9])
        init_ls 1.0 / 3.0 -> NLML  2837.9  (sigma_f 0.001, sigma_n 1.000, ls 전부 하한)

    NLML 차이가 1461 이라 근소한 차이가 아니다 — 완전히 다른 모델이다. 붕괴한 해는
    GP 평균이 ~0 이 되어 잔차 보정을 사실상 포기하고, 그 상태로 캘리브레이션을 재면
    "특징이 나쁘다"는 잘못된 결론이 나온다. 2026-07-30 에 "gamma lengthscale 이 하한
    근처인데 버그인지 데이터 특성인지 미판단"으로 남겼던 것이 바로 이 현상이다.

    `init_lengthscale` 은 재시작 후보에 **함께** 들어간다(사용자 지정값을 버리지
    않는다). 비용은 후보 수에 비례한다.
    """
    d = Zs.shape[1]
    sn0 = max(cfg.init_sigma_n, cfg.sigma_n_floor)
    # bounds: lengthscale·sigma_f 는 넉넉히, sigma_n 은 floor 이상.
    bounds = [(np.log(1e-3), np.log(1e3))] * d + [(np.log(1e-3), np.log(1e3))] \
        + [(np.log(cfg.sigma_n_floor), np.log(1e2))]

    inits = sorted({float(cfg.init_lengthscale), *cfg.init_lengthscale_restarts})
    best, best_nlml = None, np.inf
    for init_ls in inits:
        theta0 = np.log(np.concatenate([np.full(d, init_ls),
                                        [cfg.init_sigma_f, sn0]]))
        res = minimize(_nlml, theta0, args=(Zs, y, cfg.jitter),
                       method="L-BFGS-B", bounds=bounds)
        if res.fun < best_nlml:
            best, best_nlml = res.x, float(res.fun)
    theta = best

    ls = np.exp(theta[:d]); sf = float(np.exp(theta[d])); sn = float(np.exp(theta[d + 1]))
    K = ard_rbf_np(Zs, Zs, ls, sf) + (sn**2 + cfg.jitter) * np.eye(len(y))
    W = np.linalg.inv(K)
    alpha = W @ y
    return GpChannel(Z=Zs, lengthscales=ls, sigma_f=sf, sigma_n=sn, alpha=alpha, W=W)


@dataclass
class TwoChannelGP:
    """v_y, gamma 두 채널 GP + 입력·타깃 표준화. 실단위 예측 제공.

    n_lags / lag_mode: 학습에 쓴 지연 규약. 배포(`mpc_gp.GPStepModel`)가 **같은
    규약으로** 특징을 만들려면 알아야 하므로 모델과 함께 저장한다 — 표준화 통계를
    함께 저장하는 것과 같은 이유다(`gp-residual.md` 「저장하지 않으면 재현이 깨진다」).
    """
    z_scaler: Standardizer
    r_scaler: Standardizer
    channels: list[GpChannel]   # [v_y, gamma]
    n_lags: int = 0
    lag_mode: str = "full"

    @property
    def input_dim(self) -> int:
        """특징 z 의 차원 = 3 * (n_lags + 1). 딕셔너리에서 직접 읽는다."""
        return int(self.channels[0].Z.shape[1])

    def predict_mean(self, z: np.ndarray) -> np.ndarray:
        """실단위 잔차 평균 예측. z:(n,3) -> (n,2)."""
        zs = self.z_scaler.transform(z)
        mstd = np.column_stack([ch.mean_std(zs) for ch in self.channels])  # (n,2) 표준화
        return self.r_scaler.inverse(mstd)

    def predict_var(self, z: np.ndarray) -> np.ndarray:
        """실단위 **잠재** 사후분산 Var[f*]. z:(n,3) -> (n,2). 스케일^2 로 역표준화.

        **불확실성 지도(epistemic)용**이다. 캘리브레이션에는 `predict_var_obs` 를 써라.
        """
        zs = self.z_scaler.transform(z)
        vstd = np.column_stack([ch.var_std(zs) for ch in self.channels])   # (n,2)
        return vstd * (self.r_scaler.scale**2)

    def predict_var_obs(self, z: np.ndarray) -> np.ndarray:
        """실단위 **관측** 예측분산 Var[y*] = Var[f*] + sigma_n^2. z:(n,3) -> (n,2).

        **캘리브레이션용**이다 (비교 대상이 관측된 잔차이므로).
        """
        zs = self.z_scaler.transform(z)
        vstd = np.column_stack([ch.var_obs_std(zs) for ch in self.channels])
        return vstd * (self.r_scaler.scale**2)


def train(dataset: ResidualDataset, cfg: GpConfig) -> TwoChannelGP:
    """데이터셋으로 2채널 exact GP 를 학습한다.

    딕셔너리는 M개 균일 subsample. 입력·타깃 표준화 후 채널별 Type-II ML.

    `cfg.n_lags > 0` 이면 **solve 전에** 지연 특징을 붙인다 — subsample 뒤에 붙이면
    "직전 스텝"이 실제로는 수십 스텝 전이 되므로 순서가 중요하다(`apply_lags` 참조).
    """
    dataset = apply_lags(dataset, cfg.n_lags, cfg.lag_mode)
    dict_ds = dataset.subsample(cfg.M)
    z_scaler = Standardizer.fit(dict_ds.Z)
    r_scaler = Standardizer.fit(dict_ds.R)
    Zs = z_scaler.transform(dict_ds.Z)
    Rs = r_scaler.transform(dict_ds.R)
    channels = [_fit_channel(Zs, Rs[:, j], cfg) for j in range(Rs.shape[1])]
    return TwoChannelGP(z_scaler=z_scaler, r_scaler=r_scaler, channels=channels,
                        n_lags=cfg.n_lags, lag_mode=cfg.lag_mode)


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
    np.savez(path, n_channels=len(gp.channels), n_lags=gp.n_lags,
             lag_mode=gp.lag_mode, **d)
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
    # n_lags/lag_mode 는 이 필드가 생기기 전(2026-08-01) 저장본에 없다 -> 기본값 (하위호환).
    n_lags = int(d["n_lags"]) if "n_lags" in d.files else 0
    lag_mode = str(d["lag_mode"]) if "lag_mode" in d.files else "full"
    return TwoChannelGP(z_scaler=zc, r_scaler=rc, channels=channels,
                        n_lags=n_lags, lag_mode=lag_mode)
