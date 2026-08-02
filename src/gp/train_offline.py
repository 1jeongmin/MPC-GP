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
    """단일 채널 exact GP (표준화 공간). 예측 평균·분산 제공.

    noise_gp: 선택적 **입력의존 잡음** 모델(이분산). 있으면 `sigma_n` 상수 대신
    `sigma_n^2(z)` 를 쓴다 — 자세한 근거는 `fit_noise_channel` docstring.
    """
    Z: np.ndarray            # (M, d) 표준화 딕셔너리 입력
    lengthscales: np.ndarray  # (d,)
    sigma_f: float
    sigma_n: float
    alpha: np.ndarray        # (M,) = (K+sig_n^2 I)^-1 y_std
    W: np.ndarray            # (M, M) = (K+sig_n^2 I)^-1 (분산용)
    noise_gp: "GpChannel | None" = None   # log(sigma_n^2) 을 예측하는 2차 GP
    # 2차 GP 전용: 학습 타깃 log(e^2) 의 평균. GP 는 평균 0 을 가정하므로 타깃을
    # 중심화해 학습하고 예측할 때 다시 더한다. 외삽 구간에서 2차 GP 의 평균이 0 으로
    # 돌아가면 sigma_n^2 -> exp(prior_mean) = **학습 전체의 평균 잡음** 으로 수렴한다
    # (등분산 모델과 같은 값). 즉 "모르는 곳에서는 평균으로 돌아간다"는 안전한 기본값.
    prior_mean: float = 0.0

    def sigma_n2_of(self, z_std: np.ndarray) -> np.ndarray:
        """이 입력에서의 잡음분산 sigma_n^2. z_std:(n,d) -> (n,).

        등분산이면 상수를 브로드캐스트하고, 이분산이면 2차 GP 로 예측한다.
        """
        if self.noise_gp is None:
            return np.full(np.atleast_2d(z_std).shape[0], self.sigma_n**2)
        # 2차 GP 는 log(sigma_n^2) 를 학습했다 -> exp 로 되돌린다 (`_sigma_n2_from_noise_gp`
        # 가 학습·예측 공통 출처. 여기서 식을 복제하면 둘이 갈라진다).
        return _sigma_n2_from_noise_gp(self.noise_gp, z_std)

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

        `noise_gp` 가 있으면 상수 sigma_n^2 대신 **입력의존** sigma_n^2(z) 를 쓴다.
        """
        return self.var_std(z_std) + self.sigma_n2_of(z_std)


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


# log(chi^2_1) 의 평균 = digamma(1/2) + log(2) = -1.2704. e^2 = sigma^2 * chi^2_1 이므로
# log(e^2) 의 평균은 log(sigma^2) 보다 이만큼 **작다**. 보정하지 않으면 sigma_n^2 이
# 체계적으로 exp(-1.27) = 0.28 배로 과소추정된다 (즉 과신 방향으로 편향).
_LOG_CHI2_1_MEAN = -1.2703628454614782


def _sigma_n2_from_noise_gp(noise_ch: "GpChannel", z_std: np.ndarray) -> np.ndarray:
    """잡음 GP 로부터 sigma_n^2(z) 를 계산한다. 예측·학습이 **같은 식**을 쓰게 하는 단일 출처.

    `GpChannel.sigma_n2_of` 는 **평균 채널**에서 부르는 진입점이고(그 안에서 이 함수를
    쓴다), 이 함수는 잡음 채널 자체를 인자로 받는다. 둘을 헷갈려 잡음 채널에
    `sigma_n2_of` 를 부르면 상수가 돌아온다(2026-08-03 에 낸 실수).
    """
    g = noise_ch.mean_std(z_std) + noise_ch.prior_mean
    return np.exp(np.clip(g, -30.0, 30.0))


def fit_noise_channel(Zs_full: np.ndarray, resid: np.ndarray, cfg: GpConfig) -> GpChannel:
    """**입력의존 잡음** sigma_n^2(z) 를 2차 GP 로 학습한다 (이분산 모델).

    Zs_full: (N,d) 표준화 입력 **전체**(딕셔너리가 아니라). resid: (N,) = r - mu(z).

    ## 왜 필요한가 (2026-08-03 측정)

    등분산 가정(sigma_n 상수 하나)이 심하게 틀렸다는 것이 측정으로 확인됐다.
    학습 분포 안(a_y=4)에서 GP 평균을 뺀 잔여 산포가 코너 강도에 따라 이렇게 변한다:

        구간(|gamma|)   실제 std(r-mu)   GP 예측 sigma   비율
        0.000~0.016        7.6e-5          6.34e-4       0.12  (8배 과대 -> 겁먹음)
        0.174~0.286        1.28e-3         6.56e-4       1.95  (2배 과소 -> **과신**)

    **실제 산포는 6.9배 변하는데 예측 sigma 는 1.0배(=상수)다.** 상수 하나로 직선과
    한계 코너를 동시에 덮을 방법이 없으므로, 직선에서는 과대·코너에서는 과신이
    구조적으로 강제된다. OOD(a_y=6)는 "코너가 더 험해진 것"이라 과신이 더 심해진다
    — rtf_ay6 의 z_std=2.914 가 이것이다.

    ## 방법 — 2단계 학습 (Kersting et al. 2007 의 단순화판)

    1. 평균 GP 를 먼저 학습한다(기존과 동일).
    2. 잔여오차 제곱 e^2 = (r - mu(z))^2 을 구하고, **log(e^2)** 에 2차 GP 를 얹는다.
       log 를 쓰는 이유: (a) 분산은 양수여야 하는데 exp 로 되돌리면 자동 보장되고,
       (b) 잡음 크기가 자릿수로 변하므로(6.9배) 로그 공간에서 더 선형에 가깝다.
    3. 편향 보정: e^2 = sigma^2 * chi^2_1 이라 E[log e^2] = log sigma^2 - 1.27 이다.
       이 상수를 되돌려주지 않으면 sigma_n^2 이 일률적으로 0.28배 과소추정된다.

    **ARD 가 어느 입력이 산포를 좌우하는지 스스로 고른다** — |gamma| 하나로는 부족하다는
    것이 측정에서 드러났기 때문이다(위 표의 최고 구간에서 산포가 오히려 내려간다.
    가장 급한 코너는 vx_max 가 낮아 실제 횡력이 작다).

    ## 안전장치

    - 딕셔너리는 `cfg.noise_M` 개만 쓴다(기본 500). 잡음 함수는 평균 함수보다 매끄러워
      적은 점으로 충분하고, 스텝당 비용이 O(M^2) 라 예산을 아껴야 한다.
    - 타깃을 중심화해 학습하므로, **외삽 구간에서 2차 GP 평균이 0 으로 돌아가면
      sigma_n^2 이 학습 전체 평균 잡음으로 수렴한다** = 등분산 모델과 같은 값.
      즉 최악의 경우에도 종전보다 나빠지지 않는다.
    - log(e^2) 의 하한을 sigma_n_floor^2 로 깐다 — e^2 가 0 에 가까운 점에서 log 가
      -inf 로 발산하는 것을 막는다(정칙화 정신은 sigma_n_floor 와 동일).

    ## ★ 반드시 필요한 두 가지 처리 (2026-08-03 측정으로 확인)

    **① 국소 평활 — 안 하면 신호가 잡음에 묻힌다.**
    한 점의 e^2 는 sigma^2 의 **극도로 부정확한** 추정치다. e^2 = sigma^2 * chi^2_1 이라
    `Var[log e^2] = pi^2/2 ≈ 4.93`(표준편차 2.22)로 **sigma 와 무관하게 고정**인데,
    잡으려는 신호 범위는 log(6.9) ≈ 1.93 밖에 안 된다. **잡음이 신호보다 크다.**
    실제로 평활 없이 학습했더니 잡음 GP 가 "이건 다 잡음"으로 판단해 거의 상수를
    내놓았고(구간별 변동 1.12배 vs 실제 17배), 폐루프 캘리브레이션이 오히려 나빠졌다
    (lpf_ay4 z_std 1.153 -> 1.725).
    시간축 이동평균(`cfg.noise_smooth_window` 스텝)으로 먼저 뭉치면 로그공간 잡음
    분산이 창 크기에 반비례해 줄어든다(w=25 이면 4.93 -> 0.20, 표준편차 0.44 로
    신호 1.93 보다 충분히 작아진다). 인접 스텝은 z 가 비슷하므로 국소 평균으로 타당하다.

    **② 1차 모멘트 정합 — 안 하면 수준이 통째로 낮게 깔린다.**
    `exp(E[log e^2])` 는 기하평균 성격이라 꼬리가 두꺼우면 산술평균보다 체계적으로
    작다. 그래서 학습 전체에서 `mean(e^2 / sigma_n^2(z)) = 1` 이 되도록 상수를
    맞춘다. **이것은 커버리지를 맞추는 사후 스케일링이 아니다** — 평가 지표를 보고
    맞추는 게 아니라 **학습 데이터의 1차 모멘트**를 맞추는 추정기 구성이고,
    올바른 우도라면 ML 이 스스로 했을 일이다. 평가 시나리오는 쳐다보지 않는다.
    """
    e2 = np.maximum(resid**2, cfg.sigma_n_floor**2)

    w = int(cfg.noise_smooth_window)
    if w > 1:
        # 중심 이동평균 (경계는 가장자리 값으로 패딩 — 길이 유지).
        pad = w // 2
        k = np.ones(w) / w
        e2_s = np.convolve(np.pad(e2, pad, mode="edge"), k, mode="same")[pad:pad + len(e2)]
        e2_s = np.maximum(e2_s, cfg.sigma_n_floor**2)
    else:
        e2_s = e2

    t = np.log(e2_s) - _LOG_CHI2_1_MEAN        # 편향 보정된 log(sigma^2) 추정치
    prior_mean = float(np.mean(t))
    # 딕셔너리: 평균 GP 와 같은 균일 subsample 규약(시간순 대표성 유지).
    n = len(t)
    idx = (np.unique(np.linspace(0, n - 1, cfg.noise_M).astype(int))
           if cfg.noise_M < n else np.arange(n))
    ch = _fit_channel(Zs_full[idx], t[idx] - prior_mean, cfg)
    ch.prior_mean = prior_mean

    # ② 1차 모멘트 정합: 학습 전체에서 E[e^2 / sigma_n^2(z)] = 1 이 되게 수준을 맞춘다.
    #
    # ★ `ch.sigma_n2_of(...)` 를 부르면 안 된다 — `ch` 는 **잡음 GP 자기 자신**이라
    #   `ch.noise_gp is None` 이고, 그러면 모델링된 sigma_n^2(z) 가 아니라 잡음 GP 의
    #   상수 sigma_n^2 이 돌아온다. 2026-08-03 에 실제로 이 실수를 냈고, 정합이 통째로
    #   무효가 되어(E[e^2/sigma_n^2]=19.5) "이분산이 캘리브레이션을 악화시킨다"는
    #   잘못된 결론이 나올 뻔했다. 여기서는 예측식을 직접 쓴다.
    sigma_n2 = _sigma_n2_from_noise_gp(ch, Zs_full)
    scale = float(np.mean(e2 / np.maximum(sigma_n2, 1e-300)))
    ch.prior_mean += float(np.log(max(scale, 1e-300)))

    # 사후 검증 — 정합이 실제로 걸렸는지 확인한다(위 실수의 재발 방지).
    achieved = float(np.mean(e2 / np.maximum(_sigma_n2_from_noise_gp(ch, Zs_full), 1e-300)))
    if not (0.9 < achieved < 1.1):
        raise RuntimeError(
            f"잡음 모델 1차 모멘트 정합 실패: E[e^2/sigma_n^2]={achieved:.3f} (목표 1.0). "
            "정합 계산식이 sigma_n^2(z) 를 제대로 참조하는지 확인하라.")
    return ch


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

    if cfg.noise_model == "input_dependent":
        # 2단계: 평균 GP 가 못 맞춘 나머지의 크기를 입력의 함수로 학습한다.
        # 잔여오차는 **전체 데이터**에서 뽑는다 — 딕셔너리 위에서만 재면 그 점들은
        # GP 가 거의 보간해버려 산포가 실제보다 작게 나온다(딕셔너리 밖의 진짜
        # 산포를 재야 한다. 2026-07-31 의 "M=100 부분표본" 진단과 같은 함정).
        Zs_full = z_scaler.transform(dataset.Z)
        Rs_full = r_scaler.transform(dataset.R)
        for j, ch in enumerate(channels):
            resid = Rs_full[:, j] - ch.mean_std(Zs_full)
            ch.noise_gp = fit_noise_channel(Zs_full, resid, cfg)

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
        if ch.noise_gp is not None:   # 이분산 잡음 GP (있을 때만)
            g = ch.noise_gp
            d[f"nZ{j}"] = g.Z; d[f"nls{j}"] = g.lengthscales
            d[f"nsf{j}"] = g.sigma_f; d[f"nsn{j}"] = g.sigma_n
            d[f"nalpha{j}"] = g.alpha; d[f"nW{j}"] = g.W
            d[f"npm{j}"] = g.prior_mean
    np.savez(path, n_channels=len(gp.channels), n_lags=gp.n_lags,
             lag_mode=gp.lag_mode, **d)
    return path


def load_gp(path: Path) -> TwoChannelGP:
    d = np.load(path)
    zc = Standardizer(d["z_mean"], d["z_scale"])
    rc = Standardizer(d["r_mean"], d["r_scale"])
    channels = []
    for j in range(int(d["n_channels"])):
        # 이분산 잡음 GP 는 이 필드가 생기기 전 저장본에 없다 -> None (하위호환).
        ngp = None
        if f"nZ{j}" in d.files:
            ngp = GpChannel(Z=d[f"nZ{j}"], lengthscales=d[f"nls{j}"],
                            sigma_f=float(d[f"nsf{j}"]), sigma_n=float(d[f"nsn{j}"]),
                            alpha=d[f"nalpha{j}"], W=d[f"nW{j}"],
                            prior_mean=float(d[f"npm{j}"]))
        channels.append(GpChannel(Z=d[f"Z{j}"], lengthscales=d[f"ls{j}"],
                                  sigma_f=float(d[f"sf{j}"]), sigma_n=float(d[f"sn{j}"]),
                                  alpha=d[f"alpha{j}"], W=d[f"W{j}"], noise_gp=ngp))
    # n_lags/lag_mode 는 이 필드가 생기기 전(2026-08-01) 저장본에 없다 -> 기본값 (하위호환).
    n_lags = int(d["n_lags"]) if "n_lags" in d.files else 0
    lag_mode = str(d["lag_mode"]) if "lag_mode" in d.files else "full"
    return TwoChannelGP(z_scaler=zc, r_scaler=rc, channels=channels,
                        n_lags=n_lags, lag_mode=lag_mode)
