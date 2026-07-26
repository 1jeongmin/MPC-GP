"""UQ 캘리브레이션 — Part 1 의 **주 지표** (.claude/rules/sim-experiment.md 지표).

추종오차가 아니라 이것이 Part 1 의 주 지표다. 묻는 것은 "누가 잘 따라가나" 가 아니라
**"각 방법이 자기 모델오차의 불확실성을 옳게 말하는가"** 다.

## 비교 대상은 반드시 같아야 한다 — 1스텝 이산 잔차 r_k

    r_k = x_{k+1}^plant - f_nom_d(x_k, u_k, vx_k, kappa_k)      (gp-residual.md)
    단위: v_y 채널 [m/s], gamma 채널 [rad/s]  (상태 **차이**, 미분 아님)

- **GP** 는 이 r_k 를 직접 예측한다: (mu_GP, var_GP) 를 그대로 쓴다.
- **KF** 는 r_k 가 아니라 **연속 외란** d [m/s^2, rad/s^2] 를 추정한다. 단위가 다르므로
  P 의 d-블록을 그대로 GP 사후분산과 나란히 놓으면 **틀린 비교**다. 이산 잔차 공간으로
  환산해야 한다 (Phase 8 결정):

      G(x,u,vx,kappa) = d[ rk4_step(f_nom + B_c d) ] / d[d]        (4x2, ca.jacobian)
      G_dyn = G[0:2, :]                                            (v_y, gamma 채널)

      mean_KF = G_dyn @ d_hat
      cov_KF  = G_dyn @ P[4:6,4:6] @ G_dyn^T

  `G ~ dt*I` 이지만 A 결합의 2차항이 있으므로 **dt 로 근사하지 마라.** 야코비안으로
  정확히 뽑는다 (수기 미분 금지 — ekf.py 전방호환 원칙과 동일).
  `G` 는 `A(vx)` 를 거치므로 `vx` 에 의존한다. 스텝마다 그 스텝의 vx 로 평가한다.

## 프레이밍 주의 (ekf-baseline.md 「역할 혼동 금지」)

두 분산은 종류가 다르다: GP 사후분산은 **epistemic**(데이터 희소성), KF 의 P 는 고정
Q_kf 를 시간축으로 전파한 **aleatoric** 이다. 이 비교의 목적은 "KF 가 틀렸다" 가 아니라
**둘이 다른 것을 재고 있음을 정량화**하는 것이다. 커버리지 숫자를 우열로만 읽지 마라.
"""
from __future__ import annotations

import casadi as ca
import numpy as np

from src.config import VehicleConfig
from src.models.integrators import rk4_step
from src.models.linear_bicycle import dynamics_ca

# 캘리브레이션 대상 채널 (GP 학습 채널과 동일. 기구학 채널은 진단용이라 제외).
CHANNELS = ("v_y", "gamma")

# 표준정규 양측 구간의 z 임계값. 커버리지 명목수준 -> z.
_Z_CRIT = {0.68: 0.9944578832097535, 0.95: 1.959963984540054}


def disturbance_to_residual_ca(vehicle: VehicleConfig, dt_ctrl: float) -> ca.Function:
    """G(x,u,vx,kappa) -> (4,2): 연속 외란 d 가 1스텝 이산 상태차이에 주는 감도.

    KF 의 DisturbanceStepModel 과 **같은 주입 경로**를 미분한다 (연속 우변에 d 를
    더한 뒤 RK4). 복사가 아니라 같은 형태를 재현해야 환산이 의미를 갖는다.
    명목이 선형이므로 G 는 x,u,kappa 와 무관하고 vx 에만 의존하지만, 실차(비선형)
    전환 시에도 그대로 쓰이도록 전체 작동점을 인자로 받는다.
    """
    f_nom = dynamics_ca(vehicle)
    x = ca.SX.sym("x", 4)
    u = ca.SX.sym("u")
    vx = ca.SX.sym("vx")
    kappa = ca.SX.sym("kappa")
    d = ca.SX.sym("d", 2)

    def f(xx, uu, vv, kk):
        return f_nom(xx, uu, vv, kk) + ca.vertcat(d[0], d[1], 0.0, 0.0)

    x_next = rk4_step(f, x, u, (vx, kappa), float(dt_ctrl))
    return ca.Function("G_d2r", [x, u, vx, kappa], [ca.jacobian(x_next, d)])


def kf_residual_prediction(log: dict[str, np.ndarray], vehicle: VehicleConfig,
                           dt_ctrl: float) -> tuple[np.ndarray, np.ndarray]:
    """KF 로그를 1스텝 이산 잔차의 예측분포로 환산한다.

    반환: (mean (K,2), var (K,2)) — GP 의 (gp_mean, gp_var) 와 **같은 대상·같은 단위**.
    var 는 cov_KF 의 대각만 취한다 (채널별 구간 커버리지를 보기 위함).

    시간 정렬: runner 는 스텝 k 에서 update -> solve 순으로 돌고 d_hat/P 를 그 뒤에
    기록하므로, 로그의 k 번째 d_hat/P 는 스텝 k 의 예측에 실제로 쓰인 값이고
    residual[k] 는 같은 스텝 [k, k+1] 의 잔차다. 추가 shift 가 필요 없다.
    """
    for key in ("ekf_d_hat", "ekf_P_d_block"):
        if key not in log:
            raise KeyError(f"KF 캘리브레이션에 '{key}' 로그가 필요하다 (KF 케이스가 맞나?).")

    G_fn = disturbance_to_residual_ca(vehicle, dt_ctrl)
    x, delta, vx, kappa = log["x"], log["delta"], log["vx"], log["kappa"]
    d_hat, P_d = log["ekf_d_hat"], log["ekf_P_d_block"]

    K = x.shape[0]
    mean = np.zeros((K, 2))
    var = np.zeros((K, 2))
    for k in range(K):
        G = np.array(G_fn(x[k], delta[k], vx[k], kappa[k]))[0:2, :]   # (2,2)
        mean[k] = G @ d_hat[k]
        var[k] = np.diag(G @ P_d[k] @ G.T)
    return mean, var


def calibration_metrics(r_true: np.ndarray, mean: np.ndarray, var: np.ndarray,
                        levels: tuple[float, ...] = (0.68, 0.95)) -> dict:
    """예측분포 (mean, var) 가 참 잔차 r_true 를 얼마나 옳게 덮는지 정량화한다.

    r_true, mean, var: (K, 2) — 채널 순서는 CHANNELS.
    반환: 채널별 + 전체(pooled) 지표 dict.

    지표:
      - z_mean / z_std : 표준화 잔차 z=(r-mu)/sigma 의 평균·표준편차.
                         잘 캘리브레이션되면 ~0, ~1. z_std < 1 이면 **과대추정**
                         (보수적), > 1 이면 **과소추정**(위험).
      - coverage[lv]   : |z| <= z_crit(lv) 인 비율. 명목 lv 와 비교한다.
      - nlpd           : 평균 negative log predictive density (낮을수록 좋음).
                         0.5*log(2*pi*sigma^2) + 0.5*z^2.
    """
    r_true = np.asarray(r_true, float)
    mean = np.asarray(mean, float)
    var = np.asarray(var, float)
    if not (r_true.shape == mean.shape == var.shape):
        raise ValueError(f"shape 불일치: r{r_true.shape} mu{mean.shape} var{var.shape}")
    if np.any(var <= 0.0):
        raise ValueError("사후분산에 0 이하 값이 있다. 분산 계산을 먼저 고쳐라.")

    sigma = np.sqrt(var)
    z = (r_true - mean) / sigma
    nlpd_pt = 0.5 * np.log(2.0 * np.pi * var) + 0.5 * z**2

    def _one(zc: np.ndarray, np_pt: np.ndarray) -> dict:
        out = {"z_mean": float(np.mean(zc)), "z_std": float(np.std(zc)),
               "nlpd": float(np.mean(np_pt))}
        out["coverage"] = {lv: float(np.mean(np.abs(zc) <= _Z_CRIT[lv])) for lv in levels}
        return out

    per_channel = {ch: _one(z[:, i], nlpd_pt[:, i]) for i, ch in enumerate(CHANNELS)}
    return {"per_channel": per_channel, "pooled": _one(z.ravel(), nlpd_pt.ravel()),
            "n": int(z.shape[0]), "levels": list(levels)}


def kf_innovation_consistency(log: dict[str, np.ndarray]) -> dict | None:
    """부수 지표 — KF 가 **자기 모델 안에서** 일관적인지 (GP 와의 비교 아님).

    표준 KF 캘리브레이션은 innovation 을 S = H P^- H^T + R_kf 로 정규화해 본다.
    대상이 '측정' 이라 GP(잔차)와 직접 비교할 수 없다. 그래서 head-to-head 지표가
    아니라 "KF 가 strawman 이 아니라 제대로 동작 중" 임을 보이는 근거로만 쓴다.

    현재 로그에는 S 가 없고 innovation 만 있으므로, 여기서는 innovation 의 백색성
    (lag-1 자기상관)만 낸다. 자기상관이 크면 필터가 정보를 덜 뽑아낸 것이다.
    """
    if "ekf_innovation" not in log:
        return None
    nu = np.asarray(log["ekf_innovation"], float)
    out = {}
    for j in range(nu.shape[1]):
        v = nu[:, j]
        v = v - v.mean()
        denom = float(np.sum(v * v))
        lag1 = float(np.sum(v[1:] * v[:-1]) / denom) if denom > 0 else float("nan")
        out[f"ch{j}"] = {"rms": float(np.sqrt(np.mean(nu[:, j] ** 2))), "lag1_autocorr": lag1}
    return out


def format_calibration(m: dict, title: str = "") -> str:
    """캘리브레이션 dict 를 사람이 읽는 문자열로."""
    lines = [f"[캘리브레이션] {title}".rstrip()]
    lines.append(f"  대상 = 1스텝 이산 잔차 r_k, n={m['n']}")
    for ch, e in m["per_channel"].items():
        cov = "  ".join(f"{int(lv*100)}%={e['coverage'][lv]:.1%}" for lv in m["levels"])
        lines.append(f"  {ch:6s} z_mean={e['z_mean']:+.3f} z_std={e['z_std']:.3f}  "
                     f"{cov}  NLPD={e['nlpd']:+.3f}")
    p = m["pooled"]
    cov = "  ".join(f"{int(lv*100)}%={p['coverage'][lv]:.1%}" for lv in m["levels"])
    lines.append(f"  {'pooled':6s} z_mean={p['z_mean']:+.3f} z_std={p['z_std']:.3f}  "
                 f"{cov}  NLPD={p['nlpd']:+.3f}")
    return "\n".join(lines)
