"""Phase 6 검증 — 증강 KF (Part 1 비교군).

- 가관측성: (A_aug, H) 가 가관측. 아니면 d_hat 이 안 잡힌다 -> 멈추고 보고.
- 잡음 없는 극한에서 d_hat 이 실제 (연속화) 잔차를 추종.
- KF 가 선형(야코비안 상수)임을 대조 -> KF≡EKF 전방호환 구조 검증.
- P 대각·innovation 이 매 스텝 로깅되는지.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from src.config import load_group
from src.control.mpc_base import build_step_function
from src.control.mpc_kf import make_kf_mpc
from src.control.mpc_nominal import NominalStepModel
from src.estimation.ekf import (MEAS_ROWS, augmented_process_ca,
                                 linear_measurement_ca, make_augmented_kf)
from src.models.integrators import make_rhs_np
from src.models.linear_bicycle import A_matrix
from src.models.nonlinear_bicycle import make_nonlinear_rhs_np
from src.path.reference import Reference
from src.sim.runner import run_closed_loop


@pytest.fixture
def sedan():
    return load_group("vehicle", "sedan")


@pytest.fixture
def ekf_cfg():
    return load_group("ekf", "default")


# --------------------------------------------------------------------------- #
# 가관측성                                                                      #
# --------------------------------------------------------------------------- #
def test_augmented_system_observable(sedan) -> None:
    """(A_aug, H) 가 모든 대표 속도에서 가관측 (rank 6)."""
    for vx in (5.0, 15.0, 25.0):
        A = A_matrix(sedan, vx)
        Aa = np.zeros((6, 6)); Aa[:4, :4] = A; Aa[0, 4] = 1.0; Aa[1, 5] = 1.0
        H = np.zeros((len(MEAS_ROWS), 6))
        for i, r in enumerate(MEAS_ROWS):
            H[i, r] = 1.0
        O = np.vstack([H @ np.linalg.matrix_power(Aa, i) for i in range(6)])
        assert np.linalg.matrix_rank(O) == 6, f"vx={vx} 에서 불가관측"


# --------------------------------------------------------------------------- #
# KF ≡ EKF (야코비안 상수) — 전방호환 구조 검증                                  #
# --------------------------------------------------------------------------- #
def test_kf_is_linear_constant_jacobian(sedan, ekf_cfg) -> None:
    """명목 선형이므로 예측·측정 야코비안이 상태 무관(상수)."""
    kf = make_augmented_kf(sedan, ekf_cfg, dt=0.02)
    assert kf.is_linear(), "야코비안이 상수가 아니다 (선형 명목인데 이상)"


# --------------------------------------------------------------------------- #
# 잡음 없는 극한에서 d_hat 이 실제 외란을 추종                                    #
# --------------------------------------------------------------------------- #
def test_dhat_tracks_true_disturbance_noiseless(sedan, ekf_cfg) -> None:
    """상수 참 외란을 준 선형 플랜트에서, 잡음 없는 KF 의 d_hat 이 그 외란으로 수렴.

    참 플랜트 = 명목 + 상수 외란 [d_vy*, d_gamma*] (연속). KF 는 이를 모르고 추정.
    측정 잡음 R->0 으로 두고, 충분히 적분하면 d_hat -> [d_vy*, d_gamma*].
    """
    dt = 0.02
    d_true = np.array([0.4, 0.03])   # 참 외란 (m/s^2, rad/s^2)
    f_nom = make_rhs_np(sedan)

    def plant_rhs(x, u, vx, kappa):
        xd = f_nom(x, u, vx, kappa)
        return xd + np.array([d_true[0], d_true[1], 0.0, 0.0])

    # 잡음 거의 0 인 KF (전상태 측정 4채널).
    cfg0 = replace(ekf_cfg, R_kf_diag=(1e-12, 1e-12, 1e-12, 1e-12))
    kf = make_augmented_kf(sedan, cfg0, dt)
    from src.models.integrators import plant_step
    rng = np.random.default_rng(0)

    x = np.zeros(4)
    u, vx, kappa = 0.01, 15.0, 0.0
    for k in range(1500):
        if k > 0:
            kf.predict(u, vx, kappa)
        y = kf.simulate_measurement(x, rng)
        kf.update(y)
        x = plant_step(plant_rhs, x, u, (vx, kappa), dt)

    assert np.allclose(kf.d_hat, d_true, atol=2e-2), (
        f"d_hat={kf.d_hat} 가 참 외란 {d_true} 로 수렴 안 함")


# --------------------------------------------------------------------------- #
# 폐루프 통합 + 로깅                                                            #
# --------------------------------------------------------------------------- #
def test_kf_beats_mpc_only(sedan, ekf_cfg) -> None:
    """핵심 게이트: 비선형 플랜트에서 MPC+KF 추종오차 < MPC-only.

    KF 는 MPC 를 돕는 기술이므로 단독 MPC 보다 나아야 한다. 나빠지면 설계 오류
    (부분관측·저 Q_d 등). 여기서 멈추고 원인을 보고하라.
    """
    from src.control.mpc_nominal import make_nominal_mpc
    from src.eval.metrics import compute_metrics
    mpc_cfg = load_group("mpc", "default")
    sim = replace(load_group("sim", "default"), duration=10.0)
    path = load_group("path", "single_curve")
    reference = Reference(path, sedan.vx_range)
    plant_nl = make_nonlinear_rhs_np(sedan)
    nominal_step = build_step_function(NominalStepModel(sedan, sim.dt_ctrl))
    NF = {"v_y": 9.57e-9}

    # MPC only
    ctl0 = make_nominal_mpc(sedan, mpc_cfg, sim.dt_ctrl)
    e0 = compute_metrics(run_closed_loop(ctl0, reference, plant_nl, nominal_step, sim),
                         sedan, sim.dt_ctrl, NF)["tracking"]["rms_e_y"]
    # MPC + KF
    kf = make_augmented_kf(sedan, ekf_cfg, sim.dt_ctrl)
    ctl1 = make_kf_mpc(sedan, mpc_cfg, sim.dt_ctrl, kf)
    log = run_closed_loop(ctl1, reference, plant_nl, nominal_step, sim,
                          estimator=kf, rng=np.random.default_rng(0))
    e1 = compute_metrics(log, sedan, sim.dt_ctrl, NF)["tracking"]["rms_e_y"]

    assert e1 < e0, f"MPC+KF({e1:.3e}) 가 MPC-only({e0:.3e}) 보다 나쁘다"

    # 로깅 확인.
    for key in ("ekf_P_diag", "ekf_innovation", "ekf_d_hat", "ekf_x_hat"):
        assert key in log, f"{key} 가 로깅되지 않았다"
    assert log["ekf_P_diag"].shape[1] == 6 and log["ekf_d_hat"].shape[1] == 2
    assert np.max(np.abs(log["ekf_d_hat"])) > 1e-2
    assert np.all(log["converged"].astype(bool))


# --------------------------------------------------------------------------- #
# 혁신일관성 NIS — Q/R 튜닝의 부 지표 (2026-07-30)                               #
# --------------------------------------------------------------------------- #
def test_nis_matches_manual_formula(sedan, ekf_cfg) -> None:
    """NIS = nu^T S^-1 nu 가 수기 계산과 일치.

    가장 틀리기 쉬운 지점은 **어느 P 로 S 를 만드는가**다. S 는 갱신 **전**(prior)
    공분산으로 만들어야 한다 — 갱신 후 P 를 쓰면 NIS 가 체계적으로 커진다.
    """
    dt = 0.02
    kf = make_augmented_kf(sedan, ekf_cfg, dt)
    kf.predict(0.01, 15.0, 0.0)

    P_prior = kf.P.copy()
    x_prior = kf.x_hat.copy()
    y = np.array([0.05, 0.01, 0.002, 0.03])

    H = np.zeros((len(MEAS_ROWS), 6))
    for i, r in enumerate(MEAS_ROWS):
        H[i, r] = 1.0
    nu_expect = y - H @ x_prior
    S = H @ P_prior @ H.T + np.diag(ekf_cfg.R_kf_diag)
    nis_expect = float(nu_expect @ np.linalg.solve(S, nu_expect))

    kf.update(y)
    assert np.allclose(kf.innovation, nu_expect, atol=1e-12)
    assert abs(kf.nis - nis_expect) <= 1e-9 * max(1.0, abs(nis_expect)), (
        f"nis={kf.nis} != 수기값 {nis_expect}")


def test_nis_consistent_when_model_and_noise_match(sedan) -> None:
    """모델·잡음이 정확히 맞는 필터의 평균 NIS 가 n_meas(=4) 근방.

    이것이 NIS 를 "P 가 옳은 크기인가" 의 판정 기준으로 쓸 수 있는 근거다
    (일관된 필터에서 NIS ~ chi^2_{n_meas}, 평균 = n_meas). 플랜트를 **명목 선형**
    으로 두어 모델오차를 0 으로 만들고, 측정잡음만 R 과 일치시킨다.
    """
    from src.estimation.state_estimator import make_state_kf
    from src.models.integrators import plant_step

    dt = 0.02
    std = np.array([0.03, 0.002, 0.003, 0.03])       # 주입 잡음 = R 의 제곱근
    cfg = replace(load_group("state_kf", "default"),
                  Q_diag=(1e-10, 1e-10, 1e-10, 1e-10),   # 모델오차 0 -> Q ~ 0 이 정합
                  R_diag=tuple(std**2))
    kf = make_state_kf(sedan, cfg, dt)
    plant = make_rhs_np(sedan)                        # 명목과 **동일** (모델오차 없음)
    rng = np.random.default_rng(0)

    x = np.zeros(4)
    u, vx, kappa = 0.01, 15.0, 0.0
    nis_hist = []
    for k in range(3000):
        y = x + rng.normal(scale=std)
        kf.filter_step(y, u, vx, kappa, first=(k == 0))
        nis_hist.append(kf.nis)
        x = plant_step(plant, x, u, (vx, kappa), dt)

    mean_nis = float(np.mean(nis_hist[500:]))         # 초기 과도 제외
    assert 3.0 <= mean_nis <= 5.5, (
        f"정합 필터의 평균 NIS={mean_nis:.3f} 가 n_meas=4 근방이 아니다 "
        "(NIS 수식이나 S 구성이 틀렸을 가능성)")
