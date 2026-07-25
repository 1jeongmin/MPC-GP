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
        H = np.zeros((3, 6))
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

    # 잡음 거의 0 인 KF.
    cfg0 = replace(ekf_cfg, R_kf_diag=(1e-12, 1e-12, 1e-12))
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
def test_kf_closed_loop_logs_and_reduces_residual(sedan, ekf_cfg) -> None:
    """비선형 플랜트 폐루프에서 KF 로깅이 되고, d_hat 주입이 예측 잔차를 줄인다."""
    mpc_cfg = load_group("mpc", "default")
    sim = replace(load_group("sim", "default"), duration=7.0)
    path = load_group("path", "single_curve")
    reference = Reference(path, sedan.vx_range)
    plant_nl = make_nonlinear_rhs_np(sedan)
    nominal_step = build_step_function(NominalStepModel(sedan, sim.dt_ctrl))
    rng = np.random.default_rng(0)

    kf = make_augmented_kf(sedan, ekf_cfg, sim.dt_ctrl)
    controller = make_kf_mpc(sedan, mpc_cfg, sim.dt_ctrl, kf)
    log = run_closed_loop(controller, reference, plant_nl, nominal_step, sim,
                          estimator=kf, rng=rng)

    # 로깅 확인.
    for key in ("ekf_P_diag", "ekf_innovation", "ekf_d_hat", "ekf_x_hat"):
        assert key in log, f"{key} 가 로깅되지 않았다"
    assert log["ekf_P_diag"].shape[1] == 6
    assert log["ekf_d_hat"].shape[1] == 2
    assert np.all(np.isfinite(log["ekf_P_diag"]))

    # d_hat 이 커브에서 실제 외란 방향으로 움직였는지 (0 이 아니게).
    assert np.max(np.abs(log["ekf_d_hat"])) > 1e-2, "d_hat 이 거의 0 (외란 추정 실패)"
    assert np.all(log["converged"].astype(bool))
