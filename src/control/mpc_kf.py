"""증강 KF 결합 MPC — 외란 추정 d_hat 을 예측에 주입 (Part 1 케이스 2).

DisturbanceStepModel 은 KF 의 외란 추정 d_hat 을 **연속 외란**으로 명목 우변에
더한 뒤 RK4 로 적분한다. GP 의 이산 잔차 주입(RK4 밖)과 대비된다 — d 는 연속 rate
(m/s^2, rad/s^2) 단위이기 때문이다.

MpcBase / mpc_nominal 은 수정하지 않는다. StepModel 교체(주입)만으로 결합한다.
"""
from __future__ import annotations

import casadi as ca
import numpy as np

from src.config import MpcConfig, VehicleConfig
from src.control.mpc_base import MpcBase
from src.estimation.ekf import AugmentedKF
from src.models.integrators import rk4_step
from src.models.linear_bicycle import dynamics_ca


class DisturbanceStepModel:
    """이산 전이 = rk4_step(f_nom + d 를 v_y,gamma 우변에 더함).

    extra_param_dim=2 (d_vy, d_gamma). extra_param_values() 는 KF 의 현재 d_hat 를
    반환한다 — 온라인 갱신에도 NLP 재빌드 없이 값만 바뀐다.
    """
    extra_param_dim = 2

    def __init__(self, vehicle: VehicleConfig, dt_ctrl: float, estimator: AugmentedKF):
        self.dt = float(dt_ctrl)
        self._f_nom = dynamics_ca(vehicle)
        self.estimator = estimator

    def step_sym(self, x, u, vx, kappa, p_extra):
        """연속 외란 p_extra=[d_vy,d_gamma] 를 우변에 더하고 RK4 적분."""
        f_nom = self._f_nom

        def f(xx, uu, vv, kk):
            xd = f_nom(xx, uu, vv, kk)
            return xd + ca.vertcat(p_extra[0], p_extra[1], 0.0, 0.0)

        return rk4_step(f, x, u, (vx, kappa), self.dt)

    def extra_param_values(self) -> np.ndarray:
        return np.asarray(self.estimator.d_hat, float).reshape(2)


def make_kf_mpc(vehicle: VehicleConfig, cfg_mpc: MpcConfig, dt_ctrl: float,
                estimator: AugmentedKF) -> MpcBase:
    """KF 외란 주입 MPC 컨트롤러를 조립한다."""
    model = DisturbanceStepModel(vehicle, dt_ctrl, estimator)
    return MpcBase(model, cfg_mpc, vehicle, dt_ctrl)
