"""명목 MPC — 보정 없음 (Part 1 케이스 1: mpc_only).

명목 이산 전이 StepModel 을 정의하고, MpcBase 에 주입해 컨트롤러를 만든다.
GP 케이스(mpc_gp)는 같은 MpcBase 에 다른 StepModel 을 주입할 뿐, 이 파일과
MpcBase 를 수정하지 않는다 (인터페이스 + 주입).
"""
from __future__ import annotations

import numpy as np

from src.config import MpcConfig, VehicleConfig
from src.control.mpc_base import MpcBase
from src.models.integrators import rk4_step
from src.models.linear_bicycle import dynamics_ca


class NominalStepModel:
    """명목 이산 전이: x_{k+1} = rk4_step(f_nom, x_k, u_k, (vx_k,kappa_k), dt).

    보정이 없으므로 추가 CasADi 파라미터가 없다 (extra_param_dim = 0).
    """
    extra_param_dim = 0

    def __init__(self, vehicle: VehicleConfig, dt_ctrl: float):
        self.dt = float(dt_ctrl)
        # 명목 연속 우변 CasADi Function (SX 로 빌드되지만 MX 그래프에서도 호출 가능).
        self._f_nom = dynamics_ca(vehicle)

    def step_sym(self, x, u, vx, kappa, p_extra):
        """RK4 1스텝. p_extra 는 명목에서 사용하지 않는다 (None)."""
        return rk4_step(self._f_nom, x, u, (vx, kappa), self.dt)

    def extra_param_values(self) -> np.ndarray:
        return np.empty(0, dtype=float)


def make_nominal_mpc(vehicle: VehicleConfig, cfg_mpc: MpcConfig, dt_ctrl: float) -> MpcBase:
    """명목 MPC 컨트롤러를 조립한다."""
    model = NominalStepModel(vehicle, dt_ctrl)
    return MpcBase(model, cfg_mpc, vehicle, dt_ctrl)
