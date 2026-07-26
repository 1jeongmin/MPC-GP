"""GP 잔차 보정 MPC (Part 1 케이스 3) — 이산 결합 (RK4 밖).

**결합 형태 (Phase 7 진단으로 확정)**: GP 평균 mu_GP 를 **현재 작동점에서 한 번**
평가해 지평 전체에 **상수**로 이산 주입한다:
    x_{k+1} = rk4_step(f_nom) + B_d @ mu_hat,   mu_hat = mu_GP(z_now) 고정

왜 상수인가: mu_GP(z_k) 를 각 지평 스텝의 예측 상태에서 평가하면(상태의존 결합),
mean-only MPC 가 GP 가 부정확한 영역을 최적화에서 악용해 **불안정**해진다(격자 완벽
적합에도 -74%). 현재 작동점 상수 주입은 안정적이며 +12% 로 KF(+5.8%)를 이긴다.

왜 GP 가 KF 를 이기나: 둘 다 지평 상수 외란이지만, GP 는 상태->외란을 직접 학습해
현재 상태에서 즉시 정확하고, KF 는 시간축 필터라 지연된다. GP 는 모델 보정자,
KF 는 상태추정기라는 프레이밍(ekf-baseline.md)과 일치한다.

GP 잔차는 이산 상태차이 단위이므로 RK4 **밖**에서 더한다 (KF 연속 외란 RK4 안과 대비).
MpcBase / mpc_nominal 은 수정하지 않는다 (StepModel 주입 + 작동점 훅).
"""
from __future__ import annotations

import casadi as ca
import numpy as np

from src.config import MpcConfig, VehicleConfig
from src.control.mpc_base import MpcBase
from src.gp.casadi_export import build_mu_function, gp_param_vector
from src.gp.train_offline import TwoChannelGP
from src.models.integrators import rk4_step
from src.models.linear_bicycle import dynamics_ca


class GPStepModel:
    """이산 전이 = rk4_step(f_nom) + B_d @ mu_hat. mu_hat 은 현재 작동점 GP 평균(상수).

    extra_param_dim = 2 (mu_vy, mu_gamma). set_operating_point 이 현재 상태에서
    mu 를 계산해 저장하고, extra_param_values 가 이를 반환한다.
    추론은 casadi_export 의 CasADi 표현으로 한다 (제어 루프에서 torch 미사용).
    """
    extra_param_dim = 2

    def __init__(self, vehicle: VehicleConfig, dt_ctrl: float, gp: TwoChannelGP):
        self.dt = float(dt_ctrl)
        self._f_nom = dynamics_ca(vehicle)
        self.gp = gp                                  # 분산 로깅용 (numpy)
        M = len(gp.channels[0].Z)
        self._mu_func = build_mu_function(M)          # CasADi GP 평균 표현
        self._p = gp_param_vector(gp)                 # 주입 파라미터 (고정)
        self._mu = np.zeros(2)
        self._var = np.zeros(2)

    def set_operating_point(self, x0: np.ndarray, u_prev: float) -> None:
        """현재 작동점 z=[v_y, gamma, delta_prev] 에서 GP 평균·사후분산을 평가·저장.

        평균은 CasADi 표현으로(= NLP 에 주입되는 값과 정확히 동일한 경로),
        사후분산은 numpy GP 로 계산한다. 분산은 **제어에 쓰지 않지만**(mean-only,
        non-cautious) 매 스텝 반드시 로깅한다 — gp-residual.md 「사후분산은 제어에
        넣지 않되, 반드시 저장한다」. 이것이 Part 1 의 주 증거물이다.
        """
        z = np.array([x0[0], x0[1], u_prev])
        self._mu = np.array(self._mu_func(z, self._p)).reshape(2)
        self._var = np.asarray(self.gp.predict_var(z), float).reshape(2)

    def step_sym(self, x, u, vx, kappa, p_extra):
        correction = ca.vertcat(p_extra[0], p_extra[1], 0.0, 0.0)   # B_d @ mu_hat
        return rk4_step(self._f_nom, x, u, (vx, kappa), self.dt) + correction

    def extra_param_values(self) -> np.ndarray:
        return self._mu

    def step_log(self) -> dict[str, np.ndarray]:
        """MpcBase.step_log 훅 — 이번 스텝의 GP 평균·사후분산을 로그에 싣는다.

        단위: gp_mean/gp_var 는 **이산 잔차 공간**(v_y [m/s], gamma [rad/s]) 이다.
        표준화 공간이 아니다 (casadi_export 가 역표준화해 반환).
        """
        return {"gp_mean": self._mu.copy(), "gp_var": self._var.copy()}


def make_gp_mpc(vehicle: VehicleConfig, cfg_mpc: MpcConfig, dt_ctrl: float,
                gp: TwoChannelGP) -> MpcBase:
    """GP 보정 MPC 컨트롤러를 조립한다."""
    model = GPStepModel(vehicle, dt_ctrl, gp)
    return MpcBase(model, cfg_mpc, vehicle, dt_ctrl)
