"""비선형 플랜트 자전거 모델 — 타이어 힘 법칙만 Fiala 로 교체.

선형판과의 **유일한** 차이는 타이어 힘이다 (선형 F=C*alpha -> Fiala 포화).
슬립각 정의와 **오차 기구학(e_psi, e_y 식)은 절대 건드리지 않는다.** 잔차의
출처를 타이어 비선형성 하나로 통제하기 위한 실험 설계다 (vehicle-model.md).

운동방정식 (vehicle-model.md 부호 규약에서 유도, 선형판과 동일 구조):
  alpha_f = delta - (v_y + a*gamma)/vx
  alpha_r =       - (v_y - b*gamma)/vx
  m*(v_y_dot + vx*gamma) = Fyf + Fyr
  Iz*gamma_dot           = a*Fyf - b*Fyr
  e_psi_dot = gamma - vx*kappa      # 불변
  e_y_dot   = v_y + vx*e_psi        # 불변

선형판은 Fyf=Cf*alpha_f, Fyr=Cr*alpha_r 이고, 여기서는 Fiala 로만 바뀐다.
"""
from __future__ import annotations

from typing import Callable

import casadi as ca
import numpy as np

from src.config import VehicleConfig
from src.models.tire import fiala_fy_ca, fiala_fy_np


def xdot_nl_np(cfg: VehicleConfig, x: np.ndarray, u: float, vx: float, kappa: float) -> np.ndarray:
    """비선형 플랜트 연속 우변 xdot (4,). numpy.

    x=[v_y,gamma,e_psi,e_y], u=delta[rad], vx[m/s], kappa[1/m].
    """
    cfg.check_vx(vx)
    mu = cfg.require_mu()
    x = np.asarray(x, dtype=float).reshape(4)
    v_y, gamma, e_psi, _e_y = x

    alpha_f = u - (v_y + cfg.a * gamma) / vx
    alpha_r = -(v_y - cfg.b * gamma) / vx
    Fyf = fiala_fy_np(alpha_f, cfg.Cf, mu, cfg.Fzf)
    Fyr = fiala_fy_np(alpha_r, cfg.Cr, mu, cfg.Fzr)

    vy_dot = (Fyf + Fyr) / cfg.m - vx * gamma
    gamma_dot = (cfg.a * Fyf - cfg.b * Fyr) / cfg.Iz
    e_psi_dot = gamma - vx * kappa      # 불변 기구학
    e_y_dot = v_y + vx * e_psi          # 불변 기구학
    return np.array([vy_dot, gamma_dot, e_psi_dot, e_y_dot], dtype=float)


def make_nonlinear_rhs_np(cfg: VehicleConfig) -> Callable:
    """비선형 플랜트 우변 클로저 f(x,u,vx,kappa)->xdot(4,). runner 주입용."""
    def f(x, u, vx, kappa):
        return xdot_nl_np(cfg, x, u, vx, kappa)
    return f


def dynamics_nl_ca(cfg: VehicleConfig) -> ca.Function:
    """비선형 플랜트 CasADi Function f(x[4],u,vx,kappa)->xdot[4].

    numpy 경로와 1e-10 이내 동일해야 한다 (게이트 대조). 심볼릭이므로 vx 특이점
    런타임 검사는 하지 않는다.
    """
    mu = cfg.require_mu()
    x = ca.SX.sym("x", 4)
    u = ca.SX.sym("u")
    vx = ca.SX.sym("vx")
    kappa = ca.SX.sym("kappa")
    v_y, gamma, e_psi = x[0], x[1], x[2]

    alpha_f = u - (v_y + cfg.a * gamma) / vx
    alpha_r = -(v_y - cfg.b * gamma) / vx
    Fyf = fiala_fy_ca(alpha_f, cfg.Cf, mu, cfg.Fzf)
    Fyr = fiala_fy_ca(alpha_r, cfg.Cr, mu, cfg.Fzr)

    vy_dot = (Fyf + Fyr) / cfg.m - vx * gamma
    gamma_dot = (cfg.a * Fyf - cfg.b * Fyr) / cfg.Iz
    e_psi_dot = gamma - vx * kappa
    e_y_dot = v_y + vx * e_psi
    xdot = ca.vertcat(vy_dot, gamma_dot, e_psi_dot, e_y_dot)
    return ca.Function("f_nl", [x, u, vx, kappa], [xdot],
                       ["x", "u", "vx", "kappa"], ["xdot"])
