"""적분기.

두 경로를 **의도적으로 다르게** 둔다 (.claude/rules/vehicle-model.md "적분기").
이 차이가 잔차의 noise floor 를 정의하며, 그 값은 숫자로 측정·보고되어야 한다.

- `rk4_step` — MPC 예측 전용. CasADi 심볼릭과 numpy 양쪽에서 동작한다.
  RK4 1스텝은 우변을 4회 평가한다.
- `plant_step` — 플랜트 전용 고정밀 경로. scipy solve_ivp(DOP853, rtol=1e-10,
  atol=1e-12). 아래 주석에 선택 이유를 남긴다.

우변 함수 `f` 는 시그니처 `f(x, u, vx, kappa) -> xdot` 를 따른다. `params = (vx, kappa)`.
제어 스텝 안에서 vx, kappa, u 는 ZOH 로 상수다.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.integrate import solve_ivp

from src.config import VehicleConfig
from src.models.linear_bicycle import xdot_np


def make_rhs_np(cfg: VehicleConfig) -> Callable:
    """numpy 우변 클로저 f(x, u, vx, kappa) -> xdot(4,) 를 만든다."""
    def f(x, u, vx, kappa):
        return xdot_np(cfg, x, u, vx, kappa)
    return f


def rk4_step(f: Callable, x, u, params: tuple, dt: float):
    """고전 RK4 1스텝. MPC 예측 전용.

    f: 우변 f(x, u, vx, kappa) -> xdot. numpy 배열 또는 CasADi SX 모두 지원한다
       (사칙연산만 사용하므로 타입에 무관하다).
    x: 상태(4,) 또는 SX(4). u: 입력(스칼라). params: (vx, kappa). dt: [s].
    반환: 다음 상태 (입력과 같은 타입).

    RK4 차수를 바꾸기 전에 사용자에게 물어라. GP 결합 시 스텝당 우변 평가 횟수가
    그대로 solve time 에 곱해진다 (vehicle-model.md).
    """
    vx, kappa = params
    k1 = f(x, u, vx, kappa)
    k2 = f(x + 0.5 * dt * k1, u, vx, kappa)
    k3 = f(x + 0.5 * dt * k2, u, vx, kappa)
    k4 = f(x + dt * k3, u, vx, kappa)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def plant_step(f_np: Callable, x: np.ndarray, u: float, params: tuple, dt: float) -> np.ndarray:
    """플랜트 전용 고정밀 1스텝 적분. numpy 전용.

    solve_ivp 의 DOP853(8차 explicit Runge-Kutta)을 rtol=1e-10, atol=1e-12 로 쓴다.
    이유: 이 명목 동역학은 매끄럽고 비-stiff 하므로 고차 explicit 이 가장 효율적이며,
    tight tolerance 로 적분해 플랜트 기준궤적을 RK4(dt_ctrl)보다 훨씬 정확하게 만든다.
    두 경로의 차이가 곧 잔차 noise floor 가 되므로, 기준이 충분히 정확해야 그 차이가
    'RK4 이산화 오차'만 담는다 (vehicle-model.md 적분기 절).

    f_np: numpy 우변 f(x, u, vx, kappa) -> xdot(4,). x: (4,). u: 스칼라. dt: [s].
    반환: 다음 상태 (4,).
    """
    vx, kappa = params
    x0 = np.asarray(x, dtype=float).reshape(4)
    sol = solve_ivp(
        lambda t, y: f_np(y, u, vx, kappa),
        (0.0, dt),
        x0,
        method="DOP853",
        rtol=1e-10,
        atol=1e-12,
    )
    if not sol.success:
        raise RuntimeError(f"plant_step 적분 실패: {sol.message}")
    return sol.y[:, -1]
