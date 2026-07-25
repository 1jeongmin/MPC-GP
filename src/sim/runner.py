"""다중레이트 폐루프 시뮬레이터 (.claude/rules/sim-experiment.md).

- 플랜트 고정밀 적분(plant_step), MPC 예측 RK4(dt_ctrl). 이 차이가 잔차 noise floor.
- 플랜트 모델은 **주입**한다 (plant_rhs 콜러블). 선형↔비선형 교체가 runner 수정
  없이 된다.
- 제어 스텝 경계에서 vx, kappa 를 ZOH 로 상수화한다. MPC 예측이 스텝마다 vx_k,
  kappa_k 를 상수로 쓰는 것과 일치시켜, 모델 완전일치 잔차가 순수 이산화 오차
  (noise floor)만 담게 한다.

잔차 정의 (gp-residual.md 첫 절):
    r_k = x_{k+1}^plant - f_nom_d(x_k, u_k, vx_k, kappa_k)
f_nom_d 는 MPC 가 예측에 쓰는 것과 **같은** 이산 전이여야 한다. 여기서는
nominal_step_fn 을 주입받아 그대로 호출한다 (복사 금지).
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from src.config import SimConfig
from src.control.mpc_base import MpcBase, Preview
from src.models.integrators import plant_step
from src.path.reference import Reference


def run_closed_loop(
    controller: MpcBase,
    reference: Reference,
    plant_rhs: Callable,
    nominal_step_fn: Callable,
    sim_cfg: SimConfig,
    x0: np.ndarray | None = None,
    stop_margin: float = 5.0,
) -> dict[str, np.ndarray]:
    """다중레이트 폐루프를 돌리고 로그(배열 dict)를 반환한다.

    controller: MpcBase (N, dt 를 여기서 읽는다).
    reference: Reference (곡률·속도 프로파일, 프리뷰).
    plant_rhs: f(x, u, vx, kappa) -> xdot (numpy). 고정밀 적분 대상. 주입.
    nominal_step_fn: (x, u, vx, kappa) -> x_next. 잔차용 명목 이산 전이.
                     MPC 예측과 동일한 함수여야 한다.
    x0: 초기 상태 (4,). 기본 0 (경로 위, 오차 0).
    stop_margin: s 가 경로 끝 + 이 값[m] 을 넘으면 종료 (직선 크루즈 낭비 방지).
    """
    from src.sim.logger import Log

    N, dt = controller.N, controller.dt
    n_steps = int(round(sim_cfg.duration / dt))
    x = np.zeros(4) if x0 is None else np.asarray(x0, float).reshape(4)
    s = 0.0
    u_prev = 0.0
    controller.reset()
    log = Log()

    for k in range(n_steps):
        kappa_arr, vx_arr = reference.get_preview(s, N, dt)
        preview = Preview.from_arrays(kappa_arr, vx_arr)
        vx_now, kappa_now = float(vx_arr[0]), float(kappa_arr[0])

        u, info = controller.solve(x, preview, u_prev)

        # 잔차: 명목 이산 1스텝 예측 (MPC 예측과 같은 함수).
        x_nom_next = np.array(nominal_step_fn(x, u, vx_now, kappa_now)).reshape(4)
        # 플랜트 전진 (고정밀, u/vx/kappa ZOH).
        x_plant_next = plant_step(plant_rhs, x, u, (vx_now, kappa_now), dt)
        residual = x_plant_next - x_nom_next

        log.append(
            t=k * dt, x=x.copy(), delta=u, vx=vx_now, kappa=kappa_now, s=s,
            solve_time=info.solve_time, ipopt_iter=info.iterations,
            converged=info.converged, fallback=info.fallback_used,
            residual=residual,
        )

        x = x_plant_next
        s += vx_now * dt
        u_prev = u
        if s > reference.total_length + stop_margin:
            break

    return log.finalize()
