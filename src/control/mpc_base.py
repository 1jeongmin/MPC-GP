"""MPC 공통 골격 — multiple shooting + IPOPT (.claude/rules/mpc-solver.md).

설계 핵심 (사용자 확정 2026-07-25):
- 컨트롤러는 **이산 전이 StepModel** 을 주입받는다. 연속 우변 f 가 아니다.
  명목:  x_{k+1} = rk4_step(f_nom, x_k, u_k, (vx_k,kappa_k), dt)
  GP:    x_{k+1} = rk4_step(...) + B_d @ mu_GP(z_k)   (Phase 7)
  GP 잔차는 상태 차이 단위이므로 RK4 바깥에서 더한다 (gp-residual.md 이산 결합).
- GP 파라미터(Z/alpha/hyper)는 StepModel.extra_param_dim 크기의 CasADi 파라미터
  블록으로 P 에 실린다. 온라인 갱신 시에도 값만 바꿔 NLP 를 재빌드하지 않는다.
- 케이스 분기(if case==...) 없이 StepModel 교체만으로 명목/GP 를 오간다.

파라미터 P 레이아웃: [x0(4), vx(N), kappa(N), u_prev(1), p_extra(extra_dim)].
"""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Protocol

import casadi as ca
import numpy as np

from src.config import MpcConfig, VehicleConfig


@dataclass(frozen=True)
class Preview:
    """MPC 지평에 채울 외생신호 프리뷰.

    kappa: (N,) [1/m] 곡률, vx: (N,) [m/s] 종속도. 둘 다 길이 N 이어야 한다.
    """
    kappa: np.ndarray
    vx: np.ndarray

    def __post_init__(self) -> None:
        if self.kappa.shape != self.vx.shape or self.kappa.ndim != 1:
            raise ValueError(f"kappa {self.kappa.shape} 와 vx {self.vx.shape} 는 같은 1D 여야 한다.")

    def __len__(self) -> int:
        return int(self.kappa.shape[0])

    @classmethod
    def from_arrays(cls, kappa: np.ndarray, vx: np.ndarray) -> "Preview":
        return cls(np.asarray(kappa, float).reshape(-1), np.asarray(vx, float).reshape(-1))


class StepModel(Protocol):
    """이산 1스텝 전이 모델 인터페이스. 명목/GP 가 이 프로토콜을 구현한다.

    extra_param_dim: 이 모델이 P 에 요구하는 추가 CasADi 파라미터 개수 (명목 0, GP>0).
    step_sym: 심볼릭 1스텝 전이. p_extra 는 (extra_param_dim,) 심볼 (0이면 None).
    extra_param_values: solve 시점에 P 에 채울 수치 (명목은 빈 배열).
    """
    extra_param_dim: int

    def step_sym(self, x, u, vx, kappa, p_extra):  # -> x_next (SX/MX)
        ...

    def extra_param_values(self) -> np.ndarray:
        ...


@dataclass
class SolveInfo:
    """solve() 부가 정보. Part 2 실시간성 지표의 원천."""
    solve_time: float          # [s] wall-clock
    iterations: int | None     # IPOPT iteration 수
    converged: bool            # 수렴 성공 여부
    fallback_used: bool        # 실패 시 이전 해 shift 사용 여부
    cost: float                # 최적 비용 (실패 시 nan)
    X: np.ndarray              # (4, N+1) 예측 상태 궤적
    U: np.ndarray              # (1, N) 예측 입력 시퀀스


def build_step_function(model: StepModel) -> ca.Function:
    """StepModel 의 심볼릭 step_sym 을 수치 CasADi Function 으로 감싼다.

    extra_param_dim == 0 인 모델(명목) 전용. 잔차 계산에서 MPC 예측과 **동일한**
    이산 전이를 재사용하기 위한 것이다 (복사 금지). 반환 F(x[4],u,vx,kappa) -> x_next[4].
    """
    if model.extra_param_dim != 0:
        raise ValueError("build_step_function 은 extra_param_dim==0 모델 전용이다.")
    x = ca.SX.sym("x", 4)
    u = ca.SX.sym("u")
    vx = ca.SX.sym("vx")
    kappa = ca.SX.sym("kappa")
    x_next = model.step_sym(x, u, vx, kappa, None)
    return ca.Function("F_step", [x, u, vx, kappa], [x_next])


def _shift(arr: np.ndarray) -> np.ndarray:
    """warm start / fallback 용 한 스텝 shift (마지막 열 반복)."""
    return np.concatenate([arr[:, 1:], arr[:, -1:]], axis=1)


class MpcBase:
    """multiple shooting NMPC. StepModel 주입으로 명목/GP 를 겸한다.

    NLP 구조는 생성 시 한 번만 빌드하고, solve 마다 파라미터 값만 바꿔 재사용한다.
    """

    def __init__(self, model: StepModel, cfg_mpc: MpcConfig, vehicle: VehicleConfig,
                 dt_ctrl: float):
        self.model = model
        self.cfg = cfg_mpc
        self.vehicle = vehicle
        self.dt = float(dt_ctrl)
        self.N = cfg_mpc.N
        self.extra_dim = int(model.extra_param_dim)

        N = self.N
        opti = ca.Opti()
        X = opti.variable(4, N + 1)      # 상태 궤적
        U = opti.variable(1, N)          # 입력 시퀀스

        # 파라미터 P (레이아웃은 위 docstring).
        p_x0 = opti.parameter(4)
        p_vx = opti.parameter(N)
        p_kappa = opti.parameter(N)
        p_uprev = opti.parameter(1)
        p_extra = opti.parameter(self.extra_dim) if self.extra_dim > 0 else None

        delta_max = vehicle.delta_max
        drate = vehicle.delta_rate_max * self.dt   # 스텝당 조향 변화 한계

        # 초기조건.
        opti.subject_to(X[:, 0] == p_x0)

        J = 0
        for k in range(N):
            xk = X[:, k]
            uk = U[0, k]
            # 이산 전이 (StepModel 주입). vx_k, kappa_k 는 P 에서.
            x_next = self.model.step_sym(xk, uk, p_vx[k], p_kappa[k], p_extra)
            opti.subject_to(X[:, k + 1] == x_next)

            # 입력 제약.
            opti.subject_to(opti.bounded(-delta_max, uk, delta_max))
            u_prev_k = p_uprev if k == 0 else U[0, k - 1]
            opti.subject_to(opti.bounded(-drate, uk - u_prev_k, drate))

            # 스테이지 비용 (x_ref/u_ref 없음 — kappa 프리뷰가 필요한 요레이트를 만든다).
            vy, gamma, epsi, ey = xk[0], xk[1], xk[2], xk[3]
            J += (cfg_mpc.W_ey * ey**2 + cfg_mpc.W_epsi * epsi**2
                  + cfg_mpc.W_vy * vy**2 + cfg_mpc.W_gamma * gamma**2
                  + cfg_mpc.R_delta * uk**2
                  + cfg_mpc.R_ddelta * (uk - u_prev_k)**2)

        # 종단항 (추종오차만).
        J += cfg_mpc.Wf_ey * X[3, N]**2 + cfg_mpc.Wf_epsi * X[2, N]**2
        opti.minimize(J)

        opti.solver("ipopt", {"print_time": 0}, {
            "max_iter": cfg_mpc.ipopt_max_iter,
            "tol": cfg_mpc.ipopt_tol,
            "print_level": cfg_mpc.ipopt_print_level,
            "sb": "yes",   # 시작 배너 억제 (결과에 무관한 표시 옵션)
        })

        # 핸들 보관.
        self.opti = opti
        self._X, self._U, self._J = X, U, J
        self._p_x0, self._p_vx, self._p_kappa = p_x0, p_vx, p_kappa
        self._p_uprev, self._p_extra = p_uprev, p_extra

        # warm start 상태.
        self._Xprev: np.ndarray | None = None
        self._Uprev: np.ndarray | None = None

    def reset(self) -> None:
        """warm start 이력 초기화 (새 시나리오 시작 시)."""
        self._Xprev = None
        self._Uprev = None

    def solve(self, x0: np.ndarray, preview: Preview, u_prev: float) -> tuple[float, SolveInfo]:
        """1 스텝 MPC 를 풀어 (delta 명령, SolveInfo) 반환.

        x0: (4,) 현재 상태. preview: 길이 N. u_prev: 직전 delta [rad].
        수렴 실패는 조용히 넘기지 않는다 — converged=False 로 표시하고 이전 해를
        shift 한 fallback 을 쓴다 (mpc-solver.md 수렴 실패 처리).
        """
        if len(preview) != self.N:
            raise ValueError(f"preview 길이 {len(preview)} != N {self.N}")
        x0 = np.asarray(x0, float).reshape(4)

        self.opti.set_value(self._p_x0, x0)
        self.opti.set_value(self._p_vx, preview.vx)
        self.opti.set_value(self._p_kappa, preview.kappa)
        self.opti.set_value(self._p_uprev, float(u_prev))
        if self.extra_dim > 0:
            vals = np.asarray(self.model.extra_param_values(), float).reshape(-1)
            if vals.shape[0] != self.extra_dim:
                raise ValueError(f"extra_param_values 크기 {vals.shape[0]} != {self.extra_dim}")
            self.opti.set_value(self._p_extra, vals)

        # warm start (이전 해 shift, 없으면 x0 반복 / 0).
        if self._Xprev is not None:
            Xg, Ug = _shift(self._Xprev), _shift(self._Uprev)
        else:
            Xg = np.tile(x0.reshape(4, 1), (1, self.N + 1))
            Ug = np.zeros((1, self.N))
        self.opti.set_initial(self._X, Xg)
        self.opti.set_initial(self._U, Ug)

        t0 = perf_counter()
        try:
            sol = self.opti.solve()
            Xs = np.array(sol.value(self._X)).reshape(4, self.N + 1)
            Us = np.array(sol.value(self._U)).reshape(1, self.N)
            cost = float(sol.value(self._J))
            converged, fallback = True, False
        except RuntimeError:
            # IPOPT 실패 (예: max_iter 초과). fallback = 이전 해 shift.
            converged = False
            fallback = True
            if self._Xprev is not None:
                Xs, Us = _shift(self._Xprev), _shift(self._Uprev)
            else:
                Xs = np.tile(x0.reshape(4, 1), (1, self.N + 1))
                Us = np.full((1, self.N), float(u_prev))
            cost = float("nan")
        solve_time = perf_counter() - t0

        try:
            iters = int(self.opti.stats().get("iter_count"))
        except (RuntimeError, TypeError):
            iters = None

        # 다음 warm start 를 위해 해 보관.
        self._Xprev, self._Uprev = Xs, Us

        u_cmd = float(Us[0, 0])
        info = SolveInfo(solve_time=solve_time, iterations=iters, converged=converged,
                         fallback_used=fallback, cost=cost, X=Xs, U=Us)
        return u_cmd, info
