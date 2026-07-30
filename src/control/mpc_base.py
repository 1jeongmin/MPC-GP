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
    # (1, n_blk) 예측 입력 — **블록당 1개**다 (스텝당 1개가 아니다).
    # 블로킹이 없으면 n_blk == N 이라 종전과 같다. 스텝 단위로 펼치려면
    # `MpcBase.u_per_step(U)` 를 쓰라. U[0, 0] 은 항상 실제로 적용되는 u_0 다
    # (첫 블록 크기가 1 로 강제되므로).
    U: np.ndarray


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


def _shift_blocked(U: np.ndarray, blk: tuple[int, ...], n_blk: int) -> np.ndarray:
    """블록 입력 (1, n_blk) 을 **한 스텝** shift 한다.

    수신 지평은 스텝 단위로 전진하지만 블록 구조는 고정이다. 따라서 블록 배열을
    한 열 밀면 (블록 크기가 다르므로) 시간축이 어긋난다. 스텝 단위로 펼쳐 shift 한
    뒤 블록별 평균으로 다시 접는다.

    블로킹이 없으면 (전 블록 크기 1) 이 함수는 `_shift` 와 동일한 결과를 준다.
    """
    per_step = U[0, list(blk)]                          # (N,) 스텝별로 펼치기
    per_step = np.concatenate([per_step[1:], per_step[-1:]])   # 한 스텝 shift
    out = np.zeros((1, n_blk))
    counts = np.zeros(n_blk)
    np.add.at(out[0], np.asarray(blk), per_step)
    np.add.at(counts, np.asarray(blk), 1.0)
    out[0] /= counts
    return out


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
        # 입력 블로킹 (제어 호라이즌 < 예측 호라이즌). config 가 비면 전 스텝 자유 =
        # 종전 동작과 **완전히 동일**하다 (아래 span=1 경로가 옛 식과 일치).
        self.u_sizes = cfg_mpc.u_block_sizes()      # 블록별 크기 [스텝]
        self.u_blk = cfg_mpc.u_block_of_step()      # 스텝 k -> 블록 인덱스
        n_blk = len(self.u_sizes)
        self.n_blk = n_blk

        opti = ca.Opti()
        X = opti.variable(4, N + 1)      # 상태 궤적
        U = opti.variable(1, n_blk)      # 입력 결정변수 (블록당 1개)

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

        # 입력 크기 제약은 **블록당 한 번**. 스텝마다 걸면 같은 변수에 동일한 제약이
        # 중복 생성되어 NLP 행만 늘어난다.
        for b in range(n_blk):
            opti.subject_to(opti.bounded(-delta_max, U[0, b], delta_max))

        # 조향 변화율 제약·비용은 **블록 경계에서만** 건다. 블록 내부는 u_k = u_{k-1}
        # 이므로 차이가 항등적으로 0 이고, 제약을 걸면 자명한 행이 쌓인다.
        #
        # 경계에서 허용하는 변화량 = drate * span,  span = (앞블록 크기 + 뒷블록 크기)/2.
        # 근거: 블록 표현은 입력 궤적의 **파라미터화**이고, 두 블록의 대표 시점을 각
        # 블록의 중점으로 보면 그 사이 경과 스텝 수가 span 이다. 실제 액추에이터는 그
        # 시간 동안 연속적으로 움직일 수 있으므로 drate*span 이 물리에 맞는 상한이다.
        # 경계마다 drate 로 조이면 꼬리의 도달 가능한 입력 변화가 블록 크기만큼 줄어들어
        # (10스텝 블록이면 1/10) 필요한 조향을 표현조차 못 하게 된다.
        #
        # **u_0 만은 예외 없이 정확히 drate 다** — 첫 블록 크기를 1 로 강제했고(config
        # 검증) p_uprev 와의 span 이 (1+1)/2 = 1 이므로. 플랜트에 실제로 적용되는 입력은
        # u_0 뿐이니 하드 액추에이터 한계를 지키는 스텝은 이것 하나로 충분하다.
        #
        # 블로킹이 없으면 전 스텝이 경계이고 span=1 이라 옛 식과 정확히 같아진다.
        J = 0
        for k in range(N):
            xk = X[:, k]
            uk = U[0, self.u_blk[k]]
            # 이산 전이 (StepModel 주입). vx_k, kappa_k 는 P 에서.
            x_next = self.model.step_sym(xk, uk, p_vx[k], p_kappa[k], p_extra)
            opti.subject_to(X[:, k + 1] == x_next)

            if k == 0:
                u_prev_k = p_uprev
                span = 0.5 * (1 + self.u_sizes[0])
            elif self.u_blk[k] != self.u_blk[k - 1]:
                b = self.u_blk[k]
                u_prev_k = U[0, b - 1]
                span = 0.5 * (self.u_sizes[b - 1] + self.u_sizes[b])
            else:
                u_prev_k = None            # 블록 내부 — 차이가 항등적으로 0
                span = None

            if u_prev_k is not None:
                opti.subject_to(opti.bounded(-drate * span, uk - u_prev_k, drate * span))

            # 스테이지 비용. gamma 만 참조 상대, 나머지는 절대값 (mpc-solver.md 예외 규정).
            #
            # 왜 gamma 만 빼는가: 곡선에서 '완벽추종'에 필요한 요레이트는 0 이 아니라
            # vx*kappa 다. gamma^2 를 벌주면 목적함수가 동역학 제약과 반대 방향을 지시해
            # e_y 에 정상편차가 남는다 (2026-07-27 진단: 일정곡률 편차 3.35e-3 -> 6.87e-4 m).
            # 이 참조는 **순수 기구학(e_psi_dot = gamma - vx*kappa)** 이라 차량 파라미터가
            # 들어가지 않는다 — 비선형 플랜트에서도 오차 0.02%. 따라서 선형모델 오차를
            # 목적함수에 두 번째로 들여오지 않는다 (CLAUDE.md 잔차 출처 통제).
            # v_y, delta 는 참조가 모델 의존(각각 -207%, +2.7% 오차)이라 여기서 제외했다.
            vy, gamma, epsi, ey = xk[0], xk[1], xk[2], xk[3]
            gamma_ref = p_vx[k] * p_kappa[k]      # 이미 P 에 있는 프리뷰 — 새 파라미터 없음
            J += (cfg_mpc.W_ey * ey**2 + cfg_mpc.W_epsi * epsi**2
                  + cfg_mpc.W_vy * vy**2 + cfg_mpc.W_gamma * (gamma - gamma_ref)**2
                  + cfg_mpc.R_delta * uk**2)
            if u_prev_k is not None:
                # 평활화 항은 **span 으로 나눈다.** 원식은 스텝당 차분 span 개의 합인데,
                # 블로킹하면 그 변화가 한 번의 점프로 뭉친다. 나누지 않으면 같은 총
                # 변화량에 대해 penalty 가 span 배 과대평가되어(4스텝 블록이면 4배)
                # 평활화 가중이 조용히 세진다. span=1 이면 옛 식과 동일.
                J += cfg_mpc.R_ddelta * (uk - u_prev_k)**2 / span

        # 종단항 (추종오차만).
        J += cfg_mpc.Wf_ey * X[3, N]**2 + cfg_mpc.Wf_epsi * X[2, N]**2
        opti.minimize(J)

        # IPOPT 옵션. config 가 단일 소스다 (하드코딩 금지).
        # ipopt_extra 는 warm_start_init_point 등 이름 없는 옵션의 통로 —
        # 아래 기본값보다 나중에 병합되므로 config 가 sb 까지 덮어쓸 수 있다.
        ipopt_opts = {
            "max_iter": cfg_mpc.ipopt_max_iter,
            "tol": cfg_mpc.ipopt_tol,
            "print_level": cfg_mpc.ipopt_print_level,
            "sb": "yes",   # 시작 배너 억제 (결과에 무관한 표시 옵션)
        }
        ipopt_opts.update(dict(cfg_mpc.ipopt_extra))
        opti.solver("ipopt", {"print_time": 0}, ipopt_opts)

        # 핸들 보관.
        self.opti = opti
        self._X, self._U, self._J = X, U, J
        self._p_x0, self._p_vx, self._p_kappa = p_x0, p_vx, p_kappa
        self._p_uprev, self._p_extra = p_uprev, p_extra

        # warm start 상태.
        self._Xprev: np.ndarray | None = None
        self._Uprev: np.ndarray | None = None

    def u_per_step(self, U: np.ndarray) -> np.ndarray:
        """블록 입력 (1, n_blk) 을 스텝 단위 (N,) 로 펼친다.

        블로킹이 없으면 `U[0, :]` 과 같다. 제약 검증·플롯처럼 스텝 단위가 필요한
        곳에서 쓴다 (블록 값을 그대로 diff 하면 경계 span 을 무시하게 된다).
        """
        return np.asarray(U, float)[0, list(self.u_blk)]

    def boundary_spans(self) -> list[tuple[int, float]]:
        """(스텝 k, span) 목록 — 입력이 실제로 변할 수 있는 지점과 그 허용 배율.

        k=0 은 p_uprev 대비 경계다. rate 한계는 스텝 k 에서 `drate * span`.
        """
        out = [(0, 0.5 * (1 + self.u_sizes[0]))]
        for k in range(1, self.N):
            if self.u_blk[k] != self.u_blk[k - 1]:
                b = self.u_blk[k]
                out.append((k, 0.5 * (self.u_sizes[b - 1] + self.u_sizes[b])))
        return out

    def reset(self) -> None:
        """warm start 이력 초기화 (새 시나리오 시작 시)."""
        self._Xprev = None
        self._Uprev = None

    def step_log(self) -> dict[str, np.ndarray] | None:
        """이번 스텝에 모델이 남길 추가 로그. 없으면 None.

        runner 가 케이스를 몰라도 되게 하는 훅이다 (estimator 훅과 대칭 구조).
        StepModel 이 `step_log()` 를 구현하면 그 dict 가 그대로 로그 행에 합류한다.
        GP 케이스는 여기로 gp_mean/gp_var 를 내보낸다 — 사후분산은 제어에 쓰지 않지만
        **반드시 저장**해야 하는 논문의 주 증거물이다 (gp-residual.md).
        """
        fn = getattr(self.model, "step_log", None)
        return fn() if fn is not None else None

    def solve(self, x0: np.ndarray, preview: Preview, u_prev: float) -> tuple[float, SolveInfo]:
        """1 스텝 MPC 를 풀어 (delta 명령, SolveInfo) 반환.

        x0: (4,) 현재 상태. preview: 길이 N. u_prev: 직전 delta [rad].
        수렴 실패는 조용히 넘기지 않는다 — converged=False 로 표시하고 이전 해를
        shift 한 fallback 을 쓴다 (mpc-solver.md 수렴 실패 처리).
        """
        if len(preview) != self.N:
            raise ValueError(f"preview 길이 {len(preview)} != N {self.N}")
        x0 = np.asarray(x0, float).reshape(4)

        # 작동점 의존 모델(예: GP 보정) 훅: 현재 상태에서 보정을 갱신한다.
        # GP 는 모델 보정자이므로 현재 작동점에서 mu 를 평가해 지평 상수로 쓴다
        # (상태의존 지평 결합은 mean-only MPC 를 불안정하게 만든다 — Phase 7 진단).
        if hasattr(self.model, "set_operating_point"):
            self.model.set_operating_point(x0, float(u_prev))

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
            Xg = _shift(self._Xprev)
            Ug = _shift_blocked(self._Uprev, self.u_blk, self.n_blk)
        else:
            Xg = np.tile(x0.reshape(4, 1), (1, self.N + 1))
            Ug = np.zeros((1, self.n_blk))
        self.opti.set_initial(self._X, Xg)
        self.opti.set_initial(self._U, Ug)

        t0 = perf_counter()
        try:
            sol = self.opti.solve()
            Xs = np.array(sol.value(self._X)).reshape(4, self.N + 1)
            Us = np.array(sol.value(self._U)).reshape(1, self.n_blk)
            cost = float(sol.value(self._J))
            converged, fallback = True, False
        except RuntimeError:
            # IPOPT 실패 (예: max_iter 초과). fallback = 이전 해 shift.
            converged = False
            fallback = True
            if self._Xprev is not None:
                Xs = _shift(self._Xprev)
                Us = _shift_blocked(self._Uprev, self.u_blk, self.n_blk)
            else:
                Xs = np.tile(x0.reshape(4, 1), (1, self.N + 1))
                Us = np.full((1, self.n_blk), float(u_prev))
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
