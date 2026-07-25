"""Phase 3 검증 — testing.md 게이트 6 (MPC 정합성).

- 초기오차 0, kappa=0 -> delta ~ 0
- 초기 e_y=0.5 -> 오차를 줄이는 방향의 delta
- 모든 제약이 수치 허용오차 내에서 만족
- IPOPT 수렴 실패 시 처리 경로가 실제로 동작 (실패를 인위적으로 유발)
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from src.config import load_group
from src.control.mpc_base import Preview
from src.control.mpc_nominal import make_nominal_mpc

DT = 0.02


@pytest.fixture
def vehicle():
    return load_group("vehicle", "sedan")


@pytest.fixture
def mpc_cfg():
    return load_group("mpc", "default")


def _straight_preview(N: int, vx: float = 15.0) -> Preview:
    return Preview.from_arrays(np.zeros(N), np.full(N, vx))


def _curve_preview(N: int, kappa: float = 0.02, vx: float = 14.0) -> Preview:
    return Preview.from_arrays(np.full(N, kappa), np.full(N, vx))


# --------------------------------------------------------------------------- #
# 게이트 6 — 무오차/무곡률 -> delta ~ 0                                         #
# --------------------------------------------------------------------------- #
def test_gate6_zero_error_zero_steer(vehicle, mpc_cfg) -> None:
    mpc = make_nominal_mpc(vehicle, mpc_cfg, DT)
    u, info = mpc.solve(np.zeros(4), _straight_preview(mpc_cfg.N), u_prev=0.0)
    assert info.converged
    assert abs(u) < 1e-5, f"무오차·무곡률인데 delta={u}"


# --------------------------------------------------------------------------- #
# 게이트 6 — 초기 e_y=0.5 -> 오차 감소 방향                                      #
# --------------------------------------------------------------------------- #
def test_gate6_lateral_error_corrected(vehicle, mpc_cfg) -> None:
    mpc = make_nominal_mpc(vehicle, mpc_cfg, DT)
    x0 = np.array([0.0, 0.0, 0.0, 0.5])  # e_y = 0.5 m
    u, info = mpc.solve(x0, _straight_preview(mpc_cfg.N), u_prev=0.0)
    assert info.converged
    # 예측 지평에서 횡오차가 줄어든다.
    ey_traj = info.X[3, :]
    assert abs(ey_traj[-1]) < abs(ey_traj[0]), (
        f"e_y 가 줄지 않았다: {ey_traj[0]} -> {ey_traj[-1]}")
    assert abs(u) > 1e-4, "오차가 있는데 조향이 거의 0이다"


# --------------------------------------------------------------------------- #
# 게이트 6 — 제약 만족                                                          #
# --------------------------------------------------------------------------- #
def test_gate6_constraints_satisfied(vehicle, mpc_cfg) -> None:
    mpc = make_nominal_mpc(vehicle, mpc_cfg, DT)
    # 곡선 프리뷰 + 초기 오차로 입력을 크게 유도.
    x0 = np.array([0.0, 0.0, 0.1, 0.3])
    u_prev = 0.0
    u, info = mpc.solve(x0, _curve_preview(mpc_cfg.N), u_prev=u_prev)
    assert info.converged

    tol = 1e-6
    U = info.U[0, :]
    # 조향각 한계.
    assert np.all(np.abs(U) <= vehicle.delta_max + tol), f"|delta| 한계 위반: {U.max()}"
    # 조향각 변화율 한계 (k=0 은 u_prev 기준).
    drate = vehicle.delta_rate_max * DT
    du0 = U[0] - u_prev
    assert abs(du0) <= drate + tol, f"k=0 rate 위반: {du0} > {drate}"
    du = np.diff(U)
    assert np.all(np.abs(du) <= drate + tol), f"rate 위반: {np.abs(du).max()} > {drate}"


# --------------------------------------------------------------------------- #
# 게이트 6 — solve time / warm start                                          #
# --------------------------------------------------------------------------- #
def test_gate6_solve_time_and_warmstart(vehicle, mpc_cfg) -> None:
    mpc = make_nominal_mpc(vehicle, mpc_cfg, DT)
    pv = _curve_preview(mpc_cfg.N)
    u1, info1 = mpc.solve(np.array([0.0, 0.0, 0.05, 0.2]), pv, 0.0)
    u2, info2 = mpc.solve(np.array([0.0, 0.0, 0.04, 0.15]), pv, u1)
    assert info1.solve_time > 0.0 and info2.solve_time > 0.0
    assert info1.iterations is not None
    assert info1.converged and info2.converged


# --------------------------------------------------------------------------- #
# 게이트 6 — 수렴 실패 처리 (인위적 유발)                                        #
# --------------------------------------------------------------------------- #
def test_gate6_convergence_failure_fallback(vehicle, mpc_cfg) -> None:
    """max_iter=1 + tol 극단으로 1스텝 내 수렴 불가 -> fallback 경로 동작 확인."""
    cfg_fail = replace(mpc_cfg, ipopt_max_iter=1, ipopt_tol=1e-14)
    mpc = make_nominal_mpc(vehicle, cfg_fail, DT)
    x0 = np.array([0.0, 0.0, 0.2, 0.5])  # 비자명 오차
    u, info = mpc.solve(x0, _curve_preview(cfg_fail.N), u_prev=0.0)

    assert not info.converged, "인위적 실패인데 converged=True 로 나왔다"
    assert info.fallback_used, "fallback 이 사용되지 않았다"
    assert np.isfinite(u), "fallback 명령이 유한하지 않다"
    # 실패해도 solve_time 은 기록된다.
    assert info.solve_time > 0.0
