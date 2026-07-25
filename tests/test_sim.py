"""Phase 4 검증 — 모델 완전일치 폐루프 잔차 = noise floor 자릿수.

sim-experiment.md 검증 연쇄 2번: 모델 완전일치 케이스의 잔차는 Phase 1 에서
측정한 noise floor 와 같은 자릿수여야 한다. 어긋나면 GP 로 넘어가지 않는다.

무겁다(폐루프 solve 수백 회). testing.md 분류상 heavy gate.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from src.config import load_group
from src.control.mpc_base import build_step_function
from src.control.mpc_nominal import NominalStepModel, make_nominal_mpc
from src.eval.metrics import compute_metrics
from src.models.integrators import make_rhs_np
from src.path.reference import Reference
from src.sim.runner import run_closed_loop

# Phase 1 게이트 5 채널별 noise floor (sedan, vx=15).
NF = {"v_y": 9.57e-9, "gamma": 9.09e-10, "e_psi": 1.84e-10, "e_y": 5.67e-10}


@pytest.fixture(scope="module")
def matched_log():
    """sedan + single_curve, 플랜트=선형(완전일치) 폐루프 로그."""
    vehicle = load_group("vehicle", "sedan")
    mpc_cfg = load_group("mpc", "default")
    sim = replace(load_group("sim", "default"), duration=8.0)  # 경로 전체 커버, 짧게
    path = load_group("path", "single_curve")

    reference = Reference(path, vehicle.vx_range)
    controller = make_nominal_mpc(vehicle, mpc_cfg, sim.dt_ctrl)
    nominal_step = build_step_function(NominalStepModel(vehicle, sim.dt_ctrl))
    plant_rhs = make_rhs_np(vehicle)   # 완전일치: 플랜트도 선형 명목

    log = run_closed_loop(controller, reference, plant_rhs, nominal_step, sim)
    return log, vehicle, sim


def test_matched_residual_is_noise_floor_order(matched_log, capsys) -> None:
    """완전일치 잔차가 4채널 모두 noise floor 자릿수(< 1e-6)인지."""
    log, vehicle, sim = matched_log
    m = compute_metrics(log, vehicle, sim.dt_ctrl, NF)

    report = ["", "=== Phase 4: 완전일치 잔차 vs noise floor ==="]
    for ch, e in m["residual"].items():
        report.append(f"  {ch:6s} RMS={e['rms']:.3e}  x{e.get('ratio_to_floor', float('nan')):.1f} floor")
    with capsys.disabled():
        print("\n".join(report))

    # 핵심: 완전일치이므로 잔차는 이산화 오차(noise floor) 수준이어야 한다.
    # 실제 버그나 비선형 플랜트는 1e-4 이상을 낸다 -> 1e-6 문턱이 결정적이다.
    for ch, e in m["residual"].items():
        assert e["rms"] < 1e-6, f"{ch} 잔차 RMS={e['rms']:.2e} 가 noise floor 자릿수를 벗어났다"


def test_matched_kinematic_channels_smallest(matched_log) -> None:
    """e_psi, e_y 진단 채널은 이산화 오차 수준(< 5e-8)이어야 한다.

    이 두 식은 vx, kappa 가 정확히 주어지면 오차 없는 기구학이다. 값이 커지면
    기구학 구현 버그다 (gp-residual.md 진단).
    """
    log, vehicle, sim = matched_log
    m = compute_metrics(log, vehicle, sim.dt_ctrl, NF)
    assert m["residual"]["e_psi"]["rms"] < 5e-8
    assert m["residual"]["e_y"]["rms"] < 5e-8


def test_matched_no_convergence_failures(matched_log) -> None:
    """완전일치 폐루프에서 수렴 실패가 없어야 한다."""
    log, vehicle, sim = matched_log
    assert np.all(log["converged"].astype(bool)), "수렴 실패 스텝이 있다"
    assert not np.any(log["fallback"].astype(bool))


def test_matched_tracking_reasonable(matched_log) -> None:
    """완전일치 선형 케이스는 추종오차가 작아야 한다 (RMS e_y < 5 cm)."""
    log, vehicle, sim = matched_log
    m = compute_metrics(log, vehicle, sim.dt_ctrl, NF)
    assert m["tracking"]["rms_e_y"] < 0.05, f"RMS e_y={m['tracking']['rms_e_y']:.3e} m 가 크다"
