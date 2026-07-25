"""Phase 5 검증 — 비선형 플랜트 잔차 특성.

- 동적 채널(v_y, gamma) 잔차가 매칭 대비 급증 (GP 가 학습할 신호).
- 기구학 채널 잔차는 동적 잔차의 0.5*dt 이미지 (독립 정보 아님) — 이 관계가
  성립해야 GP 가 v_y, gamma 만 학습하면 된다는 설계가 검증된다.
- e_psi, e_y 절대 크기로 버그 판정하지 마라 (gp-residual.md 불일치 케이스 진단).

무겁다(폐루프 solve 수백 회).
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from src.config import load_group
from src.control.mpc_base import build_step_function
from src.control.mpc_nominal import NominalStepModel, make_nominal_mpc
from src.models.integrators import make_rhs_np
from src.models.nonlinear_bicycle import make_nonlinear_rhs_np
from src.path.reference import Reference
from src.sim.runner import run_closed_loop

NF = {"v_y": 9.57e-9, "gamma": 9.09e-10}


def _rms(a):
    return float(np.sqrt(np.mean(a ** 2)))


@pytest.fixture(scope="module")
def mismatch_log():
    vehicle = load_group("vehicle", "sedan")
    mpc_cfg = load_group("mpc", "default")
    sim = replace(load_group("sim", "default"), duration=7.0)
    path = load_group("path", "single_curve")
    reference = Reference(path, vehicle.vx_range)
    controller = make_nominal_mpc(vehicle, mpc_cfg, sim.dt_ctrl)
    nominal_step = build_step_function(NominalStepModel(vehicle, sim.dt_ctrl))
    plant_nl = make_nonlinear_rhs_np(vehicle)   # 비선형 Fiala 플랜트
    log = run_closed_loop(controller, reference, plant_nl, nominal_step, sim)
    return log, sim.dt_ctrl


def test_dynamic_channels_have_strong_signal(mismatch_log) -> None:
    """v_y, gamma 잔차가 noise floor 대비 크게 증폭 (학습할 신호)."""
    log, _ = mismatch_log
    r = log["residual"]
    snr_vy = _rms(r[:, 0]) / NF["v_y"]
    snr_gamma = _rms(r[:, 1]) / NF["gamma"]
    assert snr_vy > 1e3, f"v_y SNR={snr_vy:.0f} 이 작다 (타이어 비선형이 안 실렸나)"
    assert snr_gamma > 1e3, f"gamma SNR={snr_gamma:.0f} 이 작다"


def test_kinematic_channels_are_dt_image_of_dynamic(mismatch_log) -> None:
    """RMS(r_epsi) ≈ 0.5*dt*RMS(r_gamma), RMS(r_ey) ≈ 0.5*dt*RMS(r_vy).

    기구학 채널이 동적 채널의 0.5*dt 이미지임을 확인 (독립 정보 아님).
    이탈하면 기구학 구현 버그.
    """
    log, dt = mismatch_log
    r = log["residual"]
    expected = 0.5 * dt
    ratio_epsi = _rms(r[:, 2]) / _rms(r[:, 1])
    ratio_ey = _rms(r[:, 3]) / _rms(r[:, 0])
    assert abs(ratio_epsi - expected) / expected < 0.3, (
        f"e_psi/gamma={ratio_epsi:.4f} != 0.5*dt={expected:.4f}")
    assert abs(ratio_ey - expected) / expected < 0.3, (
        f"e_y/v_y={ratio_ey:.4f} != 0.5*dt={expected:.4f}")


def test_mismatch_far_exceeds_matched(mismatch_log) -> None:
    """동일 경로에서 비선형 플랜트 잔차 >> 선형(매칭) 잔차."""
    log, dt = mismatch_log
    r_nl = _rms(log["residual"][:, 0])   # v_y, 비선형

    # 같은 설정으로 매칭(선형 플랜트) 잔차.
    vehicle = load_group("vehicle", "sedan")
    mpc_cfg = load_group("mpc", "default")
    sim = replace(load_group("sim", "default"), duration=7.0)
    path = load_group("path", "single_curve")
    ref = Reference(path, vehicle.vx_range)
    ctl = make_nominal_mpc(vehicle, mpc_cfg, sim.dt_ctrl)
    nom = build_step_function(NominalStepModel(vehicle, sim.dt_ctrl))
    log_lin = run_closed_loop(ctl, ref, make_rhs_np(vehicle), nom, sim)
    r_lin = _rms(log_lin["residual"][:, 0])

    assert r_nl > 1e3 * r_lin, f"비선형 잔차 {r_nl:.2e} 가 선형 {r_lin:.2e} 대비 안 크다"
