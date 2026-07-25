"""Phase 0 검증 — testing.md 게이트 2b, 3, 4.

게이트 2b는 4x4 전체가 아니라 부분행렬에 대한 검사다. A 는 블록 하삼각이고
오차 기구학에서 원점 고유값이 2개 나오므로, 4x4 전체에 Hurwitz 검사를 걸면
반드시 실패한다 (그것이 정상이다).
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from src.config import VehicleConfig, load_group
from src.models import linear_bicycle as lb


# --------------------------------------------------------------------------- #
# fixtures                                                                     #
# --------------------------------------------------------------------------- #
@pytest.fixture
def sedan() -> VehicleConfig:
    """언더스티어 기준 차량 (configs/vehicle/sedan.yaml)."""
    return load_group("vehicle", "sedan")


def _oversteer_cfg() -> VehicleConfig:
    """오버스티어 config (K_us < 0). a*Cf > b*Cr 이 되도록 전축을 강하게.

    b*Cr = 1.2*80000 = 96000 < a*Cf = 1.5*120000 = 180000  -> 오버스티어.
    v_crit = sqrt(-L/K_us) ~= 23.6 m/s (테스트 가능한 범위).
    """
    return VehicleConfig(
        m=1500.0, Iz=2250.0, a=1.5, b=1.2, Cf=120000.0, Cr=80000.0,
        delta_max=0.5, delta_rate_max=1.0, vx_range=(5.0, 40.0),
    )


# --------------------------------------------------------------------------- #
# 게이트 2b — 고유값 · 임계속도                                                  #
# --------------------------------------------------------------------------- #
def test_gate2b_Alat_hurwitz_sedan(sedan: VehicleConfig) -> None:
    """A_lat = A[0:2,0:2] 의 고유값 2개 모두 실수부 음수 (sedan, vx=15)."""
    A = lb.A_matrix(sedan, 15.0)
    A_lat = A[0:2, 0:2]
    eig = np.linalg.eigvals(A_lat)
    assert np.all(eig.real < 0.0), f"A_lat 고유값 실수부가 음수가 아니다: {eig}"


def test_gate2b_error_kinematics_block_zero(sedan: VehicleConfig) -> None:
    """오차 기구학 블록 A[2:4,2:4] 의 고유값 2개가 모두 0 (구조 검증)."""
    A = lb.A_matrix(sedan, 15.0)
    block = A[2:4, 2:4]
    eig = np.linalg.eigvals(block)
    assert np.allclose(eig, 0.0, atol=1e-12), f"오차 기구학 블록 고유값이 0이 아니다: {eig}"


def test_gate2b_block_triangular_spectrum(sedan: VehicleConfig) -> None:
    """블록 하삼각: eig(A) 전체 = eig(A_lat) ∪ eig(A[2:4,2:4])."""
    A = lb.A_matrix(sedan, 15.0)
    full = np.sort_complex(np.linalg.eigvals(A))
    parts = np.sort_complex(np.concatenate([
        np.linalg.eigvals(A[0:2, 0:2]),
        np.linalg.eigvals(A[2:4, 2:4]),
    ]))
    assert np.allclose(full, parts, atol=1e-10), f"전체 {full} != 블록합 {parts}"


def test_gate2b_det_Alat_matches_discriminant(sedan: VehicleConfig) -> None:
    """det(A_lat) 와 CfCr*L*(L+K_us*vx^2)/(m*Iz*vx^2) 가 수치적으로 일치.

    이 판별식은 steady_state_yaw_rate 분모(L + K_us*vx^2)와 부호를 공유한다.
    """
    K_us = lb.understeer_gradient(sedan)
    for vx in (5.0, 10.0, 15.0, 25.0):
        A_lat = lb.A_matrix(sedan, vx)[0:2, 0:2]
        det_num = np.linalg.det(A_lat)
        L = sedan.L
        det_ana = (sedan.Cf * sedan.Cr * L * (L + K_us * vx * vx)
                   / (sedan.m * sedan.Iz * vx * vx))
        assert math.isclose(det_num, det_ana, rel_tol=1e-9), (
            f"vx={vx}: det_num={det_num} != det_ana={det_ana}")
        # 언더스티어이므로 det > 0, 분모도 > 0
        assert det_num > 0.0 and (L + K_us * vx * vx) > 0.0


def test_gate2b_oversteer_critical_speed_transition() -> None:
    """오버스티어 config 에서 v_crit 근처 안정 -> 불안정 전환."""
    cfg = _oversteer_cfg()
    K_us = lb.understeer_gradient(cfg)
    assert K_us < 0.0, "테스트 config 가 오버스티어가 아니다"

    v_crit = lb.critical_speed(cfg)
    assert math.isfinite(v_crit) and v_crit > 0.0

    # v_crit 아래: A_lat 안정 (실수부 모두 음수)
    A_below = lb.A_matrix(cfg, v_crit * 0.9)[0:2, 0:2]
    assert np.all(np.linalg.eigvals(A_below).real < 0.0)

    # v_crit 위: A_lat 불안정 (실수부 하나 이상 양수)
    A_above = lb.A_matrix(cfg, v_crit * 1.1)[0:2, 0:2]
    assert np.any(np.linalg.eigvals(A_above).real > 0.0)


def test_gate2b_understeer_critical_speed_inf(sedan: VehicleConfig) -> None:
    """언더스티어(K_us > 0)의 임계속도는 inf."""
    assert lb.understeer_gradient(sedan) > 0.0
    assert lb.critical_speed(sedan) == math.inf


# --------------------------------------------------------------------------- #
# 게이트 3 — numpy <-> CasADi 일치 (1e-10 이내)                                 #
# --------------------------------------------------------------------------- #
def test_gate3_np_ca_match(sedan: VehicleConfig) -> None:
    """같은 입력에 대해 numpy xdot 와 CasADi xdot 차이가 1e-10 이내."""
    f = lb.dynamics_ca(sedan)
    rng = np.random.default_rng(0)
    for _ in range(20):
        x = rng.normal(size=4)
        u = float(rng.normal())
        vx = float(rng.uniform(*sedan.vx_range))
        kappa = float(rng.normal(scale=0.05))

        xd_np = lb.xdot_np(sedan, x, u, vx, kappa)
        xd_ca = np.array(f(x, u, vx, kappa)).reshape(4)
        assert np.max(np.abs(xd_np - xd_ca)) < 1e-10, (
            f"불일치: np={xd_np}, ca={xd_ca}")


# --------------------------------------------------------------------------- #
# 게이트 4 — 특이점 방어                                                         #
# --------------------------------------------------------------------------- #
def test_gate4_vx_range_lower_nonpositive_rejected() -> None:
    """vx_range 하한 <= 0 이면 config 유효성 검사가 예외를 던진다."""
    with pytest.raises(ValueError):
        VehicleConfig(m=1500.0, Iz=2250.0, a=1.2, b=1.5, Cf=80000.0, Cr=80000.0,
                      delta_max=0.5, delta_rate_max=1.0, vx_range=(0.0, 25.0))
    with pytest.raises(ValueError):
        VehicleConfig(m=1500.0, Iz=2250.0, a=1.2, b=1.5, Cf=80000.0, Cr=80000.0,
                      delta_max=0.5, delta_rate_max=1.0, vx_range=(-1.0, 25.0))


def test_gate4_runtime_vx_below_lower_raises(sedan: VehicleConfig) -> None:
    """런타임 vx 가 하한 미만이면 예외를 던진다."""
    lo = sedan.vx_range[0]
    with pytest.raises(ValueError):
        lb.A_matrix(sedan, lo - 0.1)
    with pytest.raises(ValueError):
        sedan.check_vx(lo * 0.5)
    # 하한 이상은 통과해야 한다
    lb.A_matrix(sedan, lo)


def test_gate4_negative_physical_param_rejected() -> None:
    """물리 파라미터가 음수/0 이면 예외."""
    with pytest.raises(ValueError):
        VehicleConfig(m=-1.0, Iz=2250.0, a=1.2, b=1.5, Cf=80000.0, Cr=80000.0,
                      delta_max=0.5, delta_rate_max=1.0, vx_range=(5.0, 25.0))
