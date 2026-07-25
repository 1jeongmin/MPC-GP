"""Phase 2 검증 — testing.md 게이트 8.

- 일정곡률 구간에서 복원된 (x,y)가 반지름 1/kappa 인 원호와 일치.
- 프리뷰 배열 길이가 항상 N, 경로 끝에서 마지막 값이 유지.
- kappa=0 구간에서 vx_max 가 유한 (0-division 없음).
"""
from __future__ import annotations

import numpy as np
import pytest

from src.config import load_group
from src.path.reference import Reference


VX_RANGE = (5.0, 25.0)  # sedan


@pytest.fixture
def ref() -> Reference:
    return Reference(load_group("path", "single_curve"), VX_RANGE)


def _circumradius(p1, p2, p3) -> float:
    """세 점의 외접원 반지름. 세 점이 원호 위에 있으면 그 원의 반지름."""
    a = np.linalg.norm(p2 - p3)
    b = np.linalg.norm(p1 - p3)
    c = np.linalg.norm(p1 - p2)
    area = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))
    return a * b * c / (4.0 * area)


# --------------------------------------------------------------------------- #
# 게이트 8 — 원호 복원                                                          #
# --------------------------------------------------------------------------- #
def test_gate8_constant_curvature_is_arc(ref: Reference) -> None:
    """일정곡률 구간의 복원 점들이 반지름 1/kappa_max 원호 위에 있다.

    single_curve: straight20 / clothoid30 / constant40 / clothoid30 / straight20.
    일정곡률 구간은 s in [50, 90], kappa = 0.02 -> R = 50 m.
    """
    ds = 0.02
    x, y, _ = ref.reconstruct_xy(ds=ds)
    s = np.linspace(0.0, ref.total_length, len(x))

    # 일정곡률 구간 내부(경계 클로소이드 영향 배제 위해 여유).
    mask = (s >= 52.0) & (s <= 88.0)
    xs, ys = x[mask], y[mask]
    assert len(xs) > 10

    R_expected = 1.0 / 0.02  # 50 m
    # 구간 내 여러 삼점 조합의 외접원 반지름이 R_expected 와 일치.
    idx = np.linspace(0, len(xs) - 1, 5).astype(int)
    pts = np.column_stack([xs, ys])
    radii = []
    for i in range(len(idx) - 2):
        r = _circumradius(pts[idx[i]], pts[idx[i + 1]], pts[idx[i + 2]])
        radii.append(r)
    radii = np.array(radii)
    assert np.all(np.abs(radii - R_expected) / R_expected < 1e-3), (
        f"복원 반지름 {radii} != {R_expected}")


def test_gate8_straight_segments_are_lines(ref: Reference) -> None:
    """직선 구간에서 곡률 0, 복원 경로 곡률 없음."""
    # 첫 직선 구간 s in [0,20] 에서 kappa == 0.
    s = np.linspace(0.0, 19.0, 20)
    assert np.allclose(ref.kappa_of_s(s), 0.0, atol=1e-12)


# --------------------------------------------------------------------------- #
# 게이트 8 — 프리뷰 길이 · 끝값 유지                                             #
# --------------------------------------------------------------------------- #
def test_gate8_preview_length_always_N(ref: Reference) -> None:
    """프리뷰 배열 길이가 항상 N (경로 중간·끝 어디서든)."""
    N, dt = 20, 0.02
    for s0 in (0.0, 60.0, ref.total_length - 1.0, ref.total_length, ref.total_length + 50.0):
        kappa, vx = ref.get_preview(s0, N, dt)
        assert kappa.shape == (N,) and vx.shape == (N,)


def test_gate8_preview_holds_last_value_at_end(ref: Reference) -> None:
    """경로 끝을 넘어선 프리뷰는 마지막 값(직선 kappa=0, vx=vx_hi)을 유지한다."""
    N, dt = 30, 0.02
    kappa, vx = ref.get_preview(ref.total_length + 10.0, N, dt)
    assert np.allclose(kappa, 0.0, atol=1e-12)
    assert np.allclose(vx, VX_RANGE[1])
    # 끝 근처에서 시작해도 후반부는 상수로 고정.
    kappa2, vx2 = ref.get_preview(ref.total_length - 0.5, N, dt)
    assert np.allclose(kappa2[-5:], 0.0, atol=1e-12)
    assert np.allclose(vx2[-5:], VX_RANGE[1])


# --------------------------------------------------------------------------- #
# 게이트 8 — kappa=0 에서 vx 유한 (0-division 방어)                             #
# --------------------------------------------------------------------------- #
def test_gate8_straight_vx_finite(ref: Reference) -> None:
    """직선(kappa=0)에서 vx_max 가 유한하고 vx_hi 로 clip 된다."""
    s = np.linspace(0.0, 20.0, 50)  # 첫 직선 구간
    vx = ref.vx_max_of_s(s)
    assert np.all(np.isfinite(vx))
    assert np.allclose(vx, VX_RANGE[1])


def test_gate8_curve_vx_matches_formula(ref: Reference) -> None:
    """일정곡률 구간 vx = sqrt(a_y_max/kappa_max), vx_range clip."""
    vx = float(ref.vx_max_of_s(70.0))  # 일정곡률 구간 한가운데
    expected = np.clip(np.sqrt(4.0 / 0.02), *VX_RANGE)
    assert abs(vx - expected) < 1e-9


def test_gate8_total_length(ref: Reference) -> None:
    """총 호길이 = 20+30+40+30+20 = 140 m."""
    assert abs(ref.total_length - 140.0) < 1e-9


def test_gate8_dlc_loads_and_reconstructs() -> None:
    """dlc 경로도 로드·복원되는지 (정의만, 검증은 single_curve)."""
    ref = Reference(load_group("path", "dlc"), VX_RANGE)
    x, y, theta = ref.reconstruct_xy(ds=0.05)
    assert np.all(np.isfinite(x)) and np.all(np.isfinite(y))
    # 시작·끝 헤딩이 0 근처로 복귀 (차선변경 후 원래 방향).
    assert abs(theta[-1]) < 1e-6
