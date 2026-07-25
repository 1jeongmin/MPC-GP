"""Phase 5 검증 — Fiala 타이어 (최우선 게이트) + 비선형 플랜트 대조.

alpha->0 에서 dFy/dalpha = C_alpha 가 통과하지 않으면, 잔차에 "타이어 비선형성"이
아니라 "코너링 강성 불일치"라는 다른 원인이 섞인다. 실험 설계가 오염된다.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.config import load_group
from src.models.nonlinear_bicycle import dynamics_nl_ca, xdot_nl_np
from src.models.tire import fiala_fy_ca, fiala_fy_np, sliding_slip_angle

import casadi as ca


@pytest.fixture
def sedan():
    return load_group("vehicle", "sedan")


def _params(sedan):
    """전축 타이어 파라미터 (C_alpha, mu, Fz)."""
    return sedan.Cf, sedan.mu, sedan.Fzf


# --------------------------------------------------------------------------- #
# 최우선 게이트 — alpha->0 에서 dFy/dalpha = C_alpha                            #
# --------------------------------------------------------------------------- #
def test_linear_limit_slope(sedan) -> None:
    """중앙차분으로 alpha=0 근방 dFy/dalpha 가 C_alpha 로 수렴."""
    Ca, mu, Fz = _params(sedan)
    h = 1e-6
    dFy = (fiala_fy_np(h, Ca, mu, Fz) - fiala_fy_np(-h, Ca, mu, Fz)) / (2 * h)
    assert abs(dFy - Ca) / Ca < 1e-4, f"dFy/dalpha={dFy:.1f} != C_alpha={Ca}"


def test_linear_limit_small_angle(sedan) -> None:
    """0 근방에서 Fiala ~= C_alpha*alpha (선형 타이어 극한).

    선형 극한은 alpha->0 에서만 성립한다. Fiala 는 큐빅 항 때문에 이미 alpha~0.005
    에서 선형 대비 약 1.8% 낮다 (타이어 비선형성 onset — 잔차의 물리적 출처).
    """
    Ca, mu, Fz = _params(sedan)
    for a in (1e-5, 1e-4, 1e-3):
        fy = float(fiala_fy_np(a, Ca, mu, Fz))
        assert abs(fy - Ca * a) / (Ca * a) < 1e-2, f"alpha={a}: {fy} vs {Ca*a}"


# --------------------------------------------------------------------------- #
# 게이트 — 포화 (|alpha|->큰 값에서 |Fy|->mu*Fz)                                #
# --------------------------------------------------------------------------- #
def test_saturation(sedan) -> None:
    """큰 슬립각에서 |Fy| -> mu*Fz."""
    Ca, mu, Fz = _params(sedan)
    Fymax = mu * Fz
    for a in (0.3, 0.5, 1.0, 1.5):
        fy = float(fiala_fy_np(a, Ca, mu, Fz))
        assert abs(fy - Fymax) < 1e-6, f"alpha={a}: |Fy|={fy} != mu*Fz={Fymax}"
        assert abs(float(fiala_fy_np(-a, Ca, mu, Fz)) + Fymax) < 1e-6


def test_saturation_boundary_c1(sedan) -> None:
    """alpha_sl 에서 Fy=mu*Fz 이고 도함수 연속(0에 가까움)."""
    Ca, mu, Fz = _params(sedan)
    a_sl = sliding_slip_angle(Ca, mu, Fz)
    Fymax = mu * Fz
    assert abs(float(fiala_fy_np(a_sl, Ca, mu, Fz)) - Fymax) < 1e-6
    # 경계 직전/직후 도함수 연속 (포화측 기울기 ~0).
    h = 1e-7
    d_out = (fiala_fy_np(a_sl + h, Ca, mu, Fz) - fiala_fy_np(a_sl, Ca, mu, Fz)) / h
    assert abs(d_out) < 1e-3


# --------------------------------------------------------------------------- #
# 게이트 — 원점 대칭 (홀함수)                                                    #
# --------------------------------------------------------------------------- #
def test_origin_symmetry(sedan) -> None:
    Ca, mu, Fz = _params(sedan)
    for a in (0.01, 0.1, 0.25, 0.4):
        assert abs(float(fiala_fy_np(a, Ca, mu, Fz)) + float(fiala_fy_np(-a, Ca, mu, Fz))) < 1e-9


# --------------------------------------------------------------------------- #
# 게이트 — numpy <-> CasADi 일치 (1e-10)                                        #
# --------------------------------------------------------------------------- #
def test_tire_np_ca_match(sedan) -> None:
    Ca, mu, Fz = _params(sedan)
    a = ca.SX.sym("a")
    f = ca.Function("fy", [a], [fiala_fy_ca(a, Ca, mu, Fz)])
    for alpha in np.linspace(-0.6, 0.6, 25):
        fy_np = float(fiala_fy_np(alpha, Ca, mu, Fz))
        fy_ca = float(f(alpha))
        assert abs(fy_np - fy_ca) < 1e-10, f"alpha={alpha}: np={fy_np} ca={fy_ca}"


def test_nonlinear_bicycle_np_ca_match(sedan) -> None:
    """비선형 플랜트 우변 numpy <-> CasADi 일치 (1e-10)."""
    f = dynamics_nl_ca(sedan)
    rng = np.random.default_rng(0)
    for _ in range(20):
        x = rng.normal(scale=[1.0, 0.3, 0.1, 0.5], size=4)
        u = float(rng.normal(scale=0.1))
        vx = float(rng.uniform(*sedan.vx_range))
        kappa = float(rng.normal(scale=0.02))
        xd_np = xdot_nl_np(sedan, x, u, vx, kappa)
        xd_ca = np.array(f(x, u, vx, kappa)).reshape(4)
        assert np.max(np.abs(xd_np - xd_ca)) < 1e-10


# --------------------------------------------------------------------------- #
# 게이트 — 비선형 플랜트가 선형 극한에서 선형 명목과 일치                        #
# --------------------------------------------------------------------------- #
def test_nonlinear_reduces_to_linear_small_slip(sedan) -> None:
    """작은 상태(작은 슬립)에서 비선형 우변 ~= 선형 우변.

    이것이 성립해야 잔차가 '타이어 비선형성'만 담는다.
    """
    from src.models.linear_bicycle import xdot_np as xdot_lin
    x = np.array([0.02, 0.005, 0.001, 0.01])   # 매우 작은 상태
    u, vx, kappa = 0.002, 15.0, 0.0
    xd_nl = xdot_nl_np(sedan, x, u, vx, kappa)
    xd_lin = xdot_lin(sedan, x, u, vx, kappa)
    # 동적 채널(v_y,gamma)이 상대적으로 일치. 기구학 채널은 완전히 동일해야 한다.
    assert np.allclose(xd_nl[2:], xd_lin[2:], atol=1e-12), "기구학 채널이 어긋난다"
    assert np.max(np.abs(xd_nl[:2] - xd_lin[:2])) / (np.abs(xd_lin[:2]).max()) < 1e-2
