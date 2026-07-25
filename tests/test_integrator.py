"""Phase 1 검증 — testing.md 게이트 1, 2, 5.

게이트 1(정상상태 요레이트 1% 이내)이 실패하면 A/B 부호나 Cf/Cr 정의가 틀린 것이다.
여기서 멈추고 보고한다. MPC 로 넘어가지 않는다.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from src.config import VehicleConfig, load_group
from src.models import linear_bicycle as lb
from src.models.integrators import make_rhs_np, plant_step, rk4_step


# --------------------------------------------------------------------------- #
# helpers                                                                      #
# --------------------------------------------------------------------------- #
def _integrate_to_steady(cfg: VehicleConfig, vx: float, delta: float,
                         T: float = 30.0, dt: float = 0.01) -> np.ndarray:
    """고정밀 경로로 스텝 조향 응답을 T초 적분해 최종 상태(4,)를 반환한다.

    x0 = 0, kappa = 0, delta 스텝. ZOH 로 각 스텝 delta 상수.
    """
    f = make_rhs_np(cfg)
    x = np.zeros(4)
    n = int(round(T / dt))
    for _ in range(n):
        x = plant_step(f, x, delta, (vx, 0.0), dt)
    return x


def _neutral_cfg() -> VehicleConfig:
    """뉴트럴 스티어 (a==b, Cf==Cr -> K_us == 0), vx=15 사용 가능."""
    return VehicleConfig(
        m=1500.0, Iz=2250.0, a=1.35, b=1.35, Cf=80000.0, Cr=80000.0,
        delta_max=0.5, delta_rate_max=1.0, vx_range=(5.0, 25.0),
    )


def _oversteer_cfg() -> VehicleConfig:
    """오버스티어 (K_us < 0), v_crit ~= 23.6 m/s. vx=15 는 안정 영역."""
    return VehicleConfig(
        m=1500.0, Iz=2250.0, a=1.5, b=1.2, Cf=120000.0, Cr=80000.0,
        delta_max=0.5, delta_rate_max=1.0, vx_range=(5.0, 40.0),
    )


# --------------------------------------------------------------------------- #
# 게이트 1 — 정상상태 요레이트 (최우선)                                          #
# --------------------------------------------------------------------------- #
def test_gate1_steady_state_yaw_rate() -> None:
    """sedan, vx=15, kappa=0, x0=0, delta=0.02 스텝 -> gamma_ss 해석해와 1% 이내."""
    cfg = load_group("vehicle", "sedan")
    vx, delta = 15.0, 0.02
    x_end = _integrate_to_steady(cfg, vx, delta)
    gamma_num = x_end[1]
    gamma_ana = lb.steady_state_yaw_rate(cfg, vx, delta)

    rel_err = abs(gamma_num - gamma_ana) / abs(gamma_ana)
    assert rel_err < 0.01, (
        f"정상상태 요레이트 불일치 {rel_err:.4%}: "
        f"적분={gamma_num:.6f}, 해석={gamma_ana:.6f} rad/s. "
        f"A/B 부호 또는 Cf/Cr 정의를 의심하라.")


# --------------------------------------------------------------------------- #
# 게이트 2 — 언더/오버/뉴트럴 스티어                                             #
# --------------------------------------------------------------------------- #
def test_gate2_understeer_sign_and_ordering() -> None:
    """K_us 부호와 정상상태 요레이트 대소관계가 이론과 일치.

    같은 delta, vx 에서 gamma_ss = delta*vx/(L + K_us*vx^2) 이므로
    오버스티어(K_us<0) > 뉴트럴(K_us=0) > 언더스티어(K_us>0).
    각 케이스에서 적분 정상상태가 해석해와 1% 이내인지도 확인한다.
    """
    under = load_group("vehicle", "sedan")
    neutral = _neutral_cfg()
    over = _oversteer_cfg()

    k_under = lb.understeer_gradient(under)
    k_neutral = lb.understeer_gradient(neutral)
    k_over = lb.understeer_gradient(over)

    assert k_under > 0.0
    assert abs(k_neutral) < 1e-12
    assert k_over < 0.0

    vx, delta = 15.0, 0.02
    g_under = lb.steady_state_yaw_rate(under, vx, delta)
    g_neutral = lb.steady_state_yaw_rate(neutral, vx, delta)
    g_over = lb.steady_state_yaw_rate(over, vx, delta)

    # 대소관계: 오버 > 뉴트럴 > 언더
    assert g_over > g_neutral > g_under

    # 각 케이스 적분 정상상태가 해석해와 1% 이내
    for cfg, g_ana in ((under, g_under), (neutral, g_neutral), (over, g_over)):
        x_end = _integrate_to_steady(cfg, vx, delta)
        rel_err = abs(x_end[1] - g_ana) / abs(g_ana)
        assert rel_err < 0.01, f"K_us={lb.understeer_gradient(cfg):+.2e} 케이스 불일치 {rel_err:.4%}"


# --------------------------------------------------------------------------- #
# 게이트 5 — noise floor 정량화 (값을 삼키지 말고 출력)                          #
# --------------------------------------------------------------------------- #
def test_gate5_noise_floor_quantified(capsys) -> None:
    """RK4(dt_ctrl) 와 고정밀 적분기의 1스텝 차이를 채널별로 출력한다.

    모델 완전일치 케이스에서 이 값이 잔차의 noise floor 다. 이후 잔차 판정은
    채널별 noise floor 대비 배율로 이루어지므로 채널별로 따로 낸다
    (.claude/rules/gp-residual.md "진단 임계값").
    """
    cfg = load_group("vehicle", "sedan")
    f = make_rhs_np(cfg)
    dt_ctrl = 0.02
    vx = 15.0

    # 대표 궤적을 따라가며 각 제어스텝에서 동일 상태로부터 두 적분기의 1스텝 차이.
    rng = np.random.default_rng(0)
    x = np.zeros(4)
    diffs = []
    for k in range(1000):
        delta = 0.02 * math.sin(0.2 * k * dt_ctrl)   # 완만한 조향 여기(勵起)
        kappa = 0.0
        x_ref = plant_step(f, x, delta, (vx, kappa), dt_ctrl)      # 고정밀
        x_rk4 = np.asarray(rk4_step(f, x, delta, (vx, kappa), dt_ctrl))  # RK4
        diffs.append(np.abs(x_ref - x_rk4))
        x = x_ref  # 플랜트 궤적을 따라 진행

    diffs = np.array(diffs)
    rms = np.sqrt(np.mean(diffs ** 2, axis=0))
    mx = np.max(diffs, axis=0)

    labels = ["v_y [m/s]", "gamma [rad/s]", "e_psi [rad]", "e_y [m]"]
    lines = ["", "=== 게이트 5: noise floor (RK4(dt_ctrl) vs DOP853, 1-step) ==="]
    for i, lab in enumerate(labels):
        lines.append(f"  {lab:16s} RMS={rms[i]:.3e}  max={mx[i]:.3e}")
    report = "\n".join(lines)
    # 값을 삼키지 않도록 stdout 으로 강제 출력 (pytest -s 로 확인).
    with capsys.disabled():
        print(report)

    # noise floor 는 RK4 이산화 오차 수준(작아야)이지만 정확히 0이면 안 된다
    # (두 적분기가 다르다는 사실 자체를 확인).
    assert np.all(rms > 0.0), "두 적분기 차이가 0이다 — 경로가 실제로 다른지 확인하라"
    assert np.all(rms[:2] < 1e-3), f"동적 상태 noise floor 가 예상보다 크다: {rms}"
