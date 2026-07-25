"""명목 선형 동역학 자전거 모델 (오차좌표계). numpy / CasADi 이중 경로.

상태  x = [v_y, gamma, e_psi, e_y]^T   (횡속도[m/s], 요레이트[rad/s],
                                        헤딩오차[rad], 횡방향오차[m])
입력  u = delta                        (전륜 조향각[rad])
외생  vx [m/s], kappa [1/m]            (결정변수 아님 — 파라미터로만 진입)

명목 동역학:  xdot = A(vx) x + B(vx) u + E(vx) kappa

행렬 성분·부호 규약은 .claude/rules/vehicle-model.md 가 유일한 기준이다.
numpy 경로(`*_np`)와 CasADi 경로(`*_ca`)는 이름으로 구분한다
(.claude/rules/python-conventions.md "CasADi <-> numpy 분리").
"""
from __future__ import annotations

import math

import casadi as ca
import numpy as np

from src.config import VehicleConfig


# --------------------------------------------------------------------------- #
# numpy 경로                                                                    #
# --------------------------------------------------------------------------- #
def A_matrix(cfg: VehicleConfig, vx: float) -> np.ndarray:
    """명목 시스템행렬 A(vx). shape (4, 4). vx [m/s].

    vehicle-model.md 의 확정 정식화를 그대로 옮긴다. vx->0 에서 발산하므로
    호출 전 vx 유효성을 검사한다 (특이점 방어).
    """
    cfg.check_vx(vx)
    m, Iz, a, b, Cf, Cr = cfg.m, cfg.Iz, cfg.a, cfg.b, cfg.Cf, cfg.Cr
    return np.array(
        [
            [-(Cf + Cr) / (m * vx), (b * Cr - a * Cf) / (m * vx) - vx, 0.0, 0.0],
            [(b * Cr - a * Cf) / (Iz * vx), -(a * a * Cf + b * b * Cr) / (Iz * vx), 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, vx, 0.0],
        ],
        dtype=float,
    )


def B_matrix(cfg: VehicleConfig, vx: float) -> np.ndarray:
    """명목 입력행렬 B(vx). shape (4, 1). vx [m/s] (미사용이나 시그니처 통일)."""
    cfg.check_vx(vx)
    return np.array([[cfg.Cf / cfg.m], [cfg.a * cfg.Cf / cfg.Iz], [0.0], [0.0]], dtype=float)


def E_matrix(cfg: VehicleConfig, vx: float) -> np.ndarray:
    """명목 외생행렬 E(vx) (kappa 계수). shape (4, 1). vx [m/s]."""
    cfg.check_vx(vx)
    return np.array([[0.0], [0.0], [-vx], [0.0]], dtype=float)


def xdot_np(cfg: VehicleConfig, x: np.ndarray, u: float, vx: float, kappa: float) -> np.ndarray:
    """명목 연속시간 우변 xdot. shape (4,).

    x: (4,) [v_y, gamma, e_psi, e_y], u: delta [rad], vx [m/s], kappa [1/m].
    반환 단위: [m/s^2, rad/s^2, rad/s, m/s].
    """
    x = np.asarray(x, dtype=float).reshape(4)
    A = A_matrix(cfg, vx)
    B = B_matrix(cfg, vx)
    E = E_matrix(cfg, vx)
    return (A @ x + B.reshape(4) * float(u) + E.reshape(4) * float(kappa))


# --------------------------------------------------------------------------- #
# CasADi 경로                                                                   #
# --------------------------------------------------------------------------- #
def dynamics_ca(cfg: VehicleConfig) -> ca.Function:
    """명목 연속시간 우변을 계산하는 CasADi Function 을 만든다.

    반환 Function 시그니처: f(x[4], u[1], vx[1], kappa[1]) -> xdot[4].
    numpy 경로와 대수적으로 동일한 식이어야 한다 (게이트 3, 1e-10 이내).
    심볼릭 경로이므로 vx 특이점 런타임 검사는 하지 않는다 (numpy 경로에서만).
    """
    m, Iz, a, b, Cf, Cr = cfg.m, cfg.Iz, cfg.a, cfg.b, cfg.Cf, cfg.Cr

    x = ca.SX.sym("x", 4)      # [v_y, gamma, e_psi, e_y]
    u = ca.SX.sym("u")         # delta
    vx = ca.SX.sym("vx")
    kappa = ca.SX.sym("kappa")

    v_y, gamma, e_psi, _e_y = x[0], x[1], x[2], x[3]

    # 횡동역학 (vehicle-model.md 부호 규약에서 유도된 A/B 행 그대로).
    vy_dot = (-(Cf + Cr) / (m * vx)) * v_y \
        + ((b * Cr - a * Cf) / (m * vx) - vx) * gamma \
        + (Cf / m) * u
    gamma_dot = ((b * Cr - a * Cf) / (Iz * vx)) * v_y \
        + (-(a * a * Cf + b * b * Cr) / (Iz * vx)) * gamma \
        + (a * Cf / Iz) * u

    # 오차 기구학 (절대 수정 금지 — vehicle-model.md).
    e_psi_dot = gamma - vx * kappa
    e_y_dot = v_y + vx * e_psi

    xdot = ca.vertcat(vy_dot, gamma_dot, e_psi_dot, e_y_dot)
    return ca.Function("f_nom", [x, u, vx, kappa], [xdot],
                       ["x", "u", "vx", "kappa"], ["xdot"])


# --------------------------------------------------------------------------- #
# 해석 검증 함수 (vehicle-model.md "해석 검증 함수")                              #
# --------------------------------------------------------------------------- #
def understeer_gradient(cfg: VehicleConfig) -> float:
    """언더스티어 그래디언트 K_us [rad*s^2/m 계열].

    정의: K_us = m*(b*Cr - a*Cf) / (L*Cf*Cr) = (m/L)*(b/Cf - a/Cr).
    중력가속도 g 로 나눈 rad/g 규약과 혼용하지 마라.
    K_us > 0 언더스티어 / == 0 뉴트럴 / < 0 오버스티어.
    """
    L = cfg.L
    return cfg.m * (cfg.b * cfg.Cr - cfg.a * cfg.Cf) / (L * cfg.Cf * cfg.Cr)


def steady_state_yaw_rate(cfg: VehicleConfig, vx: float, delta: float) -> float:
    """정상상태 요레이트 gamma_ss [rad/s].

    gamma_ss = delta*vx / (L + K_us*vx^2).  delta [rad], vx [m/s].
    분모는 det(A_lat) 판별식과 같은 식이다 (vehicle-model.md 안정성 절).
    """
    cfg.check_vx(vx)
    K_us = understeer_gradient(cfg)
    denom = cfg.L + K_us * vx * vx
    if denom == 0.0:
        return math.inf  # 임계속도 정확히 도달 — 해석해 발산
    return delta * vx / denom


def critical_speed(cfg: VehicleConfig, k_us_tol: float = 1e-12) -> float:
    """오버스티어 임계속도 v_crit [m/s].

    K_us >= 0 (언더/뉴트럴)이면 모든 vx 에서 안정 -> inf 반환.
    K_us < 0 (오버스티어)이면 v_crit = sqrt(-L / K_us).
    K_us == 0 은 부동소수점 등호가 아니라 |K_us| < tol 로 판정한다.
    """
    K_us = understeer_gradient(cfg)
    if K_us >= -k_us_tol:      # 언더스티어 또는 (허용오차 내) 뉴트럴
        return math.inf
    return math.sqrt(-cfg.L / K_us)
