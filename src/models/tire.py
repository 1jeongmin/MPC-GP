"""타이어 모델 — Fiala 브러시 모델 (단일 마찰계수).

출처: 표준 Fiala 브러시 모델. 큐빅 포화 형태는 여러 권위 있는 문헌에서 동일하게
확인된다 — ScienceDirect "Fiala Model overview", MathWorks "Fiala Wheel 2DOF",
Stanford(Gerdes 그룹) 자율주행 문헌(arXiv:2407.12989). 원 출처 Fiala(1954).
**특정 논문 DOI 는 확인 안 됨.** 다만 아래 성질은 대수적으로 자체 검증했다:
  - alpha->0 에서 dFy/dalpha = C_alpha (선형 극한)
  - |alpha| = alpha_sl 에서 Fy = mu*Fz 이고 도함수 0 (C1 연속 포화)
  - 원점 대칭 (홀함수)

부호 규약: 문헌은 대개 Fy = -C_alpha*tan(alpha)+... (양의 슬립 -> 음의 힘).
우리 규약은 **Fy = +C_alpha*alpha** (vehicle-model.md: F_yf = Cf*alpha_f, Cf>0).
따라서 모든 항의 부호를 뒤집었다. 그대로 옮기면 잔차 부호가 뒤집힌다.

구현 형태 (numpy/CasADi 동일):
  Fymax = mu*Fz,  alpha_sl = atan(3*Fymax/C_alpha)
  z = tan(clip(alpha, -alpha_sl, alpha_sl))
  Fy = C_alpha*z - (C_alpha^2/(3*Fymax))*z*|z| + (C_alpha^3/(27*Fymax^2))*z^3
clip 덕분에 |alpha|>=alpha_sl 에서 큐빅이 정확히 ±Fymax 로 포화하므로 분기가 필요 없다.
"""
from __future__ import annotations

import casadi as ca
import numpy as np


def fiala_fy_np(alpha, C_alpha: float, mu: float, Fz: float):
    """Fiala 횡력 [N] (numpy). alpha [rad] 스칼라/배열. 우리 부호 규약.

    C_alpha [N/rad] 코너링 강성(양수), mu [-] 마찰계수, Fz [N] 수직하중.
    """
    alpha = np.asarray(alpha, dtype=float)
    Fymax = mu * Fz
    alpha_sl = np.arctan(3.0 * Fymax / C_alpha)
    z = np.tan(np.clip(alpha, -alpha_sl, alpha_sl))
    return (C_alpha * z
            - (C_alpha**2 / (3.0 * Fymax)) * z * np.abs(z)
            + (C_alpha**3 / (27.0 * Fymax**2)) * z**3)


def fiala_fy_ca(alpha, C_alpha: float, mu: float, Fz: float):
    """Fiala 횡력 (CasADi). alpha 는 SX/MX 심볼. 파라미터는 수치.

    numpy 경로와 1e-10 이내 동일해야 한다 (게이트 대조).
    """
    Fymax = mu * Fz
    alpha_sl = float(np.arctan(3.0 * Fymax / C_alpha))
    z = ca.tan(ca.fmax(-alpha_sl, ca.fmin(alpha_sl, alpha)))
    return (C_alpha * z
            - (C_alpha**2 / (3.0 * Fymax)) * z * ca.fabs(z)
            + (C_alpha**3 / (27.0 * Fymax**2)) * z**3)


def sliding_slip_angle(C_alpha: float, mu: float, Fz: float) -> float:
    """포화 슬립각 alpha_sl = atan(3*mu*Fz/C_alpha) [rad]."""
    return float(np.arctan(3.0 * mu * Fz / C_alpha))
