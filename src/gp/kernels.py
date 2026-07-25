"""ARD RBF 커널 (.claude/rules/gp-residual.md).

k(z, z') = sigma_f^2 * exp(-0.5 * sum_d ((z_d - z'_d)/l_d)^2)

numpy 경로(`*_np`)와 CasADi 경로(`*_ca`)를 모두 제공한다. 같은 입력에 같은 출력을
내야 한다 (대조 테스트). 커널은 표준화된 z 공간에서 계산한다 (lengthscale 도 표준화 단위).
"""
from __future__ import annotations

import casadi as ca
import numpy as np


def ard_rbf_np(A: np.ndarray, B: np.ndarray, lengthscales: np.ndarray,
               sigma_f: float) -> np.ndarray:
    """ARD RBF 그람행렬. A:(n,d), B:(m,d), lengthscales:(d,) -> (n,m)."""
    A = np.atleast_2d(A)
    B = np.atleast_2d(B)
    ls = np.asarray(lengthscales, float).reshape(1, -1)
    As = A / ls           # (n,d)
    Bs = B / ls           # (m,d)
    # 제곱거리 ||As_i - Bs_j||^2 = |As_i|^2 + |Bs_j|^2 - 2 As_i.Bs_j
    a2 = np.sum(As**2, axis=1, keepdims=True)     # (n,1)
    b2 = np.sum(Bs**2, axis=1, keepdims=True).T   # (1,m)
    d2 = a2 + b2 - 2.0 * As @ Bs.T
    d2 = np.maximum(d2, 0.0)                        # 수치 음수 방지
    return sigma_f**2 * np.exp(-0.5 * d2)


def ard_rbf_ca(z, Z, lengthscales, sigma_f):
    """단일 질의 z(d,)에 대한 커널 벡터 k(z, Z) -> (M,1) CasADi.

    z: (d,) SX/MX 심볼. Z: (M,d) 파라미터. lengthscales: (d,) 파라미터. sigma_f: 스칼라.
    numpy 경로와 1e-10 이내 일치해야 한다.
    """
    M = Z.shape[0]
    ls = ca.reshape(lengthscales, 1, -1)           # (1,d)
    zt = ca.reshape(z, 1, -1)                       # (1,d)
    diff = (ca.repmat(zt, M, 1) - Z) / ca.repmat(ls, M, 1)   # (M,d)
    d2 = ca.sum2(diff**2)                           # (M,1)
    return sigma_f**2 * ca.exp(-0.5 * d2)          # (M,1)
