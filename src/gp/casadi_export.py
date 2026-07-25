"""학습된 GP -> CasADi 표현 변환 (.claude/rules/gp-residual.md).

mu(z) = k(z_std, Z) @ alpha (표준화 공간) 후 역표준화해 실단위 잔차 반환.
Z/alpha/lengthscale/sigma_f/표준화통계를 CasADi **파라미터**로 주입한다 (온라인
갱신에도 NLP 재빌드 없이 값만 바뀌게 — mpc-solver.md). offline 이므로 값은 고정.

**numpy GP 예측과 CasADi 예측이 일치하는지 대조는 필수** (게이트 7).

파라미터 평탄화 레이아웃 (M = 딕셔너리 크기, 채널 2, 입력차원 d=3):
  Z(M*d) | alpha(M*2) | ls(2*d) | sf(2) | z_mean(d) | z_scale(d) | r_mean(2) | r_scale(2)
행렬은 order='F'(열우선) 로 평탄화 — CasADi ca.reshape(열우선)과 일치시키기 위함.
"""
from __future__ import annotations

import casadi as ca
import numpy as np

from src.gp.kernels import ard_rbf_ca
from src.gp.train_offline import TwoChannelGP

D = 3            # 입력 차원 [v_y, gamma, delta]
N_CH = 2         # 채널 [v_y, gamma]


def gp_extra_dim(M: int) -> int:
    """평탄화 파라미터 벡터 길이."""
    return M * D + M * N_CH + N_CH * D + N_CH + D + D + N_CH + N_CH


def gp_param_vector(gp: TwoChannelGP) -> np.ndarray:
    """학습된 GP 를 평탄 파라미터 벡터로. 두 채널은 같은 딕셔너리 Z 를 공유한다."""
    Z = gp.channels[0].Z                                     # (M,d) 공유
    assert np.allclose(Z, gp.channels[1].Z), "채널 간 딕셔너리 Z 불일치"
    alpha = np.column_stack([ch.alpha for ch in gp.channels])   # (M,2)
    ls = np.vstack([ch.lengthscales for ch in gp.channels])     # (2,d)
    sf = np.array([ch.sigma_f for ch in gp.channels])           # (2,)
    return np.concatenate([
        Z.flatten(order="F"), alpha.flatten(order="F"), ls.flatten(order="F"), sf,
        gp.z_scaler.mean, gp.z_scaler.scale, gp.r_scaler.mean, gp.r_scaler.scale,
    ])


def mu_ca_expr(z, p, M: int):
    """CasADi 심볼릭 GP 평균 mu(z) (실단위, 2채널). z:(d,) 심볼, p:(extra_dim,) 파라미터.

    B_d 주입 전의 잔차 평균 [mu_vy, mu_gamma] 를 반환한다 (2,1).
    """
    o = 0
    Z = ca.reshape(p[o:o + M * D], M, D); o += M * D
    alpha = ca.reshape(p[o:o + M * N_CH], M, N_CH); o += M * N_CH
    ls = ca.reshape(p[o:o + N_CH * D], N_CH, D); o += N_CH * D
    sf = p[o:o + N_CH]; o += N_CH
    z_mean = p[o:o + D]; o += D
    z_scale = p[o:o + D]; o += D
    r_mean = p[o:o + N_CH]; o += N_CH
    r_scale = p[o:o + N_CH]; o += N_CH

    z_std = (ca.reshape(z, D, 1) - z_mean) / z_scale        # (d,1)
    mus = []
    for j in range(N_CH):
        k = ard_rbf_ca(z_std, Z, ls[j, :], sf[j])           # (M,1)
        mus.append(ca.mtimes(k.T, alpha[:, j]))             # (1,1) 표준화 평균
    mu_std = ca.vertcat(*mus)                               # (2,1)
    return mu_std * r_scale + r_mean                        # 역표준화 (2,1)


def build_mu_function(M: int) -> ca.Function:
    """대조·디버그용 ca.Function([z, p], [mu]) (실단위 2채널 평균)."""
    z = ca.SX.sym("z", D)
    p = ca.SX.sym("p", gp_extra_dim(M))
    return ca.Function("gp_mu", [z, p], [mu_ca_expr(z, p, M)])
