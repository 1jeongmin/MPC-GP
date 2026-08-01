"""학습된 GP -> CasADi 표현 변환 (.claude/rules/gp-residual.md).

mu(z) = k(z_std, Z) @ alpha (표준화 공간) 후 역표준화해 실단위 잔차 반환.
Z/alpha/lengthscale/sigma_f/표준화통계를 CasADi **파라미터**로 주입한다 (온라인
갱신에도 NLP 재빌드 없이 값만 바뀌게 — mpc-solver.md). offline 이므로 값은 고정.

**numpy GP 예측과 CasADi 예측이 일치하는지 대조는 필수** (게이트 7).

파라미터 평탄화 레이아웃 (M = 딕셔너리 크기, 채널 2, 입력차원 d):
  Z(M*d) | alpha(M*2) | ls(2*d) | sf(2) | z_mean(d) | z_scale(d) | r_mean(2) | r_scale(2)
행렬은 order='F'(열우선) 로 평탄화 — CasADi ca.reshape(열우선)과 일치시키기 위함.

**입력차원 d 는 고정이 아니다** (2026-08-01): 기본 3 `[v_y, gamma, delta_prev]` 이지만
지연 특징(`GpConfig.n_lags`)을 쓰면 `3*(n_lags+1)` 이 된다. d 를 모듈 상수로 두면
lag 케이스에서 파라미터 레이아웃이 조용히 어긋나므로 **호출자가 넘기게** 한다
(값의 단일 출처는 `TwoChannelGP.input_dim`).
"""
from __future__ import annotations

import casadi as ca
import numpy as np

from src.gp.kernels import ard_rbf_ca
from src.gp.train_offline import TwoChannelGP

D_BASE = 3       # 지연 없을 때의 입력 차원 [v_y, gamma, delta_prev]
N_CH = 2         # 채널 [v_y, gamma]


def gp_extra_dim(M: int, d: int = D_BASE) -> int:
    """평탄화 파라미터 벡터 길이. d = 입력 차원(지연 포함)."""
    return M * d + M * N_CH + N_CH * d + N_CH + d + d + N_CH + N_CH


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


def mu_ca_expr(z, p, M: int, d: int = D_BASE):
    """CasADi 심볼릭 GP 평균 mu(z) (실단위, 2채널). z:(d,) 심볼, p:(extra_dim,) 파라미터.

    B_d 주입 전의 잔차 평균 [mu_vy, mu_gamma] 를 반환한다 (2,1).
    """
    o = 0
    Z = ca.reshape(p[o:o + M * d], M, d); o += M * d
    alpha = ca.reshape(p[o:o + M * N_CH], M, N_CH); o += M * N_CH
    ls = ca.reshape(p[o:o + N_CH * d], N_CH, d); o += N_CH * d
    sf = p[o:o + N_CH]; o += N_CH
    z_mean = p[o:o + d]; o += d
    z_scale = p[o:o + d]; o += d
    r_mean = p[o:o + N_CH]; o += N_CH
    r_scale = p[o:o + N_CH]; o += N_CH

    z_std = (ca.reshape(z, d, 1) - z_mean) / z_scale        # (d,1)
    mus = []
    for j in range(N_CH):
        k = ard_rbf_ca(z_std, Z, ls[j, :], sf[j])           # (M,1)
        mus.append(ca.mtimes(k.T, alpha[:, j]))             # (1,1) 표준화 평균
    mu_std = ca.vertcat(*mus)                               # (2,1)
    return mu_std * r_scale + r_mean                        # 역표준화 (2,1)


def build_mu_function(M: int, d: int = D_BASE) -> ca.Function:
    """대조·디버그용 ca.Function([z, p], [mu]) (실단위 2채널 평균)."""
    z = ca.SX.sym("z", d)
    p = ca.SX.sym("p", gp_extra_dim(M, d))
    return ca.Function("gp_mu", [z, p], [mu_ca_expr(z, p, M, d)])
