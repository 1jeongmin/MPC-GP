"""Phase 7 검증 — testing.md 게이트 7 (offline GP).

- 잔차 0 데이터 학습 -> GP 평균 ~0.
- 학습에서 멀어질수록 사후 std 단조 증가 (핵심 UQ 주장).
- numpy GP 예측과 CasADi 변환본 일치.
- 표준화 저장·재적용 시 예측 동일.
- MPC+GP 추종 < MPC+KF < MPC-only (GP 가 KF 를 이긴다).
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from src.config import load_group
from src.control.mpc_base import build_step_function
from src.control.mpc_gp import make_gp_mpc
from src.control.mpc_kf import make_kf_mpc
from src.control.mpc_nominal import NominalStepModel, make_nominal_mpc
from src.estimation.ekf import make_augmented_kf
from src.gp.casadi_export import build_mu_function, gp_param_vector
from src.gp.dataset import collect_residual_data
from src.gp.train_offline import load_gp, save_gp, train
from src.models.nonlinear_bicycle import make_nonlinear_rhs_np
from src.path.reference import Reference
from src.sim.runner import run_closed_loop


@pytest.fixture(scope="module")
def sedan():
    return load_group("vehicle", "sedan")


@pytest.fixture(scope="module")
def gp_cfg():
    return load_group("gp", "default")


@pytest.fixture(scope="module")
def trained(sedan, gp_cfg):
    """비선형 데이터로 학습한 GP + 데이터셋 (모듈 스코프 재사용)."""
    mpc = load_group("mpc", "default")
    sim = replace(load_group("sim", "default"), duration=10.0)
    path = load_group("path", "single_curve")
    ds = collect_residual_data(sedan, mpc, sim, path, plant="nonlinear")
    gp = train(ds, gp_cfg)
    return gp, ds


# --------------------------------------------------------------------------- #
# 게이트 7 — 잔차 0 -> 평균 ~0                                                   #
# --------------------------------------------------------------------------- #
def test_zero_residual_zero_mean(sedan, gp_cfg) -> None:
    """선형(매칭) 플랜트 데이터(잔차 ~0)로 학습 -> GP 평균 ~0."""
    mpc = load_group("mpc", "default")
    sim = replace(load_group("sim", "default"), duration=8.0)
    path = load_group("path", "single_curve")
    ds = collect_residual_data(sedan, mpc, sim, path, plant="linear")
    assert np.max(np.abs(ds.R)) < 1e-6, "매칭 잔차가 0이 아니다 (데이터 문제)"
    gp = train(ds, gp_cfg)
    mu = gp.predict_mean(ds.Z)
    assert np.max(np.abs(mu)) < 1e-5, f"잔차 0인데 GP 평균이 크다: {np.max(np.abs(mu))}"


# --------------------------------------------------------------------------- #
# 게이트 7 — 학습에서 멀수록 사후 std 증가 (핵심 UQ)                             #
# --------------------------------------------------------------------------- #
def test_posterior_std_grows_away_from_data(trained) -> None:
    """데이터 경계에서 바깥으로(min-거리 증가) 이동할수록 사후 std 단조 증가.

    데이터의 최대 delta 점에서 +delta 방향으로 벗어난다. 데이터 밖으로 나가면
    모든 학습점과의 거리가 단조 증가하므로 사후 std 가 사전분산으로 단조 수렴한다.
    """
    gp, ds = trained
    edge = ds.Z[np.argmax(ds.Z[:, 2])].copy()        # 최대 delta 학습점
    scale = ds.Z[:, 2].std()
    ts = np.linspace(0.0, 6.0, 15)                    # +delta 로 바깥 이동
    stds = np.array([np.sqrt(gp.predict_var(edge + np.array([0.0, 0.0, t * scale]))[0])
                     for t in ts])                    # (15, 2)
    for j, ch in enumerate(["v_y", "gamma"]):
        d = np.diff(stds[:, j])
        assert np.all(d >= -1e-9), f"{ch} 사후 std 가 단조 증가하지 않는다: {stds[:,j]}"
        assert stds[-1, j] > stds[0, j] * 1.5, f"{ch} 사후 std 가 충분히 안 커졌다"


# --------------------------------------------------------------------------- #
# 게이트 7 — numpy <-> CasADi 평균 일치                                         #
# --------------------------------------------------------------------------- #
def test_np_ca_mean_match(trained) -> None:
    gp, ds = trained
    M = len(gp.channels[0].Z)
    mu_ca = build_mu_function(M)
    p = gp_param_vector(gp)
    rng = np.random.default_rng(1)
    for _ in range(30):
        z = ds.Z[rng.integers(len(ds))] + rng.normal(scale=0.05, size=3)
        mnp = gp.predict_mean(z)[0]
        mca = np.array(mu_ca(z, p)).reshape(2)
        assert np.max(np.abs(mnp - mca)) < 1e-10, f"np={mnp} ca={mca}"


# --------------------------------------------------------------------------- #
# 게이트 7 — 표준화 저장·재적용                                                  #
# --------------------------------------------------------------------------- #
def test_save_load_roundtrip(trained, tmp_path: Path) -> None:
    gp, ds = trained
    p = tmp_path / "gp.npz"
    save_gp(p, gp)
    gp2 = load_gp(p)
    m1, m2 = gp.predict_mean(ds.Z), gp2.predict_mean(ds.Z)
    v1, v2 = gp.predict_var(ds.Z), gp2.predict_var(ds.Z)
    assert np.allclose(m1, m2, atol=1e-12) and np.allclose(v1, v2, atol=1e-12)


# --------------------------------------------------------------------------- #
# 게이트 7 — MPC+GP < MPC+KF < MPC-only (GP 가 KF 를 이긴다)                     #
# --------------------------------------------------------------------------- #
def test_gp_beats_kf_beats_only(sedan, trained) -> None:
    """비선형 플랜트에서 추종오차: MPC+GP < MPC+KF < MPC-only."""
    from src.eval.metrics import compute_metrics
    gp, _ = trained
    mpc = load_group("mpc", "default")
    ekf = load_group("ekf", "default")
    sim = replace(load_group("sim", "default"), duration=10.0)
    path = load_group("path", "single_curve")
    ref = Reference(path, sedan.vx_range)
    nom = build_step_function(NominalStepModel(sedan, sim.dt_ctrl))
    plant = make_nonlinear_rhs_np(sedan)
    NF = {"v_y": 9.57e-9}

    def ey(log):
        return compute_metrics(log, sedan, sim.dt_ctrl, NF)["tracking"]["rms_e_y"]

    e_only = ey(run_closed_loop(make_nominal_mpc(sedan, mpc, sim.dt_ctrl), ref, plant, nom, sim))
    kf = make_augmented_kf(sedan, ekf, sim.dt_ctrl)
    e_kf = ey(run_closed_loop(make_kf_mpc(sedan, mpc, sim.dt_ctrl, kf), ref, plant, nom, sim,
                              estimator=kf, rng=np.random.default_rng(0)))
    e_gp = ey(run_closed_loop(make_gp_mpc(sedan, mpc, sim.dt_ctrl, gp), ref, plant, nom, sim))

    assert e_kf < e_only, f"MPC+KF({e_kf:.3e}) 가 MPC-only({e_only:.3e}) 를 못 이김"
    assert e_gp < e_kf, f"MPC+GP({e_gp:.3e}) 가 MPC+KF({e_kf:.3e}) 를 못 이김"
