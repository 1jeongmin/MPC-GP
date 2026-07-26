"""Phase 8 검증 — Part 1 실험 실행 + UQ 캘리브레이션.

게이트:
- 조립된 세 케이스의 통제변수가 동일 (run_part1 자동검증의 단위테스트).
- gp_var 가 매 스텝 로깅되고 전부 양수.
- 학습영역 밖에서 GP 평균 사후 std 가 증가 (분포이동이 실제로 잡히는지).
- 커버리지 계산기 자체 검증 (참 N(0,1) 합성표본 -> 95% 근처).
- KF d-블록 -> 이산 잔차 환산 G 의 정합성.
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from run_part1 import CASES, SCENARIOS, assert_controlled, experiment_name  # noqa: E402

from src.config import load_experiment, load_group
from src.eval.calibration import (calibration_metrics, disturbance_to_residual_ca,
                                  kf_residual_prediction)
from src.models.integrators import rk4_step
from src.models.linear_bicycle import dynamics_ca
from src.sim.assemble import build_case, run_case, train_gp_from_config


# --------------------------------------------------------------------------- #
# 통제변수 — 케이스를 가르는 차이는 ekf/gp 그룹의 유무뿐이어야 한다               #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("scenario", SCENARIOS)
def test_controlled_variables_identical(scenario: str) -> None:
    """세 케이스의 vehicle/sim/path/mpc/plant 가 완전히 같아야 한다."""
    controlled = assert_controlled(scenario)          # 다르면 예외
    assert controlled["plant"] == "nonlinear", "Part 1 은 비선형 플랜트여야 한다."


def test_control_check_catches_mismatch(tmp_path: Path) -> None:
    """통제변수 검증이 실제로 불일치를 잡는지 — 인위적으로 어긋뜨려 확인한다.

    검증이 통과만 하는지 보는 것으로는 부족하다. 실패를 유발해 실제로 막는지 본다.
    """
    import shutil

    import run_part1

    cfg_src = ROOT / "configs"
    cfg_dst = tmp_path / "configs"
    shutil.copytree(cfg_src, cfg_dst)
    # gp 케이스만 a_y 를 몰래 바꾼다 (통제변수 위반).
    bad = cfg_dst / "experiment" / "part1_gp_ay4.yaml"
    bad.write_text(bad.read_text(encoding="utf-8")
                   + "\noverrides:\n  path.a_y_max: 5.0\n", encoding="utf-8")

    orig = run_part1.load_experiment
    run_part1.load_experiment = lambda name: orig(name, configs_dir=cfg_dst)
    try:
        with pytest.raises(ValueError, match="통제변수"):
            run_part1.assert_controlled("ay4")
    finally:
        run_part1.load_experiment = orig


def test_case_builder_dispatch() -> None:
    """그룹 존재로 케이스가 갈리는지 (문자열 분기 없이)."""
    for case in CASES:
        exp = load_experiment(experiment_name(case, "ay4"))
        assert exp.plant == "nonlinear"
    assert load_experiment(experiment_name("gp", "ay4")).gp is not None
    assert load_experiment(experiment_name("kf", "ay4")).ekf is not None
    only = load_experiment(experiment_name("mpc_only", "ay4"))
    assert only.gp is None and only.ekf is None


def test_gp_train_experiment_is_separate() -> None:
    """GP 학습 출처가 config 로 고정되어 있고, 순환 참조가 아닌지."""
    gp_cfg = load_group("gp", "default")
    tr = load_experiment(gp_cfg.train_experiment)
    assert tr.gp is None, "학습 experiment 가 gp 그룹을 참조하면 순환이다."
    assert tr.plant == "nonlinear", "잔차가 없는 선형 플랜트로 학습할 수 없다."


# --------------------------------------------------------------------------- #
# 커버리지 계산기 자체 검증 (합성 분포)                                          #
# --------------------------------------------------------------------------- #
def test_calibration_on_synthetic_normal() -> None:
    """참 N(0,1) 표본에서 z_std~1, 68/95% 커버리지가 명목값에 근접해야 한다."""
    rng = np.random.default_rng(0)
    n = 200_000
    r = rng.normal(size=(n, 2))
    m = calibration_metrics(r, np.zeros((n, 2)), np.ones((n, 2)))
    p = m["pooled"]
    assert abs(p["z_std"] - 1.0) < 0.02, p
    assert abs(p["coverage"][0.68] - 0.68) < 0.01, p
    assert abs(p["coverage"][0.95] - 0.95) < 0.01, p
    # N(0,1) 의 NLPD 이론값 = 0.5*log(2*pi) + 0.5 = 1.4189.
    assert abs(p["nlpd"] - 1.4189) < 0.02, p


def test_calibration_detects_overconfidence() -> None:
    """분산을 4배 과소하게 주면 z_std~2, 커버리지가 명목 아래로 떨어져야 한다."""
    rng = np.random.default_rng(1)
    n = 50_000
    r = rng.normal(size=(n, 2))
    m = calibration_metrics(r, np.zeros((n, 2)), np.full((n, 2), 0.25))
    p = m["pooled"]
    assert 1.9 < p["z_std"] < 2.1, p
    assert p["coverage"][0.95] < 0.80, p


def test_calibration_rejects_nonpositive_variance() -> None:
    with pytest.raises(ValueError, match="사후분산"):
        calibration_metrics(np.zeros((3, 2)), np.zeros((3, 2)), np.zeros((3, 2)))


# --------------------------------------------------------------------------- #
# KF d-블록 -> 이산 잔차 환산 G                                                  #
# --------------------------------------------------------------------------- #
def test_G_matches_finite_difference() -> None:
    """G 가 '외란을 넣은 1스텝 전이 - 명목 1스텝 전이' 를 실제로 재현하는지.

    명목이 선형이므로 d 에 대해 정확히 선형이다: step(d) - step(0) == G @ d.
    """
    sedan = load_group("vehicle", "sedan")
    dt = 0.02
    G_fn = disturbance_to_residual_ca(sedan, dt)
    f_nom = dynamics_ca(sedan)

    import casadi as ca
    x_s = ca.SX.sym("x", 4); u_s = ca.SX.sym("u")
    vx_s = ca.SX.sym("vx"); k_s = ca.SX.sym("k"); d_s = ca.SX.sym("d", 2)

    def f_d(xx, uu, vv, kk):
        return f_nom(xx, uu, vv, kk) + ca.vertcat(d_s[0], d_s[1], 0.0, 0.0)

    step = ca.Function("s", [x_s, u_s, vx_s, k_s, d_s],
                       [rk4_step(f_d, x_s, u_s, (vx_s, k_s), dt)])

    rng = np.random.default_rng(0)
    for _ in range(10):
        x = rng.normal(scale=0.2, size=4)
        u = float(rng.normal(scale=0.05))
        vx = float(rng.uniform(8.0, 22.0))
        kap = float(rng.normal(scale=0.01))
        d = rng.normal(scale=0.5, size=2)
        lhs = np.array(step(x, u, vx, kap, d)).reshape(4) \
            - np.array(step(x, u, vx, kap, np.zeros(2))).reshape(4)
        rhs = np.array(G_fn(x, u, vx, kap)) @ d
        assert np.max(np.abs(lhs - rhs.reshape(4))) < 1e-12, (lhs, rhs)


def test_G_is_not_dt_identity() -> None:
    """G ~ dt*I 근사가 실제로 부적절한지 확인 (근사했으면 환산이 틀렸을 것).

    이 테스트는 '야코비안으로 정확히 뽑는다' 는 결정의 근거를 코드에 고정한다.
    """
    sedan = load_group("vehicle", "sedan")
    dt = 0.02
    G = np.array(disturbance_to_residual_ca(sedan, dt)(np.zeros(4), 0.0, 15.0, 0.0))[0:2, :]
    rel = np.max(np.abs(G / dt - np.eye(2)))
    assert rel > 0.05, f"G/dt 가 단위행렬에 너무 가깝다 (rel={rel}). 근사 논거 재확인 필요."


# --------------------------------------------------------------------------- #
# 폐루프 로깅 + 분포이동 (무거움: GP 학습 + 폐루프)                              #
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def gp_runs():
    """짧은 duration 으로 in-distribution / 외삽 두 시나리오의 GP 케이스를 돌린다."""
    out = {}
    gp = None
    for scenario in ("ay4", "ay6"):
        exp = load_experiment(experiment_name("gp", scenario))
        exp = replace(exp, sim=replace(exp.sim, duration=8.0))
        case = build_case(exp)
        out[scenario] = run_case(case, exp)
        gp = case.artifacts["gp"]
    return out, gp


def test_gp_var_logged_every_step(gp_runs) -> None:
    """gp_mean/gp_var 가 매 스텝 로깅되고 분산이 전부 양수인지.

    Phase 8 이전에는 아예 로깅되지 않았다 — 논문 주 증거물이 기록되지 않는 상태였다.
    """
    logs, _ = gp_runs
    for scenario, log in logs.items():
        assert "gp_var" in log and "gp_mean" in log, f"{scenario}: GP 로그가 없다"
        n = len(log["t"])
        assert log["gp_var"].shape == (n, 2), log["gp_var"].shape
        assert log["gp_mean"].shape == (n, 2), log["gp_mean"].shape
        assert np.all(log["gp_var"] > 0.0), f"{scenario}: 사후분산에 0 이하가 있다"
        assert np.all(np.isfinite(log["gp_mean"])), f"{scenario}: gp_mean 에 non-finite"


def test_posterior_std_grows_out_of_distribution(gp_runs) -> None:
    """학습영역 밖(a_y=6)에서 평균 사후 std 가 안(a_y=4)보다 커야 한다.

    이것이 논문의 핵심 UQ 주장이다. 절대 캘리브레이션(커버리지)과는 별개 주장이며,
    이 테스트는 '멀어지면 커진다' 는 **순서**만 검정한다.
    """
    logs, _ = gp_runs
    s_in = np.sqrt(logs["ay4"]["gp_var"]).mean(axis=0)
    s_out = np.sqrt(logs["ay6"]["gp_var"]).mean(axis=0)
    for j, ch in enumerate(("v_y", "gamma")):
        assert s_out[j] > s_in[j], (
            f"{ch}: 분포 밖 사후 std({s_out[j]:.3e}) 가 안({s_in[j]:.3e}) 보다 크지 않다")


def test_kf_residual_prediction_shapes_and_units() -> None:
    """KF 환산이 GP 와 같은 shape·같은 대상을 내는지."""
    exp = load_experiment(experiment_name("kf", "ay4"))
    exp = replace(exp, sim=replace(exp.sim, duration=6.0))
    case = build_case(exp)
    log = run_case(case, exp)
    mean, var = kf_residual_prediction(log, exp.vehicle, exp.sim.dt_ctrl)
    n = len(log["t"])
    assert mean.shape == (n, 2) and var.shape == (n, 2)
    assert np.all(var > 0.0), "환산된 KF 분산에 0 이하가 있다"
    # 환산 전 d_hat 은 연속 rate 라 이산 잔차보다 훨씬 크다 — 환산이 실제로 축소하는지.
    assert np.max(np.abs(mean)) < np.max(np.abs(log["ekf_d_hat"])), \
        "이산 환산이 크기를 줄이지 않았다 (G 적용이 빠졌나?)"
