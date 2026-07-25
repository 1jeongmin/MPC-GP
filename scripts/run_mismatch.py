"""Phase 5 — 비선형 플랜트 잔차 특성화 (GP 설계의 입력).

플랜트=Fiala 비선형, MPC 예측=선형 명목. 잔차 = 타이어 비선형성.

산출물:
  1. 채널별 잔차 RMS 와 noise floor 대비 배율(=SNR)
  2. a_y_max 스윕별 SNR 곡선 (GP 를 어느 a_y 영역에서 돌릴지 근거)
  3. (v_y, gamma, delta) 산점도 + 잔차 크기 (GP 입력·커버리지·UQ 근거)

사용:  python scripts/run_mismatch.py
"""
from __future__ import annotations

import json
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_experiment
from src.control.mpc_base import build_step_function
from src.control.mpc_nominal import NominalStepModel, make_nominal_mpc
from src.eval.metrics import CHANNELS, compute_metrics
from src.models.integrators import make_rhs_np
from src.models.nonlinear_bicycle import make_nonlinear_rhs_np
from src.path.reference import Reference
from src.sim.logger import save_npz, write_run_meta
from src.sim.runner import run_closed_loop
from src.viz.plots import make_report_figure

# Phase 1 게이트 5 채널별 noise floor (sedan, vx=15).
NF = {"v_y": 9.57e-9, "gamma": 9.09e-10, "e_psi": 1.84e-10, "e_y": 5.67e-10}


def _build(vehicle, mpc_cfg, sim, path):
    reference = Reference(path, vehicle.vx_range)
    controller = make_nominal_mpc(vehicle, mpc_cfg, sim.dt_ctrl)
    nominal_step = build_step_function(NominalStepModel(vehicle, sim.dt_ctrl))
    return reference, controller, nominal_step


def run_one(vehicle, mpc_cfg, sim, path, plant_rhs):
    reference, controller, nominal_step = _build(vehicle, mpc_cfg, sim, path)
    return run_closed_loop(controller, reference, plant_rhs, nominal_step, sim)


def characterize(exp_name: str = "mismatch_sedan") -> dict:
    exp = load_experiment(exp_name)
    vehicle, sim, path, mpc_cfg = exp.vehicle, exp.sim, exp.path, exp.mpc
    sim = replace(sim, duration=10.0)   # 경로 전체 커버, 짧게

    run_id = f"{exp_name}_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "results" / run_id
    plant_nl = make_nonlinear_rhs_np(vehicle)

    # --- 1. 기준 비선형 런 + 채널별 SNR ---
    log = run_one(vehicle, mpc_cfg, sim, path, plant_nl)
    m = compute_metrics(log, vehicle, sim.dt_ctrl, NF)
    snr = {ch: m["residual"][ch]["ratio_to_floor"] for ch in CHANNELS}

    print(f"\n[run_id] {run_id}  (steps={len(log['t'])}, a_y_max={path.a_y_max})")
    print("채널별 잔차 RMS / noise floor 배율 (=학습 신호 SNR):")
    for ch in CHANNELS:
        print(f"  {ch:6s} RMS={m['residual'][ch]['rms']:.3e}  SNR x{snr[ch]:.0f}")
    print(f"추종 RMS e_y={m['tracking']['rms_e_y']:.4e} m, "
          f"수렴실패={m['realtime']['convergence_failure_rate']:.1%}")

    # --- 2. a_y_max 스윕 -> SNR 곡선 ---
    a_y_grid = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    sweep = {"a_y": [], "snr_v_y": [], "snr_gamma": [], "rms_v_y": [], "rms_gamma": [],
             "max_alpha_f": []}
    for ay in a_y_grid:
        p = replace(path, a_y_max=ay)
        lg = run_one(vehicle, mpc_cfg, sim, p, plant_nl)
        mm = compute_metrics(lg, vehicle, sim.dt_ctrl, NF)
        sweep["a_y"].append(ay)
        sweep["snr_v_y"].append(mm["residual"]["v_y"]["ratio_to_floor"])
        sweep["snr_gamma"].append(mm["residual"]["gamma"]["ratio_to_floor"])
        sweep["rms_v_y"].append(mm["residual"]["v_y"]["rms"])
        sweep["rms_gamma"].append(mm["residual"]["gamma"]["rms"])
        # 도달한 최대 전륜 슬립각 (타이어 작동 영역 지표).
        vy, gm = lg["x"][:, 0], lg["x"][:, 1]
        af = lg["delta"] - (vy + vehicle.a * gm) / np.maximum(lg["vx"], 1e-6)
        sweep["max_alpha_f"].append(float(np.max(np.abs(af))))

    print("\na_y_max 스윕 (SNR = 잔차/noise floor):")
    print("  a_y[m/s^2]  SNR(v_y)   SNR(gamma)  max|alpha_f|[deg]")
    for i, ay in enumerate(sweep["a_y"]):
        print(f"  {ay:5.1f}      x{sweep['snr_v_y'][i]:7.0f}  x{sweep['snr_gamma'][i]:7.0f}"
              f"   {np.degrees(sweep['max_alpha_f'][i]):.2f}")

    # --- 3. 저장 + 플롯 ---
    save_npz(out_dir, log)
    write_run_meta(out_dir, run_id, exp.to_snapshot(), sim.seed,
                   extra={"plant": "nonlinear(Fiala)", "metrics": m, "sweep": sweep,
                          "snr_baseline": snr})
    make_report_figure(log, Reference(path, vehicle.vx_range), vehicle, sim.dt_ctrl,
                       out_dir, run_id, tag="mismatch")
    _plot_sweep(sweep, out_dir, run_id)
    _plot_scatter(log, m, out_dir, run_id)
    print(f"[saved]  {out_dir}")
    return {"snr_baseline": snr, "sweep": sweep, "run_id": run_id, "out_dir": str(out_dir)}


def _plot_sweep(sweep, out_dir, run_id):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    ax[0].plot(sweep["a_y"], sweep["snr_v_y"], "o-", label="v_y")
    ax[0].plot(sweep["a_y"], sweep["snr_gamma"], "s-", label="gamma")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("a_y_max [m/s^2]"); ax[0].set_ylabel("SNR (residual / noise floor)")
    ax[0].set_title("Residual SNR vs lateral accel"); ax[0].grid(True, alpha=0.3); ax[0].legend()
    ax[1].plot(sweep["a_y"], np.degrees(sweep["max_alpha_f"]), "^-", color="C2")
    ax[1].set_xlabel("a_y_max [m/s^2]"); ax[1].set_ylabel("max |alpha_f| [deg]")
    ax[1].set_title("Peak front slip angle"); ax[1].grid(True, alpha=0.3)
    fig.suptitle(f"a_y sweep: {run_id}")
    fig.savefig(out_dir / f"{run_id}_snr_sweep.png", dpi=120)
    plt.close(fig)


def _plot_scatter(log, m, out_dir, run_id):
    """(v_y, gamma, delta) 공간 커버리지 + 잔차 크기. GP 입력·UQ 근거."""
    vy, gm = log["x"][:, 0], log["x"][:, 1]
    delta = log["delta"]
    r_dyn = np.sqrt(log["residual"][:, 0] ** 2 + log["residual"][:, 1] ** 2)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    s0 = ax[0].scatter(vy, np.degrees(gm), c=r_dyn, cmap="viridis", s=12)
    ax[0].set_xlabel("v_y [m/s]"); ax[0].set_ylabel("gamma [deg/s]")
    ax[0].set_title("Coverage in (v_y, gamma), color=|r_dyn|")
    fig.colorbar(s0, ax=ax[0], label="|residual v_y,gamma|")
    s1 = ax[1].scatter(np.degrees(gm), np.degrees(delta), c=r_dyn, cmap="viridis", s=12)
    ax[1].set_xlabel("gamma [deg/s]"); ax[1].set_ylabel("delta [deg]")
    ax[1].set_title("Coverage in (gamma, delta)")
    fig.colorbar(s1, ax=ax[1], label="|residual v_y,gamma|")
    fig.suptitle(f"Residual coverage: {run_id}")
    fig.savefig(out_dir / f"{run_id}_scatter.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    characterize(sys.argv[1] if len(sys.argv) > 1 else "mismatch_sedan")
