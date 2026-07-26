"""플롯 (.claude/rules/sim-experiment.md 플롯 규약).

내부 계산은 전부 rad. **deg 변환은 이 파일의 플롯 축에서만.**
그림 파일명에 run_id 를 포함한다. 제목은 ASCII (matplotlib 기본 폰트에 한글
글리프가 없어 깨진다).
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from src.config import CONFIGS_DIR, VehicleConfig
from src.path.reference import Reference


@lru_cache(maxsize=1)
def case_style() -> dict:
    """케이스·시나리오별 색/선종 고정 매핑 (configs/viz/case_style.yaml).

    그림마다 색이 바뀌면 논문 그림들 사이에서 케이스를 눈으로 쫓을 수 없다.
    매핑은 코드가 아니라 config 에 둔다 (sim-experiment.md 플롯 규약).
    """
    with (CONFIGS_DIR / "viz" / "case_style.yaml").open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _actual_xy(reference: Reference, s: np.ndarray, e_y: np.ndarray):
    """경로 위 실제 궤적 (x,y) 복원 — 시각화 전용.

    경로점에서 좌법선(-sin, cos) 방향으로 e_y 만큼 offset.
    """
    xp, yp, thp = reference.reconstruct_xy(ds=0.05)
    sp = np.linspace(0.0, reference.total_length, len(xp))
    s_c = np.clip(s, 0.0, reference.total_length)
    x_path = np.interp(s_c, sp, xp)
    y_path = np.interp(s_c, sp, yp)
    th = np.interp(s_c, sp, thp)
    x_act = x_path - e_y * np.sin(th)
    y_act = y_path + e_y * np.cos(th)
    return (xp, yp), (x_act, y_act)


def make_report_figure(log: dict[str, np.ndarray], reference: Reference,
                       vehicle: VehicleConfig, dt_ctrl: float,
                       out_dir: Path, run_id: str, tag: str = "") -> Path:
    """폐루프 결과 종합 그림을 저장하고 경로를 반환한다."""
    t = log["t"]
    x = log["x"]
    delta = log["delta"]
    s = log["s"]
    solve_time = log["solve_time"]
    residual = log["residual"]

    v_y, gamma, e_psi, e_y = x[:, 0], x[:, 1], x[:, 2], x[:, 3]

    fig, ax = plt.subplots(3, 2, figsize=(13, 11), constrained_layout=True)

    # (0,0) 궤적.
    (xp, yp), (xa, ya) = _actual_xy(reference, s, e_y)
    ax[0, 0].plot(xp, yp, "--", color="0.6", label="reference path")
    ax[0, 0].plot(xa, ya, "-", color="C0", lw=1.2, label="actual")
    ax[0, 0].set_aspect("equal", "box")
    ax[0, 0].set_title("Trajectory (x-y)")
    ax[0, 0].set_xlabel("x [m]"); ax[0, 0].set_ylabel("y [m]")
    ax[0, 0].legend(fontsize=8); ax[0, 0].grid(True, alpha=0.3)

    # (0,1) e_y, e_psi vs t (deg 축).
    a = ax[0, 1]
    a.plot(t, e_y, color="C0", label="e_y [m]")
    a.set_ylabel("e_y [m]", color="C0"); a.tick_params(axis="y", labelcolor="C0")
    a2 = a.twinx()
    a2.plot(t, np.degrees(e_psi), color="C1", label="e_psi [deg]")
    a2.set_ylabel("e_psi [deg]", color="C1"); a2.tick_params(axis="y", labelcolor="C1")
    a.set_title("Tracking error"); a.set_xlabel("t [s]"); a.grid(True, alpha=0.3)

    # (1,0) delta vs t + 제약선 (deg).
    a = ax[1, 0]
    a.plot(t, np.degrees(delta), color="C2", label="delta")
    dmax = np.degrees(vehicle.delta_max)
    a.axhline(dmax, ls="--", color="0.5"); a.axhline(-dmax, ls="--", color="0.5",
                                                     label="delta_max")
    a.set_title("Steering delta"); a.set_xlabel("t [s]"); a.set_ylabel("delta [deg]")
    a.legend(fontsize=8); a.grid(True, alpha=0.3)

    # (1,1) v_y, gamma vs t.
    a = ax[1, 1]
    a.plot(t, v_y, color="C0", label="v_y [m/s]")
    a.set_ylabel("v_y [m/s]", color="C0"); a.tick_params(axis="y", labelcolor="C0")
    a2 = a.twinx()
    a2.plot(t, np.degrees(gamma), color="C3", label="gamma [deg/s]")
    a2.set_ylabel("gamma [deg/s]", color="C3"); a2.tick_params(axis="y", labelcolor="C3")
    a.set_title("Dynamic states"); a.set_xlabel("t [s]"); a.grid(True, alpha=0.3)

    # (2,0) solve time 히스토그램 + dt_ctrl 기준선.
    a = ax[2, 0]
    a.hist(solve_time * 1e3, bins=30, color="C4", alpha=0.8)
    a.axvline(dt_ctrl * 1e3, ls="--", color="r", label=f"dt_ctrl={dt_ctrl*1e3:.0f}ms")
    a.set_title("Solve time"); a.set_xlabel("solve time [ms]"); a.set_ylabel("count")
    a.legend(fontsize=8); a.grid(True, alpha=0.3)

    # (2,1) 잔차 4채널 |r| (semilogy).
    a = ax[2, 1]
    labels = ["v_y", "gamma", "e_psi", "e_y"]
    for i, lab in enumerate(labels):
        a.semilogy(t, np.abs(residual[:, i]) + 1e-300, label=lab, lw=0.9)
    a.set_title("Residual |r| per channel"); a.set_xlabel("t [s]")
    a.set_ylabel("|r| (log)"); a.legend(fontsize=8, ncol=2); a.grid(True, alpha=0.3)

    title = f"Closed-loop report: {run_id}"
    if tag:
        title += f"  [{tag}]"
    fig.suptitle(title)

    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{tag}" if tag else ""
    path = out_dir / f"{run_id}{suffix}_report.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# Part 1 (Phase 8) — 케이스 비교 · 불확실성 지도                                  #
# --------------------------------------------------------------------------- #
def make_case_comparison_figure(results: dict[str, dict], out_dir: Path,
                                scenario: str) -> Path:
    """한 시나리오에서 세 케이스를 겹쳐 그린다 (색·선종은 config 고정).

    results: case 이름 -> run_experiment 반환 dict (log 를 포함).
    """
    style = case_style()["cases"]
    fig, ax = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)

    for case, res in results.items():
        st = style[case]
        log = res["log"]
        t = log["t"]
        kw = dict(color=st["color"], ls=st["linestyle"], lw=1.3, label=st["label"])
        ax[0, 0].plot(t, log["x"][:, 3], **kw)                       # e_y
        ax[0, 1].plot(t, np.degrees(log["delta"]), **kw)             # delta [deg]
        ax[1, 0].semilogy(t, np.abs(log["residual"][:, 0]) + 1e-300, **kw)   # |r_vy|
        ax[1, 1].semilogy(t, np.abs(log["residual"][:, 1]) + 1e-300, **kw)   # |r_gamma|

    ax[0, 0].set_title("Lateral error e_y"); ax[0, 0].set_ylabel("e_y [m]")
    ax[0, 1].set_title("Steering delta"); ax[0, 1].set_ylabel("delta [deg]")
    ax[1, 0].set_title("Residual |r_vy| (nominal-referenced)"); ax[1, 0].set_ylabel("|r| [m/s]")
    ax[1, 1].set_title("Residual |r_gamma| (nominal-referenced)"); ax[1, 1].set_ylabel("|r| [rad/s]")
    for a in ax.ravel():
        a.set_xlabel("t [s]"); a.grid(True, alpha=0.3); a.legend(fontsize=8)

    fig.suptitle(f"Part 1 case comparison - scenario {scenario}")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"part1_{scenario}_cases.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def _gp_std_grid(gp, ch: int, vy: np.ndarray, gam: np.ndarray, delta: float):
    """(v_y, gamma) 격자에서 채널 ch 의 GP 사후 표준편차. delta 는 고정 단면."""
    VY, GAM = np.meshgrid(vy, gam, indexing="ij")
    Z = np.column_stack([VY.ravel(), GAM.ravel(), np.full(VY.size, delta)])
    var = gp.predict_var(Z)[:, ch]
    return VY, GAM, np.sqrt(var).reshape(VY.shape)


def make_uncertainty_map_figure(gp, results: dict[str, dict], out_dir: Path,
                                scenario: str, trajectories: dict | None = None) -> Path:
    """**Part 1 주 그림** — GP 사후 std 의 상태공간 지도 vs KF P d-블록의 시간축.

    논점: GP 의 불확실성은 (v_y, gamma) 상태공간 위의 **지도**로 존재하고 학습 데이터가
    희소한 곳에서 커진다(epistemic). KF 의 P 는 시간축 값이라 같은 지도를 만들 수 없다
    (고정 Q_kf 를 전파한 aleatoric). 그래서 **같은 축에 억지로 올리지 않고** 나란히 둔다.

    trajectories: {라벨: (v_y, gamma)} 오버레이. None 이면 이 시나리오의 GP 궤적만.
    """
    sty = case_style()
    train_Z = gp.z_scaler.inverse(gp.channels[0].Z)      # 표준화 -> 물리 단위
    gp_log = results["gp"]["log"]

    if trajectories is None:
        trajectories = {scenario: (gp_log["x"][:, 0], gp_log["x"][:, 1])}

    # 격자 범위 = 학습점 + 모든 궤적을 덮되 약간 여유.
    vy_all = np.concatenate([train_Z[:, 0]] + [t[0] for t in trajectories.values()])
    ga_all = np.concatenate([train_Z[:, 1]] + [t[1] for t in trajectories.values()])
    pad_v = 0.25 * (vy_all.max() - vy_all.min() + 1e-9)
    pad_g = 0.25 * (ga_all.max() - ga_all.min() + 1e-9)
    vy = np.linspace(vy_all.min() - pad_v, vy_all.max() + pad_v, 90)
    gam = np.linspace(ga_all.min() - pad_g, ga_all.max() + pad_g, 90)
    delta_slice = float(np.median(gp_log["delta"]))

    fig, ax = plt.subplots(2, 2, figsize=(13.5, 9.5), constrained_layout=True)

    # (0,0)/(0,1) 두 채널의 사후 std 지도 + 학습점 + 궤적.
    for ch, (name, unit) in enumerate([("v_y", "m/s"), ("gamma", "rad/s")]):
        a = ax[0, ch]
        VY, GAM, STD = _gp_std_grid(gp, ch, vy, gam, delta_slice)
        cf = a.contourf(VY, np.degrees(GAM), STD, levels=24, cmap="viridis")
        fig.colorbar(cf, ax=a, label=f"GP posterior std [{unit}]")
        a.scatter(train_Z[:, 0], np.degrees(train_Z[:, 1]), s=9, c="w", edgecolors="k",
                  linewidths=0.4, label="GP training dict", zorder=3)
        for lab, (tv, tg) in trajectories.items():
            m = sty["scenarios"].get(lab, {}).get("marker", "o")
            a.plot(tv, np.degrees(tg), m, ms=2.5, alpha=0.75, label=f"traj {lab}", zorder=4)
        a.set_title(f"GP posterior std map: {name}  (delta slice = {np.degrees(delta_slice):.2f} deg)")
        a.set_xlabel("v_y [m/s]"); a.set_ylabel("gamma [deg/s]")
        a.legend(fontsize=7, loc="best")

    # (1,0) KF 의 P d-블록 대각 — 시간축에만 존재한다는 것이 논점.
    a = ax[1, 0]
    kf_log = results.get("kf", {}).get("log")
    if kf_log is not None and "ekf_P_diag" in kf_log:
        Pd = kf_log["ekf_P_diag"][:, 4:6]
        a.semilogy(kf_log["t"], Pd[:, 0], color=sty["cases"]["kf"]["color"], label="P[d_vy]")
        a.semilogy(kf_log["t"], Pd[:, 1], color=sty["cases"]["kf"]["color"], ls=":",
                   label="P[d_gamma]")
        a.legend(fontsize=8)
    a.set_title("KF: P d-block diag (time axis only - no state-space map)")
    a.set_xlabel("t [s]"); a.set_ylabel("variance (log)"); a.grid(True, alpha=0.3)

    # (1,1) 같은 궤적 위 GP 사후 std 의 시간축 — KF 와 직접 대비.
    a = ax[1, 1]
    if "gp_var" in gp_log:
        std = np.sqrt(gp_log["gp_var"])
        a.semilogy(gp_log["t"], std[:, 0], color=sty["cases"]["gp"]["color"], label="std[r_vy]")
        a.semilogy(gp_log["t"], std[:, 1], color=sty["cases"]["gp"]["color"], ls=":",
                   label="std[r_gamma]")
        a.legend(fontsize=8)
    a.set_title("GP: posterior std along the same trajectory")
    a.set_xlabel("t [s]"); a.set_ylabel("std (log)"); a.grid(True, alpha=0.3)

    fig.suptitle(f"Part 1 main figure - uncertainty: GP state-space map vs KF time axis "
                 f"[{scenario}]")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"part1_{scenario}_uncertainty.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
