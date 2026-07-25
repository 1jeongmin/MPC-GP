"""플롯 (.claude/rules/sim-experiment.md 플롯 규약).

내부 계산은 전부 rad. **deg 변환은 이 파일의 플롯 축에서만.**
그림 파일명에 run_id 를 포함한다. 제목은 ASCII (matplotlib 기본 폰트에 한글
글리프가 없어 깨진다).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.config import VehicleConfig
from src.path.reference import Reference


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
