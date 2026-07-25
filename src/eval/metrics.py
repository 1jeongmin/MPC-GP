"""성능 지표 (.claude/rules/sim-experiment.md 지표).

- 추종: RMS e_y, max|e_y|, RMS e_psi
- 제어 노력: RMS delta, RMS delta_rate, 제약 접촉 비율
- 실시간성: solve time mean/p95/max, dt_ctrl 초과율, 수렴 실패율
- 모델: 잔차 RMS (채널별), noise floor 대비 배율

채널명: x = [v_y, gamma, e_psi, e_y]. 잔차도 같은 순서.
"""
from __future__ import annotations

import numpy as np

from src.config import VehicleConfig

CHANNELS = ("v_y", "gamma", "e_psi", "e_y")


def _rms(a: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(a, float) ** 2)))


def compute_metrics(log: dict[str, np.ndarray], vehicle: VehicleConfig, dt_ctrl: float,
                    noise_floor: dict[str, float] | None = None) -> dict:
    """로그에서 지표를 계산한다.

    noise_floor: 채널명 -> RMS noise floor (Phase 1 게이트 5 측정값). 주어지면
    잔차 RMS 의 배율을 함께 낸다.
    """
    x = log["x"]                 # (K, 4)
    delta = log["delta"]         # (K,)
    residual = log["residual"]   # (K, 4)
    solve_time = log["solve_time"]
    converged = log["converged"].astype(bool)

    e_psi, e_y = x[:, 2], x[:, 3]

    # 추종.
    tracking = {
        "rms_e_y": _rms(e_y),
        "max_abs_e_y": float(np.max(np.abs(e_y))),
        "rms_e_psi": _rms(e_psi),
    }

    # 제어 노력. delta_rate = diff(delta)/dt.
    ddelta = np.diff(delta, prepend=delta[0]) / dt_ctrl
    drate_max = vehicle.delta_rate_max
    contact_delta = float(np.mean(np.abs(delta) >= 0.99 * vehicle.delta_max))
    contact_rate = float(np.mean(np.abs(ddelta) >= 0.99 * drate_max))
    control = {
        "rms_delta": _rms(delta),
        "rms_delta_rate": _rms(ddelta),
        "contact_ratio_delta": contact_delta,
        "contact_ratio_rate": contact_rate,
    }

    # 실시간성.
    realtime = {
        "solve_time_mean": float(np.mean(solve_time)),
        "solve_time_p95": float(np.percentile(solve_time, 95)),
        "solve_time_max": float(np.max(solve_time)),
        "dt_ctrl_exceed_ratio": float(np.mean(solve_time > dt_ctrl)),
        "convergence_failure_rate": float(np.mean(~converged)),
    }

    # 모델 잔차 (채널별 RMS + noise floor 배율).
    resid = {}
    for i, ch in enumerate(CHANNELS):
        rms = _rms(residual[:, i])
        entry = {"rms": rms}
        if noise_floor is not None and ch in noise_floor and noise_floor[ch] > 0:
            entry["ratio_to_floor"] = rms / noise_floor[ch]
        resid[ch] = entry

    return {"tracking": tracking, "control": control, "realtime": realtime,
            "residual": resid}


def format_metrics(m: dict) -> str:
    """지표 dict 를 사람이 읽는 문자열로."""
    lines = []
    t = m["tracking"]
    lines.append(f"추종:  RMS e_y={t['rms_e_y']:.4e} m, max|e_y|={t['max_abs_e_y']:.4e} m, "
                 f"RMS e_psi={t['rms_e_psi']:.4e} rad")
    c = m["control"]
    lines.append(f"제어:  RMS delta={c['rms_delta']:.4e} rad, RMS drate={c['rms_delta_rate']:.4e} rad/s, "
                 f"접촉(delta/rate)={c['contact_ratio_delta']:.1%}/{c['contact_ratio_rate']:.1%}")
    r = m["realtime"]
    lines.append(f"실시간: solve mean={r['solve_time_mean']*1e3:.1f}ms p95={r['solve_time_p95']*1e3:.1f}ms "
                 f"max={r['solve_time_max']*1e3:.1f}ms, dt초과={r['dt_ctrl_exceed_ratio']:.1%}, "
                 f"수렴실패={r['convergence_failure_rate']:.1%}")
    lines.append("잔차 RMS (채널별, noise floor 배율):")
    for ch, e in m["residual"].items():
        ratio = f"  x{e['ratio_to_floor']:.1f}" if "ratio_to_floor" in e else ""
        lines.append(f"    {ch:6s} {e['rms']:.4e}{ratio}")
    return "\n".join(lines)
