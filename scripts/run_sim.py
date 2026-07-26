"""범용 폐루프 실행기 — experiment config 이름을 인자로 받는다.

    python scripts/run_sim.py part1_gp_ay4

케이스(명목 / +KF / +GP)와 플랜트(선형 / 비선형)는 **전부 config 에서 조립**한다
(src/sim/assemble.py). 스크립트에 케이스 분기가 없다 — 새 케이스를 붙여도 이 파일은
바뀌지 않는다 (CLAUDE.md 실험 구조).

results/<run_id>/ 에 로그(npz), 재현성 메타(meta.json), 종합 그림을 저장한다.
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_experiment
from src.eval.calibration import (calibration_metrics, format_calibration,
                                  kf_innovation_consistency, kf_residual_prediction)
from src.eval.metrics import compute_metrics, format_metrics
from src.sim.assemble import build_case, run_case
from src.sim.logger import save_npz, write_run_meta
from src.viz.plots import make_report_figure

# Phase 1 게이트 5 에서 측정한 채널별 noise floor (sedan, vx=15, 1-step, RMS).
# 잔차 배율의 참조값. limo 등 다른 조건은 자릿수만 참고 (조건 의존).
NOISE_FLOOR_SEDAN = {"v_y": 9.57e-9, "gamma": 9.09e-10, "e_psi": 1.84e-10, "e_y": 5.67e-10}


def compute_calibration(log, exp) -> dict | None:
    """로그에서 UQ 캘리브레이션을 계산한다 (Part 1 주 지표).

    대상은 두 방법 모두 **1스텝 이산 잔차** r_k 다. GP 는 (gp_mean, gp_var) 를 그대로
    쓰고, KF 는 연속 외란 추정을 G P_d G^T 로 이산 잔차 공간에 환산해서 쓴다.
    보정이 없는 명목 케이스는 예측분포 자체가 없으므로 None 을 반환한다.
    """
    r_true = log["residual"][:, 0:2]        # 동적 채널만 (기구학 채널은 진단용)

    if "gp_mean" in log:
        m = calibration_metrics(r_true, log["gp_mean"], log["gp_var"])
        return {"source": "gp", "metrics": m}
    if "ekf_d_hat" in log:
        mean, var = kf_residual_prediction(log, exp.vehicle, exp.sim.dt_ctrl)
        m = calibration_metrics(r_true, mean, var)
        return {"source": "kf", "metrics": m,
                "innovation_consistency": kf_innovation_consistency(log)}
    return None


def run_experiment(exp_name: str, tag: str = "", out_dir: Path | None = None) -> dict:
    """experiment 를 로드해 폐루프를 돌리고 결과를 저장한다. 요약 dict 반환."""
    exp = load_experiment(exp_name)
    case = build_case(exp)
    log = run_case(case, exp)

    metrics = compute_metrics(log, exp.vehicle, exp.sim.dt_ctrl, NOISE_FLOOR_SEDAN)
    calib = compute_calibration(log, exp)

    run_id = (tag or exp_name) + "_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = out_dir or (ROOT / "results" / run_id)
    save_npz(out_dir, log)
    write_run_meta(out_dir, run_id, exp.to_snapshot(), exp.sim.seed,
                   extra={"metrics": metrics, "calibration": calib,
                          "case": case.name, "plant": exp.plant})
    make_report_figure(log, case.reference, exp.vehicle, exp.sim.dt_ctrl, out_dir,
                       run_id, tag=tag or exp_name)

    print(f"\n[run_id] {run_id}   case={case.name} plant={exp.plant} "
          f"(steps={len(log['t'])})")
    print(f"[saved]  {out_dir}")
    print(format_metrics(metrics))
    if calib is not None:
        print(format_calibration(calib["metrics"], f"{case.name} ({calib['source']})"))
    return {"run_id": run_id, "case": case.name, "experiment": exp_name,
            "metrics": metrics, "calibration": calib, "out_dir": str(out_dir),
            "log": log, "reference": case.reference, "artifacts": case.artifacts,
            "exp": exp}


def main() -> None:
    exp_name = sys.argv[1] if len(sys.argv) > 1 else "part1_mpc_only_ay4"
    run_experiment(exp_name, tag=exp_name)


if __name__ == "__main__":
    main()
