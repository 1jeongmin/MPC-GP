"""범용 폐루프 실행기 — experiment config 이름을 인자로 받는다.

    python scripts/run_sim.py baseline_matched

run_part1.py / run_part2.py 는 이 스크립트 위의 배치 래퍼가 된다.
Phase 4 에서는 플랜트를 선형(=명목)으로 주입한다 (모델 완전일치).

results/<run_id>/ 에 로그(npz), 재현성 메타(meta.json), 종합 그림을 저장한다.
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_experiment
from src.control.mpc_base import build_step_function
from src.control.mpc_nominal import NominalStepModel, make_nominal_mpc
from src.eval.metrics import compute_metrics, format_metrics
from src.models.integrators import make_rhs_np
from src.path.reference import Reference
from src.sim.logger import save_npz, write_run_meta
from src.sim.runner import run_closed_loop
from src.viz.plots import make_report_figure

# Phase 1 게이트 5 에서 측정한 채널별 noise floor (sedan, vx=15, 1-step, RMS).
# 잔차 배율의 참조값. limo 등 다른 조건은 자릿수만 참고 (조건 의존).
NOISE_FLOOR_SEDAN = {"v_y": 9.57e-9, "gamma": 9.09e-10, "e_psi": 1.84e-10, "e_y": 5.67e-10}


def run_experiment(exp_name: str, tag: str = "") -> dict:
    """experiment 를 로드해 폐루프를 돌리고 결과를 저장한다. 지표 dict 반환."""
    exp = load_experiment(exp_name)
    if exp.path is None or exp.mpc is None:
        raise ValueError(f"experiment '{exp_name}' 에 path/mpc 참조가 필요하다.")

    vehicle, sim, path, mpc_cfg = exp.vehicle, exp.sim, exp.path, exp.mpc
    reference = Reference(path, vehicle.vx_range)

    controller = make_nominal_mpc(vehicle, mpc_cfg, sim.dt_ctrl)
    # 잔차용 명목 이산 전이 (MPC 예측과 동일 함수). 컨트롤러 종류와 무관하게 명목 사용.
    nominal_step = build_step_function(NominalStepModel(vehicle, sim.dt_ctrl))

    # Phase 4: 플랜트 = 선형(명목과 동일). 완전일치 케이스.
    plant_rhs = make_rhs_np(vehicle)

    log = run_closed_loop(controller, reference, plant_rhs, nominal_step, sim)

    metrics = compute_metrics(log, vehicle, sim.dt_ctrl, NOISE_FLOOR_SEDAN)

    run_id = (tag or exp_name) + "_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "results" / run_id
    save_npz(out_dir, log)
    write_run_meta(out_dir, run_id, exp.to_snapshot(), sim.seed,
                   extra={"metrics": metrics, "plant": "linear(matched)"})
    fig_path = make_report_figure(log, reference, vehicle, sim.dt_ctrl, out_dir,
                                  run_id, tag=tag)

    print(f"\n[run_id] {run_id}   (steps={len(log['t'])})")
    print(f"[saved]  {out_dir}")
    print(format_metrics(metrics))
    return metrics


def main() -> None:
    exp_name = sys.argv[1] if len(sys.argv) > 1 else "baseline_matched"
    run_experiment(exp_name, tag=exp_name)


if __name__ == "__main__":
    main()
