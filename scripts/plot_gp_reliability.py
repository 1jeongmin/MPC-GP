"""GP 신뢰도(캘리브레이션) 그림 — 평균±95% 구간(음영) + 실측 잔차, in-sample vs OOD.

`scripts/train_gp_loop10.py`가 이미 저장한 `gp_model.npz`/`dataset.npz`를 재사용해
(재시뮬레이션 없음) 학습 궤적(in-sample) 그림을 만들고, 이미 완료된 racetrack
평가 런(rt_ay4/rt_ay6, `part1_gp_rt_*` — GP가 loop10에서 학습되고 racetrack에서
평가되는 형상외삽/OOD 축, handoff.md 참조)의 로그를 그대로 읽어 out-of-sample
그림도 만든다.

**in-sample 그림은 신뢰도의 증거가 아니다** — 학습점 근방이라 잘 맞는 게 당연하다.
진짜 신뢰도 확인은 OOD 그림이다 (rt_ay6에서 커버리지 44.3%, NLPD 양전환 확인됨,
handoff.md 2026-07-30). 둘을 나란히 두어 그 차이를 직접 보게 한다.

    python scripts/plot_gp_reliability.py [training_run_dir]

인자를 생략하면 results/training_results/ 아래 가장 최근 런을 쓴다.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from src.config import load_group
from src.gp.dataset import load_dataset
from src.gp.train_offline import load_gp
from src.viz.plots import make_gp_calibration_band_figure

# 이미 완료된 racetrack 평가 런(형상외삽/OOD 축, handoff.md 2026-07-30 표 참조).
OOD_LOGS = {
    "rt_ay4": ROOT / "results" / "part1_20260730_224024" / "rt_ay4" / "gp" / "log.npz",
    "rt_ay6": ROOT / "results" / "part1_20260730_224024" / "rt_ay6" / "gp" / "log.npz",
}


def _latest_training_run() -> Path:
    base = ROOT / "results" / "training_results"
    runs = sorted(p for p in base.iterdir() if p.is_dir())
    if not runs:
        raise SystemExit(f"{base} 에 학습 런이 없다. 먼저 scripts/train_gp_loop10.py 를 돌려라.")
    return runs[-1]


def main() -> None:
    run_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else _latest_training_run()
    run_id = run_dir.name
    print(f"[학습 런] {run_dir}")

    # --- in-sample: 학습 궤적 위 GP 자체 적합 (신뢰도 증거 아님, 적합도 진단) ---
    gp = load_gp(run_dir / "gp_model.npz")
    ds = load_dataset(run_dir / "dataset.npz")
    dt_ctrl = load_group("sim", "gp_train_loop10").dt_ctrl
    t_axis = np.arange(len(ds.Z)) * dt_ctrl   # run_closed_loop 은 dt_ctrl 균일 스텝
    mean = gp.predict_mean(ds.Z)
    var = gp.predict_var(ds.Z)
    p = make_gp_calibration_band_figure(
        t_axis, ds.R, mean, var, run_dir, run_id,
        tag="in-sample (training trajectory, loop10 x10 laps)", x_label="t [s]",
    )
    print(f"[저장] {p}")

    # --- out-of-sample: 이미 계산된 racetrack 평가 로그 그대로 재사용 ---
    for scenario, log_path in OOD_LOGS.items():
        if not log_path.exists():
            print(f"[건너뜀] {log_path} 없음")
            continue
        log = np.load(log_path)
        p = make_gp_calibration_band_figure(
            log["t"], log["residual"][:, 0:2], log["gp_mean"], log["gp_var"],
            run_dir, run_id, tag=f"out-of-sample-{scenario} (racetrack, shape-OOD)",
            x_label="t [s]",
        )
        print(f"[저장] {p}")


if __name__ == "__main__":
    main()
