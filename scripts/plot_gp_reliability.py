"""GP 신뢰도(캘리브레이션) 그림 — 평균±95% 구간(음영) + 실측 잔차, in-sample vs OOD.

`scripts/train_gp_loop10.py`가 이미 저장한 `gp_model.npz`/`dataset.npz`를 재사용해
(재시뮬레이션 없음) 학습 궤적(in-sample) 그림을 만들고, 이미 완료된 형상외삽 평가
런(4케이스의 ③`rtf_ay4` / ④`rtf_ay6` — GP가 loop10에서 학습되고 racetrack에서
평가되는 OOD 축, handoff.md 참조)의 로그를 그대로 읽어 out-of-sample 그림도 만든다.

**in-sample 그림은 신뢰도의 증거가 아니다** — 학습점 근방이라 잘 맞는 게 당연하다.
진짜 신뢰도 확인은 OOD 그림이다 (④에서 95% 커버리지 54.2%, NLPD 양전환 확인됨,
handoff.md 2026-08-03). 둘을 나란히 두어 그 차이를 직접 보게 한다.

밴드는 전부 **관측** 예측분산 `Var[y*] = Var[f*] + sigma_n^2` 로 그린다 — 비교 대상이
관측된 잔차이기 때문이다(2026-07-31 진단). `plot_gp_variance_report.py` 와 같은 규약.

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

# 이미 완료된 형상외삽(racetrack) 평가 런. **4케이스 체계의 ③④** 를 쓴다
# (2026-08-03 갱신). 이전에는 `part1_20260730_224024` 의 `rt_*` 를 하드코딩했는데,
# 그 축은 생략 확정됐고(handoff.md「4케이스 체계」) 그 로그의 `gp_var` 는 2026-07-31
# 에 고친 분산 공식 버그가 든 값이라 지금 그리면 틀린 밴드가 나온다.
OOD_TAGS = ("part1_gp_rtf_ay4", "part1_gp_rtf_ay6")


def _latest_log(tag: str) -> Path | None:
    """`results/<tag>_2026*/log.npz` 중 가장 최근 것. 없으면 None.

    `_warp`/`_het` 같은 변형은 날짜 앞에 접미사가 붙으므로 이 글롭에 걸리지 않는다
    (기본 설정 런만 잡힌다 — 의도된 동작).
    """
    runs = sorted((ROOT / "results").glob(f"{tag}_2026*/log.npz"))
    return runs[-1] if runs else None


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
    # 캘리브레이션 밴드는 **관측** 예측분산 Var[y*]=Var[f*]+sigma_n^2 이다 — 비교 대상이
    # 관측된 잔차이기 때문(`predict_var_obs` docstring, 2026-07-31 진단). 2026-08-03
    # 이전에는 여기서 `predict_var`(잠재)를 써서 밴드가 실제보다 좁게 그려졌다.
    var = gp.predict_var_obs(ds.Z)
    p = make_gp_calibration_band_figure(
        t_axis, ds.R, mean, var, run_dir, run_id,
        tag="in-sample (training trajectory, loop10 x10 laps)", x_label="t [s]",
    )
    print(f"[저장] {p}")

    # --- out-of-sample: 이미 계산된 racetrack 평가 로그 그대로 재사용 ---
    for tag in OOD_TAGS:
        log_path = _latest_log(tag)
        if log_path is None:
            print(f"[건너뜀] results/{tag}_2026*/log.npz 없음 — 먼저 그 케이스를 돌려라")
            continue
        log = np.load(log_path)
        # 비교 대상이 **관측된** 잔차이므로 Var[y*]=Var[f*]+sigma_n^2 를 쓴다
        # (2026-07-31 진단, plot_gp_variance_report.py 와 같은 규약).
        var = log["gp_var_obs"] if "gp_var_obs" in log.files else log["gp_var"]
        scenario = tag.replace("part1_gp_", "")
        p = make_gp_calibration_band_figure(
            log["t"], log["residual"][:, 0:2], log["gp_mean"], var,
            run_dir, run_id, tag=f"out-of-sample-{scenario} (racetrack, shape-OOD)",
            x_label="t [s]",
        )
        print(f"[출처] {log_path.parent.name}")
        print(f"[저장] {p}")


if __name__ == "__main__":
    main()
