"""입력 워핑 강도(`warp_factor`) 선택 — **내부 OOD 분할**로 정한다.

    python scripts/tune_input_warp.py            # 기본 스윕
    python scripts/tune_input_warp.py 1 2 4 8    # factor 직접 지정

## 왜 이 스크립트가 따로 필요한가

`warp_factor` 는 **학습 데이터로 정할 수 없다.** "학습 범위 밖에서 얼마나 틀릴지"는
범위 안의 데이터가 알려줄 수 없기 때문이다. Type-II ML 도 못 고른다 — 범위 밖에서만
효과가 있는 파라미터라 학습 우도가 거의 무관심하다.

그렇다고 평가 시나리오(racetrack, a_y=6)의 캘리브레이션을 보고 고르면 **분식**이다
(handoff 「커버리지를 맞추려고 분산을 스케일링하지 마라」와 같은 부류).

그래서 **학습맵 안에서 강도 외삽 구조를 재현**한다:

    학습:   코너 강도(|gamma|) 하위 `SPLIT_Q`%  <- 순한 구간만 본다
    검증:   상위 (100-SPLIT_Q)%                 <- 학습 때 못 본 험한 구간

이것은 `a_y` 4->6 이 만드는 상황(같은 맵, 더 험한 작동점)과 **같은 구조**이면서
평가 시나리오를 전혀 건드리지 않는다. 선택 기준은 검증 구간의 **NLPD**다
(평균 정확도와 분산 크기를 함께 보는 표준 지표).

## 한계 — 정직하게

- 이 분할은 **강도 외삽**만 재현한다. 형상 외삽(racetrack)은 재현하지 못한다.
- 학습 데이터가 줄어들므로(순한 구간만) 여기서 고른 factor 가 전체 데이터로 학습한
  GP 에 그대로 최적이라는 보장은 없다. **근사적 선택**임을 명시한다.
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_experiment
from src.eval.calibration import calibration_metrics
from src.gp.dataset import ResidualDataset, load_dataset
from src.gp.train_offline import train
from src.sim.assemble import GP_TRAIN_CODE_ID

EVAL_EXPERIMENT = "part1_gp_lpf_ay4"     # gp 그룹(M, 하이퍼파라미터)을 여기서 가져온다
# 파일명에 학습 코드 신원을 박는다 — `compare_gp_features.py` 가 같은 이름으로 만든다
# (두 스크립트가 같은 데이터셋을 공유한다). 학습 코드가 바뀌면 이름이 갈려 옛 캐시를
# 재사용할 수 없다 (assemble.GP_TRAIN_CODE_ID).
DS_CACHE = ROOT / "data" / "gp_cache" / f"compare_features_{GP_TRAIN_CODE_ID}_dataset.npz"
DEFAULT_FACTORS = (1.0, 2.0, 4.0, 8.0)
SPLIT_Q = 80.0                            # 하위 80% 로 학습, 상위 20% 로 검증
N_EVAL = 4000
SEV_DIM = 1                               # |z[1]| = |gamma| 를 코너 강도 대리지표로


def severity_split(ds: ResidualDataset) -> tuple[ResidualDataset, ResidualDataset]:
    """코너 강도로 가른다 — 순한 구간 학습 / 험한 구간 검증 (강도 외삽 재현)."""
    sev = np.abs(ds.Z[:, SEV_DIM])
    thr = np.percentile(sev, SPLIT_Q)
    return ds.take(np.flatnonzero(sev < thr)), ds.take(np.flatnonzero(sev >= thr))


def main() -> None:
    factors = tuple(float(a) for a in sys.argv[1:]) or DEFAULT_FACTORS
    if not DS_CACHE.exists():
        raise SystemExit(f"학습 데이터 캐시가 없다: {DS_CACHE}\n"
                         "먼저 scripts/compare_gp_features.py 를 돌려 수집하라.")
    ds = load_dataset(DS_CACHE)
    fit_ds, held = severity_split(ds)
    base = load_experiment(EVAL_EXPERIMENT).gp

    # 검증 구간이 학습 범위 밖으로 얼마나 나가는지 — 분할이 의도대로인지 확인한다.
    lo, hi = np.percentile(fit_ds.Z, [1, 99], axis=0)
    out = np.mean((held.Z < lo) | (held.Z > hi), axis=0)
    print(f"\n{'='*84}\n[워핑 강도 선택]  내부 OOD 분할 (|gamma| 하위 {SPLIT_Q:.0f}% 학습 / "
          f"상위 {100-SPLIT_Q:.0f}% 검증)\n"
          f"  학습 {len(fit_ds)}  검증 {len(held)}  "
          f"검증점이 학습범위 밖인 비율: v_y {out[0]:.1%} gamma {out[1]:.1%} delta {out[2]:.1%}\n"
          f"  주 기준 = 검증 NLPD (낮을수록 좋음). **평가 시나리오는 쳐다보지 않는다.**\n"
          f"{'='*84}", flush=True)
    print(f"{'factor':>8} {'학습(s)':>9} {'NLPD':>9} {'z_std':>8} {'95%커버':>9} "
          f"{'Var[f*]/천장':>13}", flush=True)

    idx = np.unique(np.linspace(0, len(held) - 1, N_EVAL).astype(int))
    Zq, Rq = held.Z[idx], held.R[idx]
    rows = []
    for f in factors:
        t0 = time.time()
        gp = train(fit_ds, replace(base, warp_factor=f))
        train_s = time.time() - t0
        m = calibration_metrics(Rq, gp.predict_mean(Zq), gp.predict_var_obs(Zq))
        p = m["pooled"]
        # Var[f*] 가 천장의 몇 %까지 올라가는지 (워핑이 실제로 먹는지 확인)
        vf = gp.predict_var(Zq)
        ceil = np.array([c.sigma_f**2 for c in gp.channels]) * gp.r_scaler.scale**2
        sat = float(np.mean(vf / ceil))
        rows.append({"factor": f, "train_s": train_s, "nlpd": p["nlpd"],
                     "z_std": p["z_std"], "cov95": p["coverage"][0.95], "var_sat": sat,
                     "sigma_f": [c.sigma_f for c in gp.channels],
                     "lengthscales": [c.lengthscales.tolist() for c in gp.channels]})
        print(f"{f:>8.1f} {train_s:>9.1f} {p['nlpd']:>9.3f} {p['z_std']:>8.3f} "
              f"{p['coverage'][0.95]:>8.1%} {sat:>12.1%}", flush=True)

    best = min(rows, key=lambda r: r["nlpd"])
    print(f"\n[NLPD 최소] factor={best['factor']}  NLPD={best['nlpd']:.3f}  "
          f"z_std={best['z_std']:.3f}  95%커버={best['cov95']:.1%}")
    print("※ 이 값은 **강도 외삽만** 재현한 근사 선택이다. 형상 외삽(racetrack)은")
    print("  재현하지 못하며, 학습 데이터가 줄어든 상태에서 고른 값이다.")

    out_path = ROOT / "results" / "tuning" / (
        "tune_input_warp_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(
        {"feature_spec": GP_TRAIN_CODE_ID, "split": f"severity |gamma| q={SPLIT_Q}",
         "n_fit": len(fit_ds), "n_held": len(held), "n_eval": int(len(idx)),
         "eval_experiment": EVAL_EXPERIMENT, "rows": rows,
         "best_by_nlpd": best["factor"]},
        ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\n[저장] {out_path}")


if __name__ == "__main__":
    main()
