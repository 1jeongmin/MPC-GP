"""GP 특징 규약 비교 — delta 시점(A/B) x 지연 차원(n_lags 0/1/2).

    python scripts/compare_gp_features.py

## 무엇을 재는가

같은 학습 주행(loop10 10바퀴, 잡음+필터)에서 **특징 구성만 바꿔** GP 를 학습하고,
**랩 블록 홀드아웃**(1~9바퀴 학습 / 10바퀴째 평가)에서 예측분포 품질을 잰다.
나머지(M, 하이퍼파라미터 초기값, sigma_n_floor, seed)는 전부 고정이라 특징의 효과만
분리된다.

변형:
  옛 규약         3D — z=[v_y, gamma, **delta_k**]. 배포가 solve 전에 못 쓰는 값이다.
  기준선          3D — z=[v_y, gamma, delta_{k-1}]  (정합 수정본)
  full  lag1/2  6/9D — 상태·입력을 통째로 지연 (b_k, b_{k-1}, ...)
  delta lag1/2  4/5D — **입력 이력만** 붙임: [v_y, gamma, delta_{k-1}, delta_{k-2}, ...]

`delta` 모드가 사용자가 말한 "delta_k, delta_{k-1}, delta_{k-2} 로 차원 늘리기"에
해당한다 (인덱스가 한 칸 밀린 이유는 `delta_k` 를 배포가 알 수 없어서다).

## 왜 랩 블록 분할인가

기존 `tune_gp_M.py` 의 무작위 홀드아웃은 `gp-residual.md` 「학습 궤적과 평가 궤적을
분리한다」 위반이었고, 홀드아웃 점이 항상 학습점의 0.26 s 이웃이라 캘리브레이션을
낙관적으로 편향시켰다(상세는 `dataset.lap_block_split` docstring).

## 이 스크립트가 하지 않는 것

- M 재선택 (목적이 아니다 — M=2000 고정).
- `sigma_n_floor`·분산 사후 스케일링 (금지 — 분식).
- 폐루프 평가 (여기는 개루프 예측 품질만. 폐루프는 `run_sim.py part1_gp_*` 로 따로).
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
from src.gp.dataset import (ResidualDataset, apply_lags, lagged_dim, lap_block_split,
                            load_dataset, save_dataset)
from src.gp.train_offline import load_gp, save_gp, train
from src.path.reference import Reference
from src.sim.assemble import GP_TRAIN_CODE_ID, collect_training_dataset

EVAL_EXPERIMENT = "part1_gp_lpf_ay4"    # 여기서 gp 그룹(M, 하이퍼파라미터)을 가져온다
CACHE_DIR = ROOT / "data" / "gp_cache"
# 파일명에 학습 코드 신원을 박는다 — `tune_input_warp.py` 가 이 파일을 읽으므로 두
# 스크립트가 같은 식을 써야 한다 (assemble.GP_TRAIN_CODE_ID).
DS_CACHE = CACHE_DIR / f"compare_features_{GP_TRAIN_CODE_ID}_dataset.npz"
N_EVAL = 4000                            # 홀드아웃 평가 표본 상한 (분산이 O(n*M^2))

# 캐시 파일명 태그 — **분할·변형 정의**(이 스크립트 안에만 있는 개념)용이다.
# `src/gp/` 코드 변경은 `GP_TRAIN_CODE_ID` 가 자동으로 잡으므로 여기서 다루지 않는다.
# 아래 VARIANTS 나 분할 규칙을 바꿀 때만 올려라
# (2026-08-01 에 랩 분할 버그를 고치면서 v2).
SPLIT_TAG = "lapblock_v3_mlrestart"

# (라벨, delta 규약, n_lags, lag_mode)
VARIANTS = [
    ("delta_k     (옛 규약)", "applied", 0, "full"),
    ("delta_k-1   (기준선)", "prev", 0, "full"),
    ("+ full lag1", "prev", 1, "full"),
    ("+ full lag2", "prev", 2, "full"),
    ("+ delta lag1", "prev", 1, "delta"),
    ("+ delta lag2", "prev", 2, "delta"),
]


def get_dataset() -> tuple[ResidualDataset, float]:
    """학습 데이터셋 + 한 바퀴 호길이. 수집이 수 분이라 디스크에 캐시한다."""
    exp = load_experiment(EVAL_EXPERIMENT)
    tr = load_experiment(exp.gp.train_experiment)
    ref = Reference(tr.path, tr.vehicle.vx_range)
    if DS_CACHE.exists():
        print(f"[데이터] 캐시 사용: {DS_CACHE}")
        return load_dataset(DS_CACHE), ref.total_length
    print(f"[데이터] 수집 중 (experiment={exp.gp.train_experiment}, "
          f"n_laps={tr.sim.n_laps}) — 수 분 걸립니다...", flush=True)
    t0 = time.time()
    ds = collect_training_dataset(tr)
    print(f"[데이터] n={len(ds)}  수집 {time.time()-t0:.1f} s", flush=True)
    save_dataset(DS_CACHE, ds, tr.to_snapshot(), tr.sim.seed)
    return ds, ref.total_length


def with_delta_convention(ds: ResidualDataset, which: str) -> ResidualDataset:
    """z 의 3번째 성분을 delta_{k-1}(기본) 또는 delta_k(옛 규약)로 바꾼다."""
    if which == "prev":
        return ds
    if which != "applied":
        raise ValueError(f"알 수 없는 delta 규약: {which!r}")
    if ds.delta_applied is None:
        raise ValueError("delta_applied 가 없다 — 옛 데이터셋 캐시다. 지우고 다시 모아라.")
    Z = ds.Z.copy()
    Z[:, 2] = ds.delta_applied
    return ResidualDataset(Z, ds.R, {**ds.meta, "delta_feature": "applied"},
                           s=ds.s, t=ds.t, delta_applied=ds.delta_applied)


def evaluate(gp, held: ResidualDataset) -> dict:
    """홀드아웃 랩에서 예측분포 품질. 관측 예측분산 Var[y*] 를 쓴다(캘리브레이션).

    표본 추출은 **변형마다 동일**해야 한다(공유 rng 를 쓰면 변형마다 다른 점에서 재게
    되어 비교에 잡음이 섞인다). 그래서 결정론적 균일 간격으로 고른다.
    """
    idx = np.arange(len(held))
    if len(idx) > N_EVAL:
        idx = np.unique(np.linspace(0, len(held) - 1, N_EVAL).astype(int))
    Z, R = held.Z[idx], held.R[idx]
    m = calibration_metrics(R, gp.predict_mean(Z), gp.predict_var_obs(Z))
    p = m["pooled"]
    return {"n_held": int(len(idx)), "nlpd": p["nlpd"], "z_std": p["z_std"],
            "z_mean": p["z_mean"], "cov95": p["coverage"][0.95],
            "per_channel": {c: {"nlpd": e["nlpd"], "z_std": e["z_std"]}
                            for c, e in m["per_channel"].items()}}


def main() -> None:
    ds_raw, total_length = get_dataset()
    base_cfg = load_experiment(EVAL_EXPERIMENT).gp

    print(f"\n{'='*92}\n[특징 규약 비교]  n={len(ds_raw)}  M={base_cfg.M}  "
          f"랩 블록 분할(1~9 학습 / 10바퀴째 홀드아웃, 한 바퀴 {total_length:.1f} m)\n"
          f"주 기준 = 홀드아웃 NLPD(낮을수록 좋음)  |  진단 = z_std(목표 1.0), 95%커버\n"
          f"{'='*92}", flush=True)
    print(f"{'변형':<24} {'d':>3} {'학습(s)':>9} {'NLPD':>9} {'z_std':>8} "
          f"{'95%커버':>9} {'sigma_n(vy)':>12} {'sigma_n(gam)':>13}", flush=True)

    rows = []
    for label, conv, n_lags, lag_mode in VARIANTS:
        ds = with_delta_convention(ds_raw, conv)
        # ★ 순서 중요: 지연 -> 랩 분할 -> subsample(train 내부).
        # 지연을 분할 뒤에 붙여도 되지만, 분할 경계에서 홀드아웃 첫 스텝의 과거가
        # 학습 구간 점이 되어 미세한 누수가 생긴다. 먼저 붙이면 경계에서도 규약이
        # 배포와 같다(0 패딩이 아니라 실제 직전 스텝) — 대신 누수는 1~2 스텝뿐이라
        # 무시 가능하고, 배포 조건 재현이 더 중요하다.
        lagged = apply_lags(ds, n_lags, lag_mode)
        fit_ds, held = lap_block_split(lagged, total_length, n_holdout_laps=1)

        cfg = replace(base_cfg, n_lags=0)   # 지연은 위에서 이미 적용됨 (이중 적용 금지)
        cache = (CACHE_DIR /
                 f"compare_features_{SPLIT_TAG}_{conv}_{lag_mode}{n_lags}_M{cfg.M}.npz")
        if cache.exists():
            gp, train_s = load_gp(cache), float("nan")
        else:
            t0 = time.time()
            gp = train(fit_ds, cfg)
            train_s = time.time() - t0
            save_gp(cache, gp)

        ev = evaluate(gp, held)
        sn = [ch.sigma_n for ch in gp.channels]
        d = lagged_dim(n_lags, lag_mode)
        rows.append({"label": label, "delta_feature": conv, "n_lags": n_lags,
                     "lag_mode": lag_mode, "dim": d, "n_held": len(held),
                     "train_s": train_s, "sigma_n": sn, "n_fit": len(fit_ds),
                     "lengthscales": [ch.lengthscales.tolist() for ch in gp.channels],
                     "sigma_f": [ch.sigma_f for ch in gp.channels], **ev})
        print(f"{label:<24} {d:>3} {train_s:>9.1f} {ev['nlpd']:>9.3f} {ev['z_std']:>8.3f} "
              f"{ev['cov95']:>8.1%} {sn[0]:>12.4f} {sn[1]:>13.4f}", flush=True)

    out = ROOT / "results" / "tuning" / (
        "compare_gp_features_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"n_data": len(ds_raw), "M": base_cfg.M, "n_eval": N_EVAL,
         "eval_experiment": EVAL_EXPERIMENT, "split": "lap_block(holdout=last lap)",
         "lap_length_m": total_length, "rows": rows},
        ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\n[저장] {out}")
    print("\n※ 이 표는 **개루프 홀드아웃**이다. 폐루프 캘리브레이션과 같은 표에 섞지 마라 —")
    print("  둘의 격차 자체가 진단 대상이다(미결 #16).")


if __name__ == "__main__":
    main()
