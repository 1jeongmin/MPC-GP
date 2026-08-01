"""GP 딕셔너리 크기 M 선택 — 홀드아웃 예측성능으로 **측정해서** 고른다.

    python scripts/tune_gp_M.py                 # 기본 스윕
    python scripts/tune_gp_M.py 100 500 2000    # M 직접 지정

## 왜 M 인가 (2026-07-31 진단)

`train()` 은 `dataset.subsample(cfg.M)` 로 5만여 스텝 중 M 개만 남기고 그 M 개로
exact GP + Type-II ML 을 돌린다. M=100 이면 RBF 가 그 100점을 거의 보간해버려
`sigma_n` 이 "100점 사이 잔여 산포"만 설명하게 되고, 임의의 z 에서의 실제 조건부
산포보다 훨씬 작게 추정된다 -> **in-distribution 에서도 z_std=3.59 로 과신**.
M 을 키우면 이웃 점들의 잡음 실현이 서로 달라 GP 가 전부 보간할 수 없게 되고
`sigma_n` 이 참 조건부 산포로 수렴한다.

## ★ 홀드아웃 방식이 바뀌었다 (2026-08-01) — 옛 수치와 섞지 마라

처음에는 같은 주행 로그에서 **딕셔너리에 안 뽑힌 점**을 홀드아웃으로 썼다. 이는
`gp-residual.md` 「학습 궤적과 평가 궤적을 분리한다」 위반이고, 낙관적으로 편향된다:
52,665스텝에서 균일 2000점이면 딕셔너리 간격이 26스텝(0.53 s)이라 홀드아웃 점이 항상
학습점의 시간축 이웃이었다(50 Hz 시계열 = 강한 상관). **지금은 랩 블록 분할**이다
(1~9바퀴 학습 / 10바퀴째 평가, `dataset.lap_block_split`).

`configs/gp/loop10.yaml` 에 기록된 M 스윕 표(NLPD −6.206 / z_std 1.241 등)는 **옛
무작위 분할 수치**다. 이 스크립트를 다시 돌리면 값이 달라지는 것이 정상이며, 두
수치를 같은 표에 놓지 마라.

## 선택 기준

- **주 기준 = 홀드아웃 NLPD** (학습에 안 쓴 **랩**에서 평가).
  표준 모델선택 지표이고 **평균 정확도와 분산 크기를 함께** 본다.
  커버리지 숫자를 맞추는 것이 아니다 — 그건 `sigma_n_floor`·분산 스케일링과 같은
  분식이고 금지돼 있다. 여기서 바꾸는 것은 **학습 절차(M)** 뿐이다.
- **진단 = 홀드아웃 z_std(목표 1.0), sigma_n, 학습 벽시계 시간.**
- 선택 규칙: NLPD 최소. 다만 개선이 2% 미만으로 평평해지면 더 작은 M 을 택한다.

`M` 은 NLP 크기와 무관하다 — mu 는 solve 당 1회 평가돼 파라미터 2개로 주입된다
(`mpc_gp.py`). 커지는 것은 학습 O(M^3), 스텝당 분산 O(M^2), 메모리 O(M^2) 뿐이다.
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
from src.gp.dataset import ResidualDataset, lap_block_split, load_dataset, save_dataset
from src.gp.train_offline import load_gp, save_gp, train
from src.path.reference import Reference
from src.sim.assemble import FEATURE_SPEC, collect_training_dataset

DEFAULT_MS = (100, 250, 500, 1000, 2000, 4000)
EVAL_EXPERIMENT = "part1_gp_lpf_ay4"      # in-distribution. 여기서 gp 그룹을 가져온다.
N_EVAL = 4000                              # 홀드아웃 평가 표본 수 (분산이 O(n*M^2))
# 캐시 파일명에 특징 규약을 박는다 — 규약이 바뀌면 옛 캐시를 **재사용할 수 없게**
# 이름이 갈린다(production 캐시가 해시로 하는 일과 같은 목적, assemble.FEATURE_SPEC).
CACHE = ROOT / "data" / "gp_cache" / f"tune_gp_M_{FEATURE_SPEC}_dataset.npz"


def get_dataset() -> tuple[ResidualDataset, float]:
    """학습 데이터셋(10바퀴, 잡음+필터) + 한 바퀴 호길이. 수집만 ~9분이라 캐시한다."""
    exp = load_experiment(EVAL_EXPERIMENT)
    tr = load_experiment(exp.gp.train_experiment)
    lap_len = Reference(tr.path, tr.vehicle.vx_range).total_length
    if CACHE.exists():
        print(f"[데이터] 캐시 사용: {CACHE}")
        return load_dataset(CACHE), lap_len
    print(f"[데이터] 수집 중 (experiment={exp.gp.train_experiment}, "
          f"n_laps={tr.sim.n_laps}) — 수 분 걸립니다...", flush=True)
    t0 = time.time()
    ds = collect_training_dataset(tr)
    print(f"[데이터] n={len(ds)}  수집 {time.time()-t0:.1f} s")
    save_dataset(CACHE, ds, tr.to_snapshot(), tr.sim.seed)
    return ds, lap_len


def evaluate(gp, held: ResidualDataset, rng: np.random.Generator) -> dict:
    """홀드아웃 **랩**에서 예측분포 품질을 잰다 (학습에 안 쓴 궤적)."""
    idx = np.arange(len(held))
    if len(idx) > N_EVAL:
        idx = rng.choice(idx, N_EVAL, replace=False)
    Z, R = held.Z[idx], held.R[idx]
    mean = gp.predict_mean(Z)
    # **관측** 예측분산 Var[y*] = Var[f*] + sigma_n^2. 잠재분산으로 재면 M 이 커질수록
    # Var[f*] 가 데이터 근처에서 붕괴해 "M 을 키우면 나빠진다"는 잘못된 결론이 나온다
    # (2026-07-31 진단 — M=1000 에서 sigma_n^2 이 Var[f*] 의 15배였다).
    var = gp.predict_var_obs(Z)
    m = calibration_metrics(R, mean, var)
    p = m["pooled"]
    return {"n_held": int(len(idx)), "nlpd": p["nlpd"], "z_std": p["z_std"],
            "z_mean": p["z_mean"], "cov95": p["coverage"][0.95],
            "per_channel": {c: {"nlpd": e["nlpd"], "z_std": e["z_std"]}
                            for c, e in m["per_channel"].items()}}


def main() -> None:
    Ms = tuple(int(a) for a in sys.argv[1:]) or DEFAULT_MS
    ds, lap_len = get_dataset()
    fit_ds, held = lap_block_split(ds, lap_len, n_holdout_laps=1)
    base_gp_cfg = load_experiment(EVAL_EXPERIMENT).gp
    rng = np.random.default_rng(0)

    print(f"\n{'='*84}\n[M 스윕]  데이터 n={len(ds)}  후보={Ms}\n"
          f"랩 블록 분할: 학습 {len(fit_ds)} / 홀드아웃 {len(held)} (마지막 1바퀴, "
          f"한 바퀴 {lap_len:.1f} m)\n"
          f"주 기준 = 홀드아웃 NLPD (낮을수록 좋음)  |  진단 = z_std(목표 1.0), sigma_n\n"
          f"{'='*84}", flush=True)
    print(f"{'M':>6} {'학습(s)':>9} {'NLPD':>10} {'z_std':>8} {'95%커버':>9} "
          f"{'sigma_n(vy)':>12} {'sigma_n(gam)':>13}", flush=True)

    rows = []
    for M in Ms:
        cfg = replace(base_gp_cfg, M=M)
        # 학습된 GP 를 M 별로 캐시한다 — M=4000 은 학습만 40분이라 재평가 때마다
        # 다시 돌릴 수 없다. 데이터셋·M 이 같으면 train() 은 결정론적이다.
        gp_cache = CACHE.with_name(f"tune_gp_M_{FEATURE_SPEC}_model_M{M}.npz")
        if gp_cache.exists():
            gp, train_s = load_gp(gp_cache), float("nan")
        else:
            t0 = time.time()
            gp = train(fit_ds, cfg)
            train_s = time.time() - t0
            save_gp(gp_cache, gp)
        ev = evaluate(gp, held, rng)
        sn = [ch.sigma_n for ch in gp.channels]
        rows.append({"M": M, "train_s": train_s, "sigma_n": sn,
                     "lengthscales": [ch.lengthscales.tolist() for ch in gp.channels],
                     "sigma_f": [ch.sigma_f for ch in gp.channels], **ev})
        print(f"{M:>6} {train_s:>9.1f} {ev['nlpd']:>10.3f} {ev['z_std']:>8.3f} "
              f"{ev['cov95']:>8.1%} {sn[0]:>12.4f} {sn[1]:>13.4f}", flush=True)

    best = min(rows, key=lambda r: r["nlpd"])
    # NLPD 개선이 2% 미만이면 더 작은 M 을 택한다 (수확체감 지점).
    thresh = abs(best["nlpd"]) * 0.02
    frugal = min((r for r in rows if r["nlpd"] <= best["nlpd"] + thresh),
                 key=lambda r: r["M"])
    print(f"\n[NLPD 최소]      M={best['M']}  NLPD={best['nlpd']:.3f}  z_std={best['z_std']:.3f}")
    print(f"[수확체감 지점]  M={frugal['M']}  NLPD={frugal['nlpd']:.3f}  "
          f"z_std={frugal['z_std']:.3f}  (NLPD 차이 2% 이내 중 최소 M)")

    out = ROOT / "results" / "tuning" / (
        "tune_gp_M_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"n_data": len(ds), "candidates": list(Ms), "n_eval": N_EVAL,
         "eval_experiment": EVAL_EXPERIMENT, "feature_spec": FEATURE_SPEC,
         "split": "lap_block(holdout=last lap)", "n_fit": len(fit_ds),
         "n_holdout": len(held), "rows": rows,
         "best_by_nlpd": best["M"], "diminishing_returns": frugal["M"]},
        ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\n[저장] {out}")


if __name__ == "__main__":
    main()
