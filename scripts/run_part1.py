"""Part 1 배치 실행기 — 3 케이스 x 3 시나리오 = 9 런 (Phase 8).

    python scripts/run_part1.py            # 전체 9 런
    python scripts/run_part1.py ay4        # 한 시나리오만

케이스 : mpc_only / kf / gp        (모델오차를 다루는 세 접근)
시나리오: ay4 / ay6 / dlc          (in-distribution / 강도 외삽 / 형상 외삽)

**주 산출물은 추종오차가 아니다.** GP 사후 std 의 상태공간 지도와 예측구간
커버리지다 (sim-experiment.md 「Part 1 의 주 지표는 캘리브레이션」).

## 통제변수 자동 검증

세 케이스가 정말 같은 조건인지 **눈으로 확인하지 않는다.** 조립된 config 스냅샷에서
통제 대상 그룹을 뽑아 diff 하고, 하나라도 다르면 **실행을 거부한다.** 케이스를 가르는
차이는 오직 ekf/gp 그룹의 유무여야 한다.
"""
from __future__ import annotations

import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_experiment
from src.sim.logger import write_run_meta
from src.viz.plots import (make_case_comparison_figure, make_gp_training_figure,
                           make_track_shape_figure, make_uncertainty_map_figure)

sys.path.insert(0, str(ROOT / "scripts"))
from run_sim import run_experiment  # noqa: E402

CASES = ("mpc_only", "kf", "gp")
# 합성 경로(우리가 만든 kappa(s)) + 외부 제공 레이싱 트랙.
# 트랙 시나리오를 따로 둔 이유: 합성 경로는 GP 가 학습한 것과 같은 생성기에서 나오므로
# 잔차 구조가 설계에 유리하게 치우쳤을 가능성이 남는다. 외부 트랙은 그 밖이다.
SCENARIOS_SYNTHETIC = ("ay4", "ay6", "dlc")
# rt_rturn = racetrack GP(#2)의 형상외삽 축 — right_turn.mat(76.2% 우회전 전용).
# a_y_max 는 rt_ay4 와 동일(4.0)하게 통제, 경로 형상만 다르다(dlc 와 같은 원칙).
SCENARIOS_RACETRACK = ("rt_ay4", "rt_ay6", "rt_rturn")
# rtn_* = 레이싱 트랙 + 현실적 센서 잡음 (Phase 8c). rt_* 와 sensor 그룹만 다르다.
# rtn_rturn = 형상외삽(rt_rturn) + 잡음 — rt_rturn 의 이례적 우위가 잡음에도 버티는지 확인.
SCENARIOS_NOISY = ("rtn_ay4", "rtn_ay6", "rtn_rturn")
# rtf_* = 잡음 + **공통 상태추정기**(Phase 8d). rtn_* 와 state_kf 그룹만 다르다.
# rtn_* 는 이제 "필터 없이 생측정값을 그대로 먹였을 때" 의 ablation 으로 남는다.
SCENARIOS_FILTERED = ("rtf_ay4", "rtf_ay6", "rtf_rturn")
# lpf_* = **학습맵(loop10)** + 잡음 + 필터 (2026-07-31). rtf_* 와 경로·sim 만 다르다.
# 사용자 지정 4케이스 분류: 1=lpf_ay4(학습맵·가속도 그대로) 2=lpf_ay6(학습맵·가속도 증가)
#                          3=rtf_ay4(racetrack·그대로)   4=rtf_ay6(racetrack·증가)
# ★ lpf_ay4 는 GP 의 **학습맵 그 자체**라 in-distribution 진단이다 — 일반화 근거로
#   쓰지 마라 (configs/experiment/part1_gp_lpf_ay4.yaml 헤더 참조).
SCENARIOS_LOOP10 = ("lpf_ay4", "lpf_ay6")
SCENARIOS = (SCENARIOS_SYNTHETIC + SCENARIOS_RACETRACK
             + SCENARIOS_NOISY + SCENARIOS_FILTERED + SCENARIOS_LOOP10)

# 케이스 간 **반드시 동일**해야 하는 스냅샷 키 (sim-experiment.md 「통제해야 할 변수」).
# 경로 형상·vx 프로파일은 path 에, MPC 지평·가중치·IPOPT 옵션은 mpc 에,
# 초기조건·seed·적분기 설정은 sim 에, 플랜트 모델은 plant 에 들어 있다.
CONTROLLED_KEYS = ("vehicle", "sim", "path", "mpc", "plant", "sensor", "state_kf")


def experiment_name(case: str, scenario: str) -> str:
    return f"part1_{case}_{scenario}"


def assert_controlled(scenario: str) -> dict:
    """한 시나리오의 세 케이스가 통제변수에서 동일한지 검증한다. 다르면 예외.

    반환: 공통 통제변수 스냅샷 (리포트에 남긴다).
    """
    snaps = {c: load_experiment(experiment_name(c, scenario)).to_snapshot() for c in CASES}
    ref_case = CASES[0]
    ref = {k: snaps[ref_case].get(k) for k in CONTROLLED_KEYS}

    diffs: list[str] = []
    for case in CASES[1:]:
        for key in CONTROLLED_KEYS:
            if snaps[case].get(key) != ref[key]:
                diffs.append(
                    f"  [{scenario}] '{key}' 가 {ref_case} 와 {case} 에서 다르다:\n"
                    f"    {ref_case}: {ref[key]}\n    {case}: {snaps[case].get(key)}"
                )
    if diffs:
        raise ValueError(
            "통제변수가 케이스마다 다르다. 이 상태의 비교는 무의미하므로 실행을 거부한다.\n"
            + "\n".join(diffs)
        )

    # 케이스를 가르는 차이는 ekf/gp 그룹의 유무뿐이어야 한다.
    marker = {c: (("gp" in snaps[c]), ("ekf" in snaps[c])) for c in CASES}
    if marker["mpc_only"] != (False, False) or marker["kf"] != (False, True) \
            or marker["gp"] != (True, False):
        raise ValueError(f"[{scenario}] 케이스 정의 그룹이 예상과 다르다: {marker}")
    return ref


def save_training_results(root_dir: Path, scenarios: tuple[str, ...]) -> Path | None:
    """GP **학습** 산출물을 `results/<run_id>/training_results/` 에 남긴다.

    평가 결과(각 시나리오 폴더)와 학습 결과를 같은 run_id 아래 두어, 어떤 GP 로
    돌린 실험인지 나중에 되짚을 수 있게 한다 (2026-07-31 사용자 요청).
    학습 자체는 이미 `train_gp_from_config` 가 (디스크 캐시로) 끝내 놓았으므로
    여기서 재주행하지 않는다 — `gp_training_artifacts` 가 캐시본을 돌려준다.
    """
    from src.gp.dataset import save_dataset
    from src.gp.train_offline import save_gp
    from src.path.reference import Reference
    from src.sim.assemble import gp_training_artifacts

    # **이번에 실제로 돌린 시나리오**에서 gp 그룹을 찾는다. 전체 SCENARIOS 를 훑으면
    # 안 돌린 시나리오의 GP(예: part1_gp_ay4 -> gp: default, M=100)를 집어서 실험이
    # 실제로 쓴 GP 와 다른 것을 기록하게 된다 (2026-07-31 에 그 버그를 냈다).
    gp_exps = [experiment_name("gp", s) for s in scenarios]
    exps = [e for e in (load_experiment(n) for n in gp_exps) if e.gp is not None]
    if not exps:
        return None
    exp = exps[0]
    srcs = {e.gp.train_experiment for e in exps}
    if len(srcs) > 1:
        raise ValueError(
            f"이번 배치의 시나리오들이 서로 다른 GP 학습 출처를 쓴다: {sorted(srcs)}. "
            "학습결과를 하나로 기록할 수 없다 — 배치를 나눠 실행하라."
        )

    tr, gp, ds = gp_training_artifacts(exp)
    out = root_dir / "training_results"
    out.mkdir(parents=True, exist_ok=True)

    save_dataset(out / "dataset.npz", ds, tr.to_snapshot(), tr.sim.seed)
    save_gp(out / "gp_model.npz", gp)
    rms = np.sqrt(np.mean(np.asarray(ds.R, float) ** 2, axis=0))
    write_run_meta(
        out, "training_results", tr.to_snapshot(), tr.sim.seed,
        extra={
            "train_experiment": exp.gp.train_experiment,
            "gp_config": {"M": exp.gp.M, "sigma_n_floor": exp.gp.sigma_n_floor,
                          "init_lengthscale": exp.gp.init_lengthscale,
                          "init_sigma_f": exp.gp.init_sigma_f,
                          "init_sigma_n": exp.gp.init_sigma_n,
                          "jitter": exp.gp.jitter},
            "n_train_steps": int(len(ds)),
            "n_dictionary": int(len(gp.channels[0].Z)),
            "noisy_sensor": bool(ds.meta.get("noisy_sensor")),
            "filtered_features": bool(ds.meta.get("filtered_features")),
            "residual_rms": {"v_y": float(rms[0]), "gamma": float(rms[1])},
            "hyperparameters": {
                name: {"lengthscales": ch.lengthscales.tolist(),
                       "sigma_f": float(ch.sigma_f), "sigma_n": float(ch.sigma_n)}
                for name, ch in zip(("v_y", "gamma"), gp.channels)},
        })

    reference = Reference(tr.path, tr.vehicle.vx_range)
    make_track_shape_figure(reference, out, "training_results")
    if ds.s is not None and ds.t is not None:
        make_gp_training_figure(gp, ds, exp.gp, out, "training_results", s=ds.s, t=ds.t)

    print(f"\n[학습결과] {out}")
    for name, ch in zip(("v_y", "gamma"), gp.channels):
        print(f"  {name:6s} M={len(ch.Z)}  ls={np.array2string(ch.lengthscales, precision=4)}  "
              f"sigma_f={ch.sigma_f:.4f}  sigma_n={ch.sigma_n:.4f}")
    return out


def _summary_row(scenario: str, case: str, res: dict) -> dict:
    """비교 요약 CSV 한 행. 지표를 평평하게 편다."""
    m, cal = res["metrics"], res["calibration"]
    row = {
        "scenario": scenario, "case": case, "run_id": res["run_id"],
        "rms_e_y": m["tracking"]["rms_e_y"],
        "max_abs_e_y": m["tracking"]["max_abs_e_y"],
        "rms_e_psi": m["tracking"]["rms_e_psi"],
        "rms_delta": m["control"]["rms_delta"],
        "solve_ms_mean": m["realtime"]["solve_time_mean"] * 1e3,
        "solve_ms_p95": m["realtime"]["solve_time_p95"] * 1e3,
        "dt_exceed": m["realtime"]["dt_ctrl_exceed_ratio"],
        "resid_rms_vy": m["residual"]["v_y"]["rms"],
        "resid_rms_gamma": m["residual"]["gamma"]["rms"],
    }
    if cal is not None:
        p = cal["metrics"]["pooled"]
        row.update({"uq_source": cal["source"], "z_std": p["z_std"],
                    "cov68": p["coverage"][0.68], "cov95": p["coverage"][0.95],
                    "nlpd": p["nlpd"]})
    else:
        row.update({"uq_source": "none", "z_std": "", "cov68": "", "cov95": "", "nlpd": ""})
    return row


def run_scenario(scenario: str, root_dir: Path) -> list[dict]:
    """한 시나리오의 세 케이스를 돌린다. 통제변수 검증을 먼저 통과해야 한다."""
    controlled = assert_controlled(scenario)
    print(f"\n{'='*72}\n[시나리오 {scenario}] 통제변수 검증 통과 "
          f"(a_y={controlled['path']['a_y_max']}, plant={controlled['plant']})\n{'='*72}")

    results, rows = {}, []
    for case in CASES:
        name = experiment_name(case, scenario)
        out = root_dir / scenario / case
        res = run_experiment(name, tag=name, out_dir=out)
        results[case] = res
        rows.append(_summary_row(scenario, case, res))

    make_case_comparison_figure(results, root_dir / scenario, scenario)
    gp_res = results["gp"]
    make_uncertainty_map_figure(gp_res["artifacts"]["gp"], results,
                                root_dir / scenario, scenario)
    return rows


def main() -> None:
    scenarios = tuple(sys.argv[1:]) or SCENARIOS
    for s in scenarios:
        if s not in SCENARIOS:
            raise SystemExit(f"알 수 없는 시나리오 {s!r}. 지원: {SCENARIOS}")

    run_id = "part1_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    root_dir = ROOT / "results" / run_id
    rows: list[dict] = []
    for scenario in scenarios:
        rows.extend(run_scenario(scenario, root_dir))

    # GP 학습 산출물을 같은 run_id 아래 남긴다 (어떤 GP 로 돌린 실험인지 되짚기 위함).
    save_training_results(root_dir, scenarios)

    # 최상위 비교 요약.
    csv_path = root_dir / "summary.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    (root_dir / "summary.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    # 배치 전체의 재현성 메타 (개별 런 메타는 각 케이스 폴더에 이미 있다).
    write_run_meta(root_dir, run_id, {"scenarios": list(scenarios), "cases": list(CASES)},
                   seed=None, extra={"summary": rows})

    print(f"\n{'='*72}\n[Part 1 요약]  {csv_path}")
    hdr = f"{'시나리오':<8}{'케이스':<10}{'RMS e_y':>10}{'z_std':>8}{'95% 커버':>10}{'NLPD':>10}"
    print(hdr)
    for r in rows:
        cov = f"{r['cov95']:.1%}" if r["cov95"] != "" else "-"
        zs = f"{r['z_std']:.3f}" if r["z_std"] != "" else "-"
        nl = f"{r['nlpd']:+.2f}" if r["nlpd"] != "" else "-"
        print(f"{r['scenario']:<8}{r['case']:<10}{r['rms_e_y']:>10.4e}{zs:>8}{cov:>10}{nl:>10}")


if __name__ == "__main__":
    main()
