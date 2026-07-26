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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_experiment
from src.sim.logger import write_run_meta
from src.viz.plots import make_case_comparison_figure, make_uncertainty_map_figure

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
SCENARIOS = SCENARIOS_SYNTHETIC + SCENARIOS_RACETRACK + SCENARIOS_NOISY

# 케이스 간 **반드시 동일**해야 하는 스냅샷 키 (sim-experiment.md 「통제해야 할 변수」).
# 경로 형상·vx 프로파일은 path 에, MPC 지평·가중치·IPOPT 옵션은 mpc 에,
# 초기조건·seed·적분기 설정은 sim 에, 플랜트 모델은 plant 에 들어 있다.
CONTROLLED_KEYS = ("vehicle", "sim", "path", "mpc", "plant", "sensor")


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
