"""KF 재튜닝 하네스 — 공통 상태추정기(state_kf) / 모델보정 KF(ekf).

    python scripts/tune_kf.py state_kf     # (1) 공통 상태추정기
    python scripts/tune_kf.py ekf          # (2) 모델보정 KF

## 목적함수 — 병행 기준 (2026-07-30 사용자 승인)

**주 지표(선택 기준)** 는 미결 #4 확정대로 추정/외란 RMSE 다. **추종오차(RMS e_y)로
고르지 않는다** — `configs/ekf/realistic_sensor.yaml` 의 이전 튜닝이 바로 그 위반이었다.

| 대상 | 케이스 | 주 지표 | 부 지표 |
|---|---|---|---|
| state_kf | part1_mpc_only_rtf_ay4 | 정규화 추정 RMSE 합 | mean(NIS) vs n_meas |
| ekf      | part1_kf_rtf_ay4       | 외란 RMSE (이산 잔차 공간) | mean(NIS) vs n_meas |

**부 지표 NIS** 는 "P 가 옳은 크기인가" 를 본다 (일관된 필터면 E[NIS]=n_meas=4).
주 지표는 점추정치의 정확도라 P 의 크기를 보지 못하므로 둘을 나란히 낸다.
캘리브레이션(z_std/커버리지)은 **맞추는 대상이 아니라 결과로만** 표에 싣는다
(「커버리지 맞추려고 분산 스케일링 금지」).

## 프로토콜

- 튜닝 시나리오는 **in-distribution(rtf_ay4) 하나뿐**이다. rtf_ay6(강도 외삽)는
  열람하지 않는다 — 기존 `tuning_selection_scenario` 규약과 동일.
- 손잡이는 **Q 만**. R 은 센서 사양(noise_std^2)에서 온 값이라 건드리지 않는다.
- config 변형은 `dataclasses.replace` 로 만든다 — `src/config.py` 의
  `_apply_overrides` 와 같은 방식이라 `__post_init__` 유효성 검사를 그대로 탄다.
- state_kf 는 **mpc_only** 케이스로 튜닝한다. 모델 보정자가 끼면 상태추정 품질만
  분리해서 볼 수 없다.
"""
from __future__ import annotations

import json
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import ExperimentConfig, load_experiment
from src.eval.calibration import calibration_metrics, kf_residual_prediction
from src.sim.assemble import build_case, run_case

# 현재 config 값 대비 배수. 과신(P 과소)이 의심되므로 위쪽을 넓게 잡는다.
SCALES = (0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)

TUNE_SCENARIO = {                       # in-distribution 만. ay6 열람 금지.
    "state_kf": "part1_mpc_only_rtf_ay4",
    "ekf": "part1_kf_rtf_ay4",
}


# --------------------------------------------------------------------------- #
# config 변형 — 동적/외란 블록의 Q 만 스케일한다                                  #
# --------------------------------------------------------------------------- #
def scaled_experiment(exp: ExperimentConfig, which: str, scale: float) -> ExperimentConfig:
    """Q 의 해당 블록만 `scale` 배 한 새 experiment 를 만든다.

    state_kf: Q_diag 의 동적 채널(v_y, gamma). 기구학 채널(1e-8)은 고정 —
              vx·kappa 가 정확하면 오차가 없는 식이라 잡음을 키울 근거가 없다.
    ekf:      Q_kf_diag 의 d-블록(4:6). 상태 블록은 고정 — 모델오차를 흡수하는 것은
              d 이지 상태가 아니다.
    """
    if which == "state_kf":
        q = list(exp.state_kf.Q_diag)
        q[0] *= scale
        q[1] *= scale
        return replace(exp, state_kf=replace(exp.state_kf, Q_diag=tuple(q)))
    q = list(exp.ekf.Q_kf_diag)
    q[4] *= scale
    q[5] *= scale
    return replace(exp, ekf=replace(exp.ekf, Q_kf_diag=tuple(q)))


# --------------------------------------------------------------------------- #
# 지표                                                                          #
# --------------------------------------------------------------------------- #
def estimation_rmse_norm(log: dict, noise_std: tuple[float, ...]) -> tuple[float, list[float]]:
    """정규화 추정 RMSE 합 = sum_ch RMS(x_fb - x_true)/noise_std[ch].

    1.0/채널 = 생측정 수준(필터가 아무것도 못 한 것), <1 = 개선.
    `configs/state_kf/default.yaml` 에 기록된 기존 목적함수와 **같은 정의**다.
    """
    err = np.asarray(log["x_fb"], float) - np.asarray(log["x"], float)   # (K,4)
    per = [float(np.sqrt(np.mean(err[:, i] ** 2)) / noise_std[i]) for i in range(4)]
    return float(sum(per)), per


def disturbance_rmse(log: dict, exp: ExperimentConfig) -> tuple[float, list[float]]:
    """외란 RMSE — 이산 잔차 공간에서 G@d_hat 이 참 잔차를 얼마나 맞히는가.

    연속 외란 d 와 이산 잔차는 단위가 다르므로 `kf_residual_prediction` 의 야코비안
    환산을 **재사용**한다 (수기 미분·복사 금지, calibration.py 와 같은 G).
    """
    mean, _ = kf_residual_prediction(log, exp.vehicle, exp.sim.dt_ctrl)
    r_true = np.asarray(log["residual"], float)[:, 0:2]
    err = mean - r_true
    per = [float(np.sqrt(np.mean(err[:, i] ** 2))) for i in range(2)]
    # 채널 단위가 다르므로(m/s vs rad/s) 합이 아니라 각 채널을 그 채널 참잔차 RMS 로
    # 정규화해 더한다 — 1.0 = "보정 안 한 것과 같음", <1 = 잔차를 실제로 줄임.
    base = [float(np.sqrt(np.mean(r_true[:, i] ** 2))) for i in range(2)]
    norm = [p / b if b > 0 else float("nan") for p, b in zip(per, base)]
    return float(sum(norm)), norm


def nis_stats(log: dict, key: str) -> dict:
    """평균/중앙 NIS. 일관된 필터면 mean ~ n_meas(=4), median ~ chi2_4 중앙값 3.357."""
    if key not in log:
        return {"mean": float("nan"), "median": float("nan")}
    v = np.asarray(log[key], float)
    v = v[len(v) // 10:]                     # 초기 과도(앞 10%) 제외
    return {"mean": float(np.mean(v)), "median": float(np.median(v))}


def calibration_of(log: dict, exp: ExperimentConfig) -> dict | None:
    """캘리브레이션 — **선택 기준이 아니라 결과 보고용**."""
    if "ekf_d_hat" not in log:
        return None
    mean, var = kf_residual_prediction(log, exp.vehicle, exp.sim.dt_ctrl)
    m = calibration_metrics(np.asarray(log["residual"], float)[:, 0:2], mean, var)
    p = m["pooled"]
    return {"z_std": p["z_std"], "cov95": p["coverage"][0.95], "nlpd": p["nlpd"]}


# --------------------------------------------------------------------------- #
# 그리드 실행                                                                    #
# --------------------------------------------------------------------------- #
def run_grid(which: str) -> list[dict]:
    base = load_experiment(TUNE_SCENARIO[which])
    if which == "state_kf" and base.state_kf is None:
        raise SystemExit(f"{TUNE_SCENARIO[which]} 가 state_kf 그룹을 참조하지 않는다.")
    if which == "ekf" and base.ekf is None:
        raise SystemExit(f"{TUNE_SCENARIO[which]} 가 ekf 그룹을 참조하지 않는다.")
    if base.sensor is None:
        raise SystemExit("튜닝 시나리오에 sensor 그룹이 없다 (잡음 없는 조건은 무의미).")

    nis_key = "skf_nis" if which == "state_kf" else "ekf_nis"
    rows: list[dict] = []
    for scale in SCALES:
        exp = scaled_experiment(base, which, scale)
        log = run_case(build_case(exp), exp)

        row: dict = {"scale": scale, "nis": nis_stats(log, nis_key)}
        if which == "state_kf":
            total, per = estimation_rmse_norm(log, base.sensor.noise_std)
            row["objective"] = total
            row["per_channel"] = per
        else:
            total, per = disturbance_rmse(log, exp)
            row["objective"] = total
            row["per_channel"] = per
        # 진단용(선택 기준 아님).
        row["rms_e_y"] = float(np.sqrt(np.mean(np.asarray(log["x"], float)[:, 3] ** 2)))
        row["calibration"] = calibration_of(log, exp)
        rows.append(row)

        cal = row["calibration"]
        cal_s = ("-" if cal is None
                 else f"z_std={cal['z_std']:6.3f} cov95={cal['cov95']:6.1%}")
        print(f"  scale={scale:7.2f}  주지표={total:8.4f}  "
              f"NIS(mean)={row['nis']['mean']:9.2f}  "
              f"RMS e_y={row['rms_e_y']:.4e}  {cal_s}", flush=True)
    return rows


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in TUNE_SCENARIO:
        raise SystemExit(f"사용법: python scripts/tune_kf.py {'|'.join(TUNE_SCENARIO)}")
    which = sys.argv[1]

    print(f"\n{'='*76}\n[{which} 재튜닝]  시나리오={TUNE_SCENARIO[which]} "
          f"(in-distribution 만)\n"
          f"주 지표 = {'정규화 추정 RMSE 합' if which=='state_kf' else '정규화 외란 RMSE 합'}"
          f"  |  부 지표 = mean(NIS), 목표 n_meas=4\n{'='*76}", flush=True)

    rows = run_grid(which)

    best_obj = min(rows, key=lambda r: r["objective"])
    best_nis = min(rows, key=lambda r: abs(r["nis"]["mean"] - 4.0))
    print(f"\n[주 지표 최적]  scale={best_obj['scale']}  "
          f"objective={best_obj['objective']:.4f}  NIS={best_obj['nis']['mean']:.2f}")
    print(f"[NIS 최적]      scale={best_nis['scale']}  "
          f"objective={best_nis['objective']:.4f}  NIS={best_nis['nis']['mean']:.2f}")
    if best_obj["scale"] != best_nis["scale"]:
        print("\n** 주 지표와 NIS 가 다른 지점을 가리킨다 — 임의로 고르지 말고 "
              "사용자에게 두 결과를 함께 보고할 것. **")

    out = ROOT / "results" / "tuning" / (
        f"tune_{which}_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"filter": which, "scenario": TUNE_SCENARIO[which], "scales": list(SCALES),
         "rows": rows, "best_by_objective": best_obj["scale"],
         "best_by_nis": best_nis["scale"]},
        ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\n[저장] {out}")


if __name__ == "__main__":
    main()
