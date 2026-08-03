"""loop10 10바퀴 GP 학습 + 결과 저장·시각화.

`configs/gp/loop10.yaml`이 가리키는 유일한 학습 출처(`gp_train_loop10`, 10바퀴,
`.claude/rules/gp-residual.md` 데이터 위생)로 GP를 학습하고, 딕셔너리·잔차 데이터셋과
학습된 GP를 `results/training_results/<run_id>/`에 npz로 저장한다. 트랙 형상과
학습 결과(상태공간 커버리지·사후 std 지도·in-sample 적합도·하이퍼파라미터)를
그림으로 남긴다.

잔차 계산은 `run_closed_loop`(runner)가 로깅한 것을 그대로 쓴다 — 재구현하지
않는다. 앞부분(reference/controller/plant 조립)은 `src/gp/dataset.py`의
`collect_residual_data`와 동일한 절차를 따르되, 그 함수는 로그의 s/t를 버리므로
여기서는 트랙 진행 그림을 위해 직접 `run_closed_loop`을 호출해 s/t를 보존한다.

    python scripts/train_gp_loop10.py

10바퀴(~2.28 km x 10, 제어스텝 약 5.3만 개) MPC 폐루프라 실행에 수 분(측정
~8.5분)이 걸린다.
"""
from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from src.config import load_experiment, load_group
from src.control.mpc_base import build_step_function
from src.control.mpc_nominal import NominalStepModel, make_nominal_mpc
from src.gp.dataset import ResidualDataset, save_dataset
from src.gp.train_offline import save_gp, train
from src.models.integrators import make_rhs_np
from src.estimation.state_estimator import make_state_kf
from src.models.nonlinear_bicycle import make_nonlinear_rhs_np
from src.path.reference import Reference
from src.sim.assemble import GP_TRAIN_CODE_ID
from src.sim.logger import write_run_meta
from src.sim.runner import run_closed_loop
from src.sim.sensor import make_sensor
from src.viz.plots import make_gp_training_figure, make_track_shape_figure


def main() -> None:
    t_wall0 = time.time()

    tr = load_experiment("gp_train_loop10")
    if tr.gp is not None:
        raise RuntimeError("학습 experiment가 gp 그룹을 참조한다 (순환 — gp-residual.md 위반).")
    gp_cfg = load_group("gp", "loop10")

    reference = Reference(tr.path, tr.vehicle.vx_range)
    controller = make_nominal_mpc(tr.vehicle, tr.mpc, tr.sim.dt_ctrl)
    nominal_step = build_step_function(NominalStepModel(tr.vehicle, tr.sim.dt_ctrl))
    plant_rhs = (make_nonlinear_rhs_np(tr.vehicle) if tr.plant == "nonlinear"
                else make_rhs_np(tr.vehicle))

    # 센서·공통 상태추정기 — production 경로(`assemble.collect_training_dataset`)와
    # **같은 팩토리·같은 seed**. 이걸 빼면 참값으로 학습돼 배포(필터값)와 어긋나고,
    # 여기 기록되는 학습결과가 실제 쓰인 GP 와 달라진다 (2026-07-31).
    rng = np.random.default_rng(tr.sim.seed)
    sensor = make_sensor(tr.sensor, rng)
    state_est = (None if tr.state_kf is None
                 else make_state_kf(tr.vehicle, tr.state_kf, tr.sim.dt_ctrl))

    target_s = reference.total_length * tr.sim.n_laps
    print(f"[설정] total_length={reference.total_length:.3f} m  n_laps={tr.sim.n_laps}  "
          f"목표 호길이={target_s:.3f} m  plant={tr.plant}")
    print("[실행] run_closed_loop 시작 (10바퀴, 수 분 소요)...")

    log = run_closed_loop(controller, reference, plant_rhs, nominal_step, tr.sim,
                          rng=rng, sensor=sensor, state_estimator=state_est)

    n_steps = len(log["t"])
    max_s = float(log["s"][-1])
    print(f"[검증] 제어스텝={n_steps}  최종 t={log['t'][-1]:.2f} s  "
          f"최종 s={max_s:.3f} m (목표 {target_s:.3f} m, 차이 {max_s - target_s:+.3f} m)")

    # 특징은 **x_fb**(제어기가 실제로 받는 상태) + **delta_{k-1}** 에서 뽑는다 — 배포에서
    # GP 가 보는 것과 같아야 한다 (src/gp/dataset.py 의 collect_residual_data 와 동일 규약).
    # ★ 2026-08-03 수정: 여기서 `log["delta"]`(=delta_k, 적용된 입력)를 쓰고 있었다.
    # 배포는 solve **전**에 GP 를 평가하므로 delta_k 를 알 수 없어 u_prev 를 쓴다 —
    # 2026-08-01 에 production 경로(dataset.py)만 고치고 이 기록용 스크립트를 빠뜨려서,
    # 여기 남는 학습 기록이 **실제 배포된 GP 와 다른 GP** 였다.
    Z = np.column_stack([log["x_fb"][:, 0], log["x_fb"][:, 1], log["delta_prev"]])
    R = log["residual"][:, 0:2]                                            # r_vy, r_gamma
    ds = ResidualDataset(Z, R, meta={"plant": tr.plant, "n": Z.shape[0], "seed": tr.sim.seed,
                                     "filtered_features": state_est is not None,
                                     "noisy_sensor": sensor is not None},
                         s=log["s"].copy(), t=log["t"].copy())
    rms = np.sqrt(np.mean(R**2, axis=0))
    print(f"[잔차] RMS r_vy={rms[0]:.6e} m/s   RMS r_gamma={rms[1]:.6e} rad/s")

    print("[학습] Type-II ML 진행 중...")
    gp = train(ds, gp_cfg)
    for name, ch in zip(("v_y", "gamma"), gp.channels):
        print(f"  {name:6s} lengthscales={np.array2string(ch.lengthscales, precision=4)}  "
              f"sigma_f={ch.sigma_f:.4f}  sigma_n={ch.sigma_n:.4f}  "
              f"(floor={gp_cfg.sigma_n_floor})")

    run_id = "gp_train_loop10_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "results" / "training_results" / run_id

    save_dataset(out_dir / "dataset.npz", ds, tr.to_snapshot(), tr.sim.seed)
    save_gp(out_dir / "gp_model.npz", gp)
    write_run_meta(
        out_dir, run_id, tr.to_snapshot(), tr.sim.seed,
        extra={
            # 이 기록이 어떤 **학습 코드**에서 나왔는지. 평가 런의 meta.json 에 찍히는
            # 같은 값과 대조하면 "이 기록이 그 GP 와 같은 코드인가"를 확인할 수 있다
            # (2026-08-03, assemble.GP_TRAIN_CODE_ID). git hash 는 dirty 면 무력하다.
            "gp_train_code_id": GP_TRAIN_CODE_ID,
            "gp_config": {"M": gp_cfg.M, "sigma_n_floor": gp_cfg.sigma_n_floor,
                          "init_lengthscale": gp_cfg.init_lengthscale,
                          "init_sigma_f": gp_cfg.init_sigma_f,
                          "init_sigma_n": gp_cfg.init_sigma_n,
                          "jitter": gp_cfg.jitter,
                          "train_experiment": gp_cfg.train_experiment},
            "n_control_steps": n_steps, "final_s_m": max_s, "target_s_m": target_s,
            "final_t_s": float(log["t"][-1]),
            "residual_rms": {"v_y": float(rms[0]), "gamma": float(rms[1])},
            "hyperparameters": {
                name: {"lengthscales": ch.lengthscales.tolist(),
                      "sigma_f": ch.sigma_f, "sigma_n": ch.sigma_n}
                for name, ch in zip(("v_y", "gamma"), gp.channels)
            },
            "wall_clock_s": time.time() - t_wall0,
        },
    )

    make_track_shape_figure(reference, out_dir, run_id)
    make_gp_training_figure(gp, ds, gp_cfg, out_dir, run_id, s=log["s"], t=log["t"])

    print(f"\n[저장] {out_dir}")
    print(f"[총 소요] {time.time() - t_wall0:.1f} s")


if __name__ == "__main__":
    main()
