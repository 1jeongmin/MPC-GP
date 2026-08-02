"""experiment config -> 실행 가능한 케이스 조립 (Phase 8).

`scripts/run_sim.py` 가 Phase 4 유물이라 명목 MPC + 선형 플랜트를 하드코딩하고 있었다.
Part 1 의 세 케이스를 같은 실행기로 돌리려면 조립을 config 에서 끌어내야 한다.

## 케이스 분기를 쓰지 않는 방법 (CLAUDE.md 「if/else 로 케이스 분기 금지」)

`if case == "gp"` 같은 문자열 분기 대신, **어떤 config 그룹이 참조되었는지**로
빌더를 고른다. 조합 파일이 곧 케이스 정의다 (sim-experiment.md 「케이스 정의는
config 로만」). 새 케이스를 붙일 때는 `_CASE_BUILDERS` 에 한 줄을 추가하고 빌더
함수를 쓰면 되며, 기존 빌더·runner·MpcBase 는 건드리지 않는다.

플랜트도 마찬가지로 `plant` 키에서 조립한다 (스크립트 주입 금지).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from src.config import ExperimentConfig
from src.control.mpc_base import MpcBase, build_step_function
from src.control.mpc_gp import make_gp_mpc
from src.control.mpc_kf import make_kf_mpc
from src.control.mpc_nominal import NominalStepModel, make_nominal_mpc
from src.estimation.ekf import make_augmented_kf
from src.models.integrators import make_rhs_np
from src.models.nonlinear_bicycle import make_nonlinear_rhs_np
from src.path.reference import Reference

ROOT = Path(__file__).resolve().parents[2]
GP_DISK_CACHE_DIR = ROOT / "data" / "gp_cache"

# GP 학습 특징의 **코드상** 규약 식별자. 디스크 캐시 해시에 들어간다.
# 특징 정의를 바꿀 때마다 올려라 — 안 올리면 config 가 같다는 이유로 옛 캐시가
# 조용히 재사용된다 (`_gp_disk_cache_path` 참조).
#   v1: z=[v_y, gamma, delta_k]      (참값 x 기준 -> 이후 x_fb 로 교체)
#   v2: z=[v_y, gamma, delta_{k-1}]  (2026-08-01, 배포와 정합 + n_lags 지원)
#   v3: 학습 절차 변경 — Type-II ML 다중 재시작 (2026-08-01, 국소최적 붕괴 수정)
#   v4: 이분산 잡음 모델의 1차 모멘트 정합 버그 수정 (2026-08-03)
#       ★ 교훈: **코드만 바꾸고 이 상수를 안 올려서** 버그 있는 캐시본이 그대로
#       재사용됐고, "고쳤는데 수치가 소수점까지 똑같다"는 상황이 실제로 벌어졌다.
#       학습·특징 관련 코드를 고칠 때마다 여기를 함께 올려라.
# (production 캐시는 GpConfig 전체를 해시하므로 재시작 후보 필드가 생긴 것만으로도
#  자동 무효화된다. 이 상수는 config 를 안 거치는 스크립트 캐시용이다.)
FEATURE_SPEC = "z_xfb_delta_prev_lagged_v4_hetnoise_momentfix"

# plant 키 -> 연속 우변 팩토리. 새 플랜트는 여기에만 추가한다.
_PLANT_FACTORIES: dict[str, Callable] = {
    "linear": make_rhs_np,
    "nonlinear": make_nonlinear_rhs_np,
}


@dataclass
class Case:
    """조립된 실행 단위. runner 에 그대로 넘긴다."""
    name: str
    controller: MpcBase
    reference: Reference
    plant_rhs: Callable
    nominal_step: Callable
    estimator: Any | None = None
    # 케이스가 만들어낸 부산물 (예: 학습된 GP). 플롯·리포트에서 쓴다.
    artifacts: dict[str, Any] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.artifacts is None:
            self.artifacts = {}


def _build_nominal(exp: ExperimentConfig, deps: dict) -> Case:
    """보정 없음 (Part 1 케이스 1)."""
    ctrl = make_nominal_mpc(exp.vehicle, exp.mpc, exp.sim.dt_ctrl)
    return Case(name="mpc_only", controller=ctrl, **deps)


def _build_kf(exp: ExperimentConfig, deps: dict) -> Case:
    """증강 KF 외란 추정 주입 (Part 1 케이스 2)."""
    kf = make_augmented_kf(exp.vehicle, exp.ekf, exp.sim.dt_ctrl)
    ctrl = make_kf_mpc(exp.vehicle, exp.mpc, exp.sim.dt_ctrl, kf)
    return Case(name="mpc_kf", controller=ctrl, estimator=kf, **deps)


def _build_gp(exp: ExperimentConfig, deps: dict) -> Case:
    """offline GP 잔차 보정 주입 (Part 1 케이스 3).

    GP 는 `gp.train_experiment` 가 가리키는 **학습 전용 experiment** 에서 학습한다.
    평가 시나리오(this exp)와 학습 출처가 config 로 분리되어 있다는 것이 요점이다
    (gp-residual.md 「학습 궤적과 평가 궤적을 분리한다」).
    """
    gp = train_gp_from_config(exp)
    ctrl = make_gp_mpc(exp.vehicle, exp.mpc, exp.sim.dt_ctrl, gp)
    return Case(name="mpc_gp", controller=ctrl, artifacts={"gp": gp}, **deps)


# 참조된 그룹 -> 빌더. 위에서부터 첫 매칭을 쓰고, 아무것도 안 맞으면 명목.
# 새 케이스는 여기에 한 줄만 추가한다.
_CASE_BUILDERS: tuple[tuple[str, Callable[[ExperimentConfig, dict], Case]], ...] = (
    ("gp", _build_gp),
    ("ekf", _build_kf),
)


# 학습된 GP 캐시. 키 = (학습 experiment 이름, GpConfig). 같은 학습 출처·같은 설정이면
# 여러 평가 시나리오가 **같은 GP** 를 써야 한다 — 재학습하면 시나리오마다 GP 가 미묘하게
# 달라져 비교가 오염된다. 프로세스 메모리에만 있어 같은 실행 안에서만 유효 — 프로세스가
# 바뀌면 아래 디스크 캐시(GP_DISK_CACHE_DIR)로 넘어간다.
_GP_CACHE: dict[tuple, Any] = {}


def _gp_disk_cache_path(exp: ExperimentConfig, tr: ExperimentConfig) -> Path:
    """학습 출처 스냅샷 + GpConfig 전체를 해시해 캐시 파일 경로를 만든다.

    학습 config(경로·차량·MPC·플랜트)나 GpConfig 하이퍼파라미터가 하나라도 바뀌면
    다른 해시가 나와 자동으로 재학습된다 — 오래된 캐시를 몰래 재사용하는 사고를
    구조적으로 막는다(수동으로 캐시를 무효화할 필요가 없다).

    **`FEATURE_SPEC` 도 해시에 넣는다**: config 가 그대로여도 **코드**가 특징 정의를
    바꾸면(예: 2026-08-01 의 delta_k -> delta_{k-1} 정합) 옛 캐시는 무효다. config 만
    해시하면 그 변경이 조용히 무시되므로, 특징 규약을 바꿀 때 이 문자열을 함께 올려라.
    """
    import hashlib
    import json
    from dataclasses import asdict

    payload = {
        "train_experiment": exp.gp.train_experiment,
        "train_snapshot": tr.to_snapshot(),
        "gp_config": asdict(exp.gp),
        "feature_spec": FEATURE_SPEC,
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()
    ).hexdigest()[:16]
    return GP_DISK_CACHE_DIR / f"{exp.gp.train_experiment}_{digest}.npz"


def train_gp_from_config(exp: ExperimentConfig):
    """`exp.gp.train_experiment` 의 궤적으로 GP 를 학습해 반환한다.

    학습 데이터는 명목 MPC 로 학습 experiment 를 주행해 모은 잔차다. 학습 experiment
    자체도 조합 파일이므로 플랜트·경로·a_y 가 재현성 스냅샷에 그대로 남는다.
    같은 (학습 experiment, GpConfig) 조합은 먼저 프로세스 메모리 캐시에서, 없으면
    디스크 캐시(`data/gp_cache/`, git 제외)에서 재사용한다 — loop10 10바퀴 학습처럼
    시간이 오래 걸리는 학습을 프로세스를 새로 띄울 때마다 반복하지 않기 위해서다
    (2026-07-30). 학습 config 가 바뀌면 해시가 달라져 자동으로 재학습된다.

    학습 experiment 가 `sensor`/`state_kf` 를 참조하면 **배포와 같은 조건**(측정잡음
    + 공통 상태추정기)으로 데이터를 모은다 — `run_case` 와 **같은 팩토리**를 쓴다.
    참값으로 학습해 놓고 필터값으로 배포하면 특징 분포가 어긋난다 (2026-07-31).
    """
    from src.config import load_experiment
    from src.gp.dataset import save_dataset
    from src.gp.train_offline import load_gp, save_gp, train

    key = (exp.gp.train_experiment, exp.gp)
    if key in _GP_CACHE:
        return _GP_CACHE[key]

    tr = load_experiment(exp.gp.train_experiment)
    if tr.gp is not None:
        raise ValueError(
            f"GP 학습 experiment '{exp.gp.train_experiment}' 가 gp 그룹을 참조한다 (순환). "
            "학습 experiment 는 명목 MPC 케이스여야 한다."
        )

    cache_path = _gp_disk_cache_path(exp, tr)
    ds_path = cache_path.with_name(cache_path.stem + "_dataset.npz")
    if cache_path.exists() and ds_path.exists():
        gp = load_gp(cache_path)
    else:
        ds = collect_training_dataset(tr)
        gp = train(ds, exp.gp)
        save_gp(cache_path, gp)
        # 데이터셋도 함께 남긴다 — 캐시 적중 시에도 학습 산출물을 실험 폴더
        # (`results/part1_*/training_results/`)에 기록할 수 있어야 한다.
        save_dataset(ds_path, ds, tr.to_snapshot(), tr.sim.seed)

    _GP_CACHE[key] = gp
    return gp


def collect_training_dataset(tr: ExperimentConfig):
    """학습 experiment 를 주행해 잔차 데이터셋을 모은다 (production 경로 단일 출처).

    센서·상태추정기는 `run_case` 와 **같은 팩토리**로 만든다. 이 함수를 우회해
    데이터를 따로 모으면 배포에 쓰이는 GP 와 다른 GP 가 만들어진다.
    """
    from src.estimation.state_estimator import make_state_kf
    from src.gp.dataset import collect_residual_data
    from src.sim.sensor import make_sensor

    rng = np.random.default_rng(tr.sim.seed)   # seed 는 학습 experiment 것 (재현성)
    sensor = make_sensor(tr.sensor, rng)
    state_est = (None if tr.state_kf is None
                 else make_state_kf(tr.vehicle, tr.state_kf, tr.sim.dt_ctrl))
    return collect_residual_data(tr.vehicle, tr.mpc, tr.sim, tr.path, plant=tr.plant,
                                 sensor=sensor, state_estimator=state_est, rng=rng)


def gp_training_artifacts(exp: ExperimentConfig) -> tuple:
    """(학습 experiment config, 학습된 GP, 데이터셋) — 리포트 기록용.

    `train_gp_from_config` 가 캐시에 남긴 데이터셋을 재사용하므로 재주행하지 않는다.
    """
    from src.config import load_experiment
    from src.gp.dataset import load_dataset

    tr = load_experiment(exp.gp.train_experiment)
    gp = train_gp_from_config(exp)                     # 캐시 적중 (이미 학습됨)
    ds_path = _gp_disk_cache_path(exp, tr).with_name(
        _gp_disk_cache_path(exp, tr).stem + "_dataset.npz")
    ds = load_dataset(ds_path) if ds_path.exists() else collect_training_dataset(tr)
    return tr, gp, ds


def build_case(exp: ExperimentConfig) -> Case:
    """experiment config 로부터 실행 케이스를 조립한다.

    필요한 그룹(path, mpc)이 없으면 조립을 거부한다 — 조용히 기본값으로 때우면
    통제변수가 케이스마다 달라진다.
    """
    if exp.path is None or exp.mpc is None:
        raise ValueError(f"experiment '{exp.name}' 에 path/mpc 참조가 필요하다.")
    if exp.plant not in _PLANT_FACTORIES:
        raise ValueError(f"알 수 없는 plant: {exp.plant!r}. 지원: {sorted(_PLANT_FACTORIES)}")

    deps = {
        "reference": Reference(exp.path, exp.vehicle.vx_range),
        "plant_rhs": _PLANT_FACTORIES[exp.plant](exp.vehicle),
        # 잔차용 명목 이산 전이 — 컨트롤러 종류와 무관하게 **항상 명목**이다
        # (gp-residual.md: 잔차는 언제나 명목 대비).
        "nominal_step": build_step_function(NominalStepModel(exp.vehicle, exp.sim.dt_ctrl)),
    }

    for group, builder in _CASE_BUILDERS:
        if getattr(exp, group) is not None:
            return builder(exp, deps)
    return _build_nominal(exp, deps)


def run_case(case: Case, exp: ExperimentConfig,
             rng: np.random.Generator | None = None) -> dict[str, np.ndarray]:
    """조립된 케이스로 폐루프를 돌린다. seed 는 config 에서 온다 (재현성).

    센서는 experiment 의 `sensor` 그룹 참조 여부로 갈린다 — 참조하지 않으면 None 이
    되어 종전대로 참 상태 피드백(이상적 센서)이다. 여기서도 케이스 분기는 없다.
    """
    from src.estimation.state_estimator import make_state_kf
    from src.sim.runner import run_closed_loop
    from src.sim.sensor import make_sensor

    if rng is None:
        rng = np.random.default_rng(exp.sim.seed)
    sensor = make_sensor(exp.sensor, rng)
    # 공통 상태추정기: state_kf 그룹 참조 여부로만 갈린다 — 케이스와 무관하게
    # 세 케이스가 동일하게 받는다(통제변수). 모델 보정자(case.estimator)와 별개다.
    state_est = (None if exp.state_kf is None
                 else make_state_kf(exp.vehicle, exp.state_kf, exp.sim.dt_ctrl))
    return run_closed_loop(case.controller, case.reference, case.plant_rhs,
                           case.nominal_step, exp.sim, estimator=case.estimator,
                           rng=rng, sensor=sensor, state_estimator=state_est)
