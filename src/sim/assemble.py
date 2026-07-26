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
# 달라져 비교가 오염된다. 부수적으로 레이싱 트랙(학습 주행 1회 = 56 s)에서 시간도 아낀다.
_GP_CACHE: dict[tuple, Any] = {}


def train_gp_from_config(exp: ExperimentConfig):
    """`exp.gp.train_experiment` 의 궤적으로 GP 를 학습해 반환한다.

    학습 데이터는 명목 MPC 로 학습 experiment 를 주행해 모은 잔차다. 학습 experiment
    자체도 조합 파일이므로 플랜트·경로·a_y 가 재현성 스냅샷에 그대로 남는다.
    같은 (학습 experiment, GpConfig) 조합은 캐시에서 재사용한다.
    """
    from src.config import load_experiment
    from src.gp.dataset import collect_residual_data
    from src.gp.train_offline import train

    key = (exp.gp.train_experiment, exp.gp)
    if key in _GP_CACHE:
        return _GP_CACHE[key]

    tr = load_experiment(exp.gp.train_experiment)
    if tr.gp is not None:
        raise ValueError(
            f"GP 학습 experiment '{exp.gp.train_experiment}' 가 gp 그룹을 참조한다 (순환). "
            "학습 experiment 는 명목 MPC 케이스여야 한다."
        )
    ds = collect_residual_data(tr.vehicle, tr.mpc, tr.sim, tr.path, plant=tr.plant)
    gp = train(ds, exp.gp)
    _GP_CACHE[key] = gp
    return gp


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
    """조립된 케이스로 폐루프를 돌린다. seed 는 config 에서 온다 (재현성)."""
    from src.sim.runner import run_closed_loop

    if rng is None:
        rng = np.random.default_rng(exp.sim.seed)
    return run_closed_loop(case.controller, case.reference, case.plant_rhs,
                           case.nominal_step, exp.sim, estimator=case.estimator, rng=rng)
