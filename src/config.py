"""config 로딩 체계.

그룹 YAML(`configs/<group>/<name>.yaml`)을 dataclass로 로드하고, 로드 시점에
유효성 검사를 수행한다. `configs/experiment/*.yaml`는 그룹 참조만 담는 조합 파일로,
값을 직접 갖지 않는다 (.claude/rules/sim-experiment.md "케이스 정의는 config로만").

단위 규약: 전부 SI, 각도는 rad (.claude/rules/python-conventions.md).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any

import yaml

# 저장소 루트 기준 configs 디렉토리. 이 파일은 src/config.py 이므로 parents[1] = 루트.
CONFIGS_DIR = Path(__file__).resolve().parents[1] / "configs"


def _load_yaml(path: Path) -> dict[str, Any]:
    """YAML 파일을 dict로 읽는다. 파일이 없으면 명확한 예외를 던진다."""
    if not path.exists():
        raise FileNotFoundError(f"config 파일이 없다: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"config 최상위가 매핑이 아니다: {path}")
    return data


@dataclass(frozen=True)
class VehicleConfig:
    """선형 동역학 자전거 모델의 물리 파라미터.

    단위: m [kg], Iz [kg*m^2], a/b [m], Cf/Cr [N/rad] (축당 코너링 강성, 양수),
          delta_max [rad], delta_rate_max [rad/s], vx_range [m/s] (하한, 상한).
    부호·정의 규약은 .claude/rules/vehicle-model.md 가 유일한 기준이다.
    """
    m: float
    Iz: float
    a: float
    b: float
    Cf: float
    Cr: float
    delta_max: float
    delta_rate_max: float
    vx_range: tuple[float, float]

    @property
    def L(self) -> float:
        """축거 a + b [m]."""
        return self.a + self.b

    def __post_init__(self) -> None:
        # 물리 파라미터는 전부 양수여야 한다.
        for name in ("m", "Iz", "a", "b", "Cf", "Cr", "delta_max", "delta_rate_max"):
            val = getattr(self, name)
            if not (isinstance(val, (int, float)) and val > 0.0):
                raise ValueError(f"VehicleConfig.{name} 는 양수여야 한다 (got {val!r}).")
        if self.L <= 0.0:
            raise ValueError(f"a + b 는 양수여야 한다 (got {self.L}).")
        # vx_range 하한 > 0: vx->0 에서 A 행렬이 발산한다
        # (.claude/rules/vehicle-model.md "특이점").
        lo, hi = self.vx_range
        if not (lo > 0.0):
            raise ValueError(f"vx_range 하한은 0보다 커야 한다 (got {lo}).")
        if not (hi > lo):
            raise ValueError(f"vx_range 는 [하한 < 상한] 이어야 한다 (got [{lo}, {hi}]).")

    def check_vx(self, vx: float) -> None:
        """런타임 vx 유효성 검사. 하한 미만이면 예외를 던진다 (특이점 방어)."""
        lo, hi = self.vx_range
        if vx < lo:
            raise ValueError(
                f"vx={vx} 가 vx_range 하한 {lo} 미만이다. "
                f"vx->0 에서 모델이 발산한다 (vehicle-model.md 특이점)."
            )

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "VehicleConfig":
        return cls(
            m=float(d["m"]),
            Iz=float(d["Iz"]),
            a=float(d["a"]),
            b=float(d["b"]),
            Cf=float(d["Cf"]),
            Cr=float(d["Cr"]),
            delta_max=float(d["delta_max"]),
            delta_rate_max=float(d["delta_rate_max"]),
            vx_range=(float(d["vx_range"][0]), float(d["vx_range"][1])),
        )


@dataclass(frozen=True)
class SimConfig:
    """다중레이트 시뮬레이션 설정.

    단위: dt_plant/dt_ctrl/duration [s], seed [정수].
    플랜트 100 Hz / 제어 50 Hz. dt_ctrl 는 dt_plant 의 정수배여야 한다.
    """
    dt_plant: float
    dt_ctrl: float
    duration: float
    seed: int

    def __post_init__(self) -> None:
        for name in ("dt_plant", "dt_ctrl", "duration"):
            val = getattr(self, name)
            if not (isinstance(val, (int, float)) and val > 0.0):
                raise ValueError(f"SimConfig.{name} 는 양수여야 한다 (got {val!r}).")
        # dt_ctrl 가 dt_plant 의 정수배가 아니면 다중레이트 루프가 어긋난다.
        ratio = self.dt_ctrl / self.dt_plant
        if abs(ratio - round(ratio)) > 1e-9:
            raise ValueError(
                f"dt_ctrl({self.dt_ctrl}) 가 dt_plant({self.dt_plant}) 의 정수배가 아니다 "
                f"(ratio={ratio})."
            )
        if self.dt_ctrl < self.dt_plant:
            raise ValueError("dt_ctrl 는 dt_plant 이상이어야 한다 (제어가 플랜트보다 느리다).")

    @property
    def substeps(self) -> int:
        """제어 1스텝당 플랜트 스텝 수."""
        return round(self.dt_ctrl / self.dt_plant)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SimConfig":
        return cls(
            dt_plant=float(d["dt_plant"]),
            dt_ctrl=float(d["dt_ctrl"]),
            duration=float(d["duration"]),
            seed=int(d["seed"]),
        )


def _dataclass_to_plain(obj: Any) -> dict[str, Any]:
    """dataclass -> 순수 dict (tuple 등을 YAML/JSON 친화 형태로)."""
    d = asdict(obj)
    return {k: (list(v) if isinstance(v, tuple) else v) for k, v in d.items()}


@dataclass(frozen=True)
class ExperimentConfig:
    """조합된 experiment 설정. 그룹별 하위 config를 담는다.

    Phase 진행에 따라 path/mpc/gp/ekf 가 추가된다. 지금은 vehicle, sim 만 필수다.
    """
    name: str
    vehicle: VehicleConfig
    sim: SimConfig
    # 조합 파일이 참조한 그룹 이름 -> config 이름 (재현성 스냅샷에 남긴다).
    raw_refs: dict[str, str] = field(default_factory=dict)

    def to_snapshot(self) -> dict[str, Any]:
        """재현성 스냅샷용 dict 덤프. 참조가 아니라 조립된 값 전체를 담는다
        (.claude/rules/sim-experiment.md "재현성")."""
        return {
            "name": self.name,
            "refs": dict(self.raw_refs),
            "vehicle": _dataclass_to_plain(self.vehicle),
            "sim": _dataclass_to_plain(self.sim),
        }


# 그룹 이름 -> 해당 dataclass 로더 매핑.
# 새 그룹을 붙일 때 여기에만 추가한다 (if 분기 확산 방지).
_GROUP_LOADERS = {
    "vehicle": VehicleConfig.from_dict,
    "sim": SimConfig.from_dict,
}


def load_group(group: str, name: str, configs_dir: Path = CONFIGS_DIR) -> Any:
    """단일 그룹 config를 로드한다. 예: load_group("vehicle", "sedan")."""
    if group not in _GROUP_LOADERS:
        raise KeyError(f"알 수 없는 config 그룹: {group!r}. 지원: {sorted(_GROUP_LOADERS)}")
    path = configs_dir / group / f"{name}.yaml"
    return _GROUP_LOADERS[group](_load_yaml(path))


def load_experiment(name: str, configs_dir: Path = CONFIGS_DIR) -> ExperimentConfig:
    """configs/experiment/<name>.yaml 조합 파일을 로드한다.

    조합 파일은 그룹 참조(`group: name`)만 담는다. 값을 직접 담고 있으면 잘못된 것이다.
    현재 vehicle, sim 은 필수. 나머지 그룹은 Phase 진행에 따라 추가된다.
    """
    exp_path = configs_dir / "experiment" / f"{name}.yaml"
    refs_raw = _load_yaml(exp_path)

    # 조합 파일에는 문자열 참조만 허용한다 (값 복붙 방지).
    refs: dict[str, str] = {}
    for group, ref in refs_raw.items():
        if not isinstance(ref, str):
            raise ValueError(
                f"experiment '{name}' 의 '{group}' 는 그룹 참조(문자열)여야 한다. "
                f"값을 직접 적지 마라 (got {ref!r})."
            )
        refs[group] = ref

    for required in ("vehicle", "sim"):
        if required not in refs:
            raise ValueError(f"experiment '{name}' 에 필수 그룹 '{required}' 참조가 없다.")

    return ExperimentConfig(
        name=name,
        vehicle=load_group("vehicle", refs["vehicle"], configs_dir),
        sim=load_group("sim", refs["sim"], configs_dir),
        raw_refs=refs,
    )
