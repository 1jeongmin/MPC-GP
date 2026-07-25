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
    # 비선형 플랜트(Phase 5)용. 선형 명목 모델에는 불필요하므로 선택적.
    mu: float | None = None      # 노면 마찰계수 [-]
    g: float = 9.81              # 중력가속도 [m/s^2]

    @property
    def L(self) -> float:
        """축거 a + b [m]."""
        return self.a + self.b

    @property
    def Fzf(self) -> float:
        """전축 정적 수직하중 [N] = m*g*b/L (무게 분배)."""
        return self.m * self.g * self.b / self.L

    @property
    def Fzr(self) -> float:
        """후축 정적 수직하중 [N] = m*g*a/L."""
        return self.m * self.g * self.a / self.L

    def require_mu(self) -> float:
        """mu 가 설정돼 있으면 반환, 아니면 예외 (비선형 플랜트 사용 시)."""
        if self.mu is None or not (self.mu > 0.0):
            raise ValueError("비선형 플랜트에는 config 에 양수 mu 가 필요하다.")
        return self.mu

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
            mu=None if d.get("mu") is None else float(d["mu"]),
            g=float(d.get("g", 9.81)),
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


@dataclass(frozen=True)
class PathSegment:
    """경로 세그먼트. 끝점 곡률만 갖는다 (시작 곡률 = 직전 세그먼트 끝점).

    length [m], kappa_end [1/m] (좌회전 +, 우회전 -). label 은 가독성용.
    """
    length: float
    kappa_end: float
    label: str = ""


@dataclass(frozen=True)
class PathConfig:
    """호길이 곡률 프로파일 kappa(s) 정의.

    단위: a_y_max [m/s^2], kappa_eps [1/m], segments (length [m], kappa_end [1/m]).
    첫 세그먼트의 시작 곡률은 0으로 가정한다 (직선에서 출발).
    kappa_max 는 참고용 메타데이터로 빌더는 사용하지 않는다.
    """
    a_y_max: float
    kappa_eps: float
    segments: tuple[PathSegment, ...]
    kappa_max: float | None = None

    def __post_init__(self) -> None:
        if not (self.a_y_max > 0.0):
            raise ValueError(f"a_y_max 는 양수여야 한다 (got {self.a_y_max}).")
        if not (self.kappa_eps > 0.0):
            raise ValueError(f"kappa_eps 는 양수여야 한다 (got {self.kappa_eps}).")
        if len(self.segments) == 0:
            raise ValueError("segments 가 비어 있다.")
        for i, seg in enumerate(self.segments):
            if not (seg.length > 0.0):
                raise ValueError(f"segments[{i}].length 는 양수여야 한다 (got {seg.length}).")

    @property
    def total_length(self) -> float:
        """경로 총 호길이 [m]."""
        return float(sum(seg.length for seg in self.segments))

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PathConfig":
        segs = tuple(
            PathSegment(
                length=float(s["length"]),
                kappa_end=float(s["kappa_end"]),
                label=str(s.get("label", "")),
            )
            for s in d["segments"]
        )
        kmax = d.get("kappa_max")
        return cls(
            a_y_max=float(d["a_y_max"]),
            kappa_eps=float(d["kappa_eps"]),
            segments=segs,
            kappa_max=None if kmax is None else float(kmax),
        )


@dataclass(frozen=True)
class MpcConfig:
    """명목 MPC 정식화·솔버 설정 (.claude/rules/mpc-solver.md).

    가중치 이름은 KF 잡음 공분산(Q_kf, R_kf)과 절대 겹치지 않게 짓는다.
    상태 가중치는 W_ 접두사, 입력은 R_ 접두사(값 자체는 명시적 이름).
    단위: 지평 N [스텝], dt_ctrl [s], 가중치 [무차원 상대], IPOPT 옵션.

    비용:
      sum_k [ W_ey*e_y^2 + W_epsi*e_psi^2 + W_vy*v_y^2 + W_gamma*gamma^2
              + R_delta*delta^2 + R_ddelta*(delta_k - delta_{k-1})^2 ]
      + Wf_ey*e_y_N^2 + Wf_epsi*e_psi_N^2   (종단항)

    dt_ctrl 은 여기 두지 않는다 — 타이밍은 SimConfig 가 단일 소스이며,
    컨트롤러 생성 시 주입한다 (두 곳에 두면 어긋난다).
    """
    N: int
    # 스테이지 상태 가중치
    W_ey: float
    W_epsi: float
    W_vy: float
    W_gamma: float
    # 입력 가중치
    R_delta: float
    R_ddelta: float
    # 종단 가중치
    Wf_ey: float
    Wf_epsi: float
    # IPOPT 옵션 (케이스 간 동일하게 유지 — 다르면 solve time 비교 무의미)
    ipopt_max_iter: int
    ipopt_tol: float
    ipopt_print_level: int

    def __post_init__(self) -> None:
        if not (isinstance(self.N, int) and self.N > 0):
            raise ValueError(f"N 은 양의 정수여야 한다 (got {self.N!r}).")
        for name in ("W_ey", "W_epsi", "W_vy", "W_gamma", "R_delta", "R_ddelta",
                     "Wf_ey", "Wf_epsi"):
            val = getattr(self, name)
            if not (val >= 0.0):
                raise ValueError(f"MpcConfig.{name} 는 음수가 아니어야 한다 (got {val}).")
        if not (self.ipopt_max_iter > 0):
            raise ValueError(f"ipopt_max_iter 는 양수여야 한다 (got {self.ipopt_max_iter}).")

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "MpcConfig":
        ip = d.get("ipopt", {})
        return cls(
            N=int(d["N"]),
            W_ey=float(d["W_ey"]),
            W_epsi=float(d["W_epsi"]),
            W_vy=float(d["W_vy"]),
            W_gamma=float(d["W_gamma"]),
            R_delta=float(d["R_delta"]),
            R_ddelta=float(d["R_ddelta"]),
            Wf_ey=float(d["Wf_ey"]),
            Wf_epsi=float(d["Wf_epsi"]),
            ipopt_max_iter=int(ip.get("max_iter", 200)),
            ipopt_tol=float(ip.get("tol", 1e-8)),
            ipopt_print_level=int(ip.get("print_level", 0)),
        )


@dataclass(frozen=True)
class EkfConfig:
    """증강 KF 설정 (.claude/rules/ekf-baseline.md).

    증강상태 순서 = [v_y, gamma, e_psi, e_y, d_vy, d_gamma] (6).
    측정 = [gamma, e_psi, e_y] (3, v_y 미측정).
    이름은 MPC 가중치(W_/R_)와 절대 겹치지 않게: Q_kf, R_kf (mpc-solver.md 표기 충돌).

    Q_kf_diag: 프로세스 잡음 연속 공분산 대각(6). 예측에서 *dt 이산화.
    R_kf_diag: 측정 잡음 공분산 대각(3).
    P0_diag: 초기 추정 공분산 대각(6).
    """
    Q_kf_diag: tuple[float, ...]
    R_kf_diag: tuple[float, ...]
    P0_diag: tuple[float, ...]
    tuning_method: str = "manual"   # 공정성: 튜닝 방식 명시 (ekf-baseline.md)

    def __post_init__(self) -> None:
        if len(self.Q_kf_diag) != 6:
            raise ValueError(f"Q_kf_diag 는 길이 6 (got {len(self.Q_kf_diag)}).")
        if len(self.R_kf_diag) != 3:
            raise ValueError(f"R_kf_diag 는 길이 3 (got {len(self.R_kf_diag)}).")
        if len(self.P0_diag) != 6:
            raise ValueError(f"P0_diag 는 길이 6 (got {len(self.P0_diag)}).")
        for name in ("Q_kf_diag", "R_kf_diag", "P0_diag"):
            if any(v < 0.0 for v in getattr(self, name)):
                raise ValueError(f"EkfConfig.{name} 성분은 음수가 아니어야 한다.")

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "EkfConfig":
        return cls(
            Q_kf_diag=tuple(float(v) for v in d["Q_kf_diag"]),
            R_kf_diag=tuple(float(v) for v in d["R_kf_diag"]),
            P0_diag=tuple(float(v) for v in d["P0_diag"]),
            tuning_method=str(d.get("tuning_method", "manual")),
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
    path: "PathConfig | None" = None
    mpc: "MpcConfig | None" = None
    ekf: "EkfConfig | None" = None
    # 조합 파일이 참조한 그룹 이름 -> config 이름 (재현성 스냅샷에 남긴다).
    raw_refs: dict[str, str] = field(default_factory=dict)

    def to_snapshot(self) -> dict[str, Any]:
        """재현성 스냅샷용 dict 덤프. 참조가 아니라 조립된 값 전체를 담는다
        (.claude/rules/sim-experiment.md "재현성")."""
        snap = {
            "name": self.name,
            "refs": dict(self.raw_refs),
            "vehicle": _dataclass_to_plain(self.vehicle),
            "sim": _dataclass_to_plain(self.sim),
        }
        if self.path is not None:
            snap["path"] = _dataclass_to_plain(self.path)
        if self.mpc is not None:
            snap["mpc"] = _dataclass_to_plain(self.mpc)
        if self.ekf is not None:
            snap["ekf"] = _dataclass_to_plain(self.ekf)
        return snap


# 그룹 이름 -> 해당 dataclass 로더 매핑.
# 새 그룹을 붙일 때 여기에만 추가한다 (if 분기 확산 방지).
_GROUP_LOADERS = {
    "vehicle": VehicleConfig.from_dict,
    "sim": SimConfig.from_dict,
    "path": PathConfig.from_dict,
    "mpc": MpcConfig.from_dict,
    "ekf": EkfConfig.from_dict,
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
        path=load_group("path", refs["path"], configs_dir) if "path" in refs else None,
        mpc=load_group("mpc", refs["mpc"], configs_dir) if "mpc" in refs else None,
        ekf=load_group("ekf", refs["ekf"], configs_dir) if "ekf" in refs else None,
        raw_refs=refs,
    )
