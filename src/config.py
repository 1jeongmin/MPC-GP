"""config 로딩 체계.

그룹 YAML(`configs/<group>/<name>.yaml`)을 dataclass로 로드하고, 로드 시점에
유효성 검사를 수행한다. `configs/experiment/*.yaml`는 그룹 참조만 담는 조합 파일로,
값을 직접 갖지 않는다 (.claude/rules/sim-experiment.md "케이스 정의는 config로만").

단위 규약: 전부 SI, 각도는 rad (.claude/rules/python-conventions.md).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field, fields, replace
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
class TableSource:
    """샘플된 kappa(s) 테이블의 출처 (측정·외부 제공 경로용).

    합성 경로는 `segments` 로 정의하지만, 실측/외부 트랙은 s 와 kappa 가 이미 촘촘히
    샘플되어 있다. 그것을 1900개짜리 segments 로 옮겨 적는 것은 의미가 없으므로
    테이블을 그대로 읽는다. `Reference` 는 두 경우 모두 같은 (s_nodes, k_nodes) 선형
    보간으로 귀결되므로 곡률 표현 자체는 달라지지 않는다.

    file: 저장소 루트 기준 상대경로 (data/ 아래. git 제외이므로 sha256 을 함께 기록).
    kind: 'mat' 만 지원. rows: 2D 배열에서 s / kappa 를 뽑을 행 인덱스.
    """
    file: str
    kind: str = "mat"
    var: str = "map"
    s_row: int = 0
    kappa_row: int = 4
    sha256: str | None = None       # 재현성: 데이터가 바뀌면 결과도 바뀐다

    def __post_init__(self) -> None:
        if self.kind != "mat":
            raise ValueError(f"지원하지 않는 테이블 종류: {self.kind!r} (현재 'mat' 만).")
        if not self.file:
            raise ValueError("TableSource.file 이 비어 있다.")


@dataclass(frozen=True)
class PathConfig:
    """호길이 곡률 프로파일 kappa(s) 정의.

    단위: a_y_max [m/s^2], kappa_eps [1/m], segments (length [m], kappa_end [1/m]).
    첫 세그먼트의 시작 곡률은 0으로 가정한다 (직선에서 출발).
    kappa_max 는 참고용 메타데이터로 빌더는 사용하지 않는다.

    경로 정의는 **둘 중 정확히 하나**다:
      - `segments`: 합성 경로 (조각선형 클로소이드). single_curve, dlc.
      - `table`:    샘플된 kappa(s) 테이블 (외부 제공 트랙). racetrack.
    둘 다 주거나 둘 다 빠지면 예외 — 어느 쪽이 경로를 정의했는지 모호해지면 안 된다.
    """
    a_y_max: float
    kappa_eps: float
    segments: tuple[PathSegment, ...] = ()
    kappa_max: float | None = None
    table: TableSource | None = None

    def __post_init__(self) -> None:
        if not (self.a_y_max > 0.0):
            raise ValueError(f"a_y_max 는 양수여야 한다 (got {self.a_y_max}).")
        if not (self.kappa_eps > 0.0):
            raise ValueError(f"kappa_eps 는 양수여야 한다 (got {self.kappa_eps}).")
        has_seg, has_tab = len(self.segments) > 0, self.table is not None
        if has_seg == has_tab:
            raise ValueError(
                "경로는 segments 또는 table 중 **정확히 하나**로 정의해야 한다 "
                f"(segments={len(self.segments)}, table={'있음' if has_tab else '없음'})."
            )
        for i, seg in enumerate(self.segments):
            if not (seg.length > 0.0):
                raise ValueError(f"segments[{i}].length 는 양수여야 한다 (got {seg.length}).")

    @property
    def total_length(self) -> float:
        """경로 총 호길이 [m]. 테이블 경로는 Reference 가 파일에서 읽으므로 여기선 0."""
        return float(sum(seg.length for seg in self.segments))

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PathConfig":
        segs = tuple(
            PathSegment(
                length=float(s["length"]),
                kappa_end=float(s["kappa_end"]),
                label=str(s.get("label", "")),
            )
            for s in d.get("segments", ()) or ()
        )
        tab_d = d.get("table")
        table = None if tab_d is None else TableSource(
            file=str(tab_d["file"]),
            kind=str(tab_d.get("kind", "mat")),
            var=str(tab_d.get("var", "map")),
            s_row=int(tab_d.get("s_row", 0)),
            kappa_row=int(tab_d.get("kappa_row", 4)),
            sha256=None if tab_d.get("sha256") is None else str(tab_d["sha256"]),
        )
        kmax = d.get("kappa_max")
        return cls(
            a_y_max=float(d["a_y_max"]),
            kappa_eps=float(d["kappa_eps"]),
            segments=segs,
            kappa_max=None if kmax is None else float(kmax),
            table=table,
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
    측정 = [v_y, gamma, e_psi, e_y] (4, 전상태 — Phase 6 진단으로 부분관측에서 수정).
    이름은 MPC 가중치(W_/R_)와 절대 겹치지 않게: Q_kf, R_kf (mpc-solver.md 표기 충돌).

    Q_kf_diag: 프로세스 잡음 연속 공분산 대각(6). 예측에서 *dt 이산화.
    R_kf_diag: 측정 잡음 공분산 대각(4).
    P0_diag: 초기 추정 공분산 대각(6).
    """
    Q_kf_diag: tuple[float, ...]
    R_kf_diag: tuple[float, ...]
    P0_diag: tuple[float, ...]
    tuning_method: str = "manual"   # 공정성: 튜닝 방식 명시 (ekf-baseline.md)
    # 공정성: GP 와 **같은 형식**으로 남기는 튜닝 예산. 스냅샷에 실려야 리포트에서
    # 비교 가능하다 (필드로 두지 않으면 from_dict 가 버려서 기록이 사라진다).
    tuning_budget_closed_loop_evals: int = 0
    tuning_selection_scenario: str = ""

    def __post_init__(self) -> None:
        if len(self.Q_kf_diag) != 6:
            raise ValueError(f"Q_kf_diag 는 길이 6 (got {len(self.Q_kf_diag)}).")
        if len(self.R_kf_diag) != 4:
            raise ValueError(f"R_kf_diag 는 길이 4 (got {len(self.R_kf_diag)}).")
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
            tuning_budget_closed_loop_evals=int(d.get("tuning_budget_closed_loop_evals", 0)),
            tuning_selection_scenario=str(d.get("tuning_selection_scenario", "")),
        )


@dataclass(frozen=True)
class StateKfConfig:
    """공통 상태추정기(명목 4상태 순수 KF) 설정 (Phase 8d).

    **모델 보정자인 EkfConfig 와 역할이 다르다** — 이쪽은 측정 잡음을 걸러 MPC 피드백에
    넣을 상태를 만드는 것이 전부다(`src/estimation/state_estimator.py`).
    이 그룹을 참조하면 **세 케이스 모두** 필터링된 x_hat 을 피드백받는다. 참조하지
    않으면 종전대로 생측정값(또는 센서가 없으면 참 상태)이 그대로 들어간다.

    상태 순서 = [v_y, gamma, e_psi, e_y] (4). 측정도 전상태 4채널.
    이름은 MPC 가중치(W_/R_)·모델보정 KF(Q_kf/R_kf)와 겹치지 않게 Q_diag/R_diag 로 둔다.
    """
    Q_diag: tuple[float, ...]
    R_diag: tuple[float, ...]
    P0_diag: tuple[float, ...]
    tuning_method: str = "manual"
    tuning_budget_closed_loop_evals: int = 0
    tuning_selection_scenario: str = ""

    def __post_init__(self) -> None:
        for name, n in (("Q_diag", 4), ("R_diag", 4), ("P0_diag", 4)):
            v = getattr(self, name)
            if len(v) != n:
                raise ValueError(f"StateKfConfig.{name} 는 길이 {n} (got {len(v)}).")
            if any(c < 0.0 for c in v):
                raise ValueError(f"StateKfConfig.{name} 성분은 음수가 아니어야 한다.")

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "StateKfConfig":
        return cls(
            Q_diag=tuple(float(v) for v in d["Q_diag"]),
            R_diag=tuple(float(v) for v in d["R_diag"]),
            P0_diag=tuple(float(v) for v in d["P0_diag"]),
            tuning_method=str(d.get("tuning_method", "manual")),
            tuning_budget_closed_loop_evals=int(d.get("tuning_budget_closed_loop_evals", 0)),
            tuning_selection_scenario=str(d.get("tuning_selection_scenario", "")),
        )


@dataclass(frozen=True)
class SensorConfig:
    """상태 측정 센서 모델 (Phase 8c).

    Phase 8 까지는 세 케이스 모두 **참 상태**를 피드백받았다(이상적 센서). 이 그룹을
    참조하면 제어 피드백에 들어가는 상태가 잡음 섞인 측정값으로 바뀐다.
    그룹을 참조하지 않으면 종전대로 이상적 센서다 — 기존 결과가 그대로 재현된다.

    채널 순서는 상태와 동일: [v_y, gamma, e_psi, e_y].
    단위: noise_std [m/s, rad/s, rad, m], bias_amp 동일, bias_freq_hz [Hz].

    측정 모델:
        y_k = x_k + n_k + b*sin(2*pi*f*t_k)
        n_k ~ N(0, diag(noise_std^2))          # 백색잡음 (기본)
        b, f                                    # 결정론적 사인 바이어스 (기본 0)

    **사인 바이어스는 기본 0 이다.** 결정론적 바이어스는 KF 의 영평균 잡음 가정을
    깨서 비교군을 부당하게 약화시킨다(ekf-baseline.md 「의도적/실수로 낮추는 설정」).
    쓰려면 그 사실을 리포트에 명시하고 R_kf 도 함께 재검토하라.
    """
    noise_std: tuple[float, ...]
    bias_amp: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0)
    bias_freq_hz: float = 0.0
    note: str = ""

    def __post_init__(self) -> None:
        if len(self.noise_std) != 4:
            raise ValueError(f"noise_std 는 길이 4 (got {len(self.noise_std)}).")
        if len(self.bias_amp) != 4:
            raise ValueError(f"bias_amp 는 길이 4 (got {len(self.bias_amp)}).")
        for name in ("noise_std", "bias_amp"):
            if any(v < 0.0 for v in getattr(self, name)):
                raise ValueError(f"SensorConfig.{name} 성분은 음수가 아니어야 한다.")
        if self.bias_freq_hz < 0.0:
            raise ValueError("bias_freq_hz 는 음수가 아니어야 한다.")

    @property
    def noise_var(self) -> tuple[float, ...]:
        """잡음 분산 (R_kf 를 센서 사양에 맞출 때 쓴다)."""
        return tuple(s * s for s in self.noise_std)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SensorConfig":
        return cls(
            noise_std=tuple(float(v) for v in d["noise_std"]),
            bias_amp=tuple(float(v) for v in d.get("bias_amp", (0.0,) * 4)),
            bias_freq_hz=float(d.get("bias_freq_hz", 0.0)),
            note=str(d.get("note", "")),
        )


@dataclass(frozen=True)
class GpConfig:
    """offline GP 설정 (.claude/rules/gp-residual.md).

    입력 z=[v_y,gamma,delta] (3D), 채널 v_y·gamma 독립 2개, ARD RBF, Type-II ML.
    M: 딕셔너리 크기(고정). 초기 하이퍼파라미터는 표준화 공간 기준.
    """
    M: int
    init_lengthscale: float = 1.0
    init_sigma_f: float = 1.0
    init_sigma_n: float = 0.1
    sigma_n_floor: float = 1.0e-2   # 표준화 단위 잡음 하한 (정칙화 — 보간 과적합 방지)
    jitter: float = 1.0e-8
    # GP 학습 데이터를 수집할 experiment 이름. 학습 궤적과 평가 궤적을 분리하기 위해
    # **config 로 고정**한다 (gp-residual.md 데이터 위생). 평가 시나리오가 바뀌어도
    # 학습 출처는 이 값 하나로 고정되므로, 평가가 나쁘다고 학습 데이터를 슬쩍 바꾸는
    # 일이 구조적으로 막힌다. 이 experiment 는 gp 그룹을 참조하면 안 된다(순환).
    train_experiment: str = "gp_train"
    # 공정성: KF 와 같은 형식의 튜닝 예산 기록 (EkfConfig 의 대응 필드와 짝).
    tuning_method: str = "type2_ml"
    tuning_budget_closed_loop_evals: int = 0
    tuning_selection_scenario: str = ""

    def __post_init__(self) -> None:
        if not (isinstance(self.M, int) and self.M > 0):
            raise ValueError(f"M 은 양의 정수여야 한다 (got {self.M!r}).")
        for name in ("init_lengthscale", "init_sigma_f", "init_sigma_n",
                     "sigma_n_floor", "jitter"):
            if not (getattr(self, name) > 0.0):
                raise ValueError(f"GpConfig.{name} 는 양수여야 한다.")
        if not self.train_experiment:
            raise ValueError("GpConfig.train_experiment 가 비어 있다.")

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "GpConfig":
        return cls(
            M=int(d["M"]),
            init_lengthscale=float(d.get("init_lengthscale", 1.0)),
            init_sigma_f=float(d.get("init_sigma_f", 1.0)),
            init_sigma_n=float(d.get("init_sigma_n", 0.1)),
            sigma_n_floor=float(d.get("sigma_n_floor", 1.0e-2)),
            jitter=float(d.get("jitter", 1.0e-8)),
            train_experiment=str(d.get("train_experiment", "gp_train")),
            tuning_method=str(d.get("tuning_method", "type2_ml")),
            tuning_budget_closed_loop_evals=int(d.get("tuning_budget_closed_loop_evals", 0)),
            tuning_selection_scenario=str(d.get("tuning_selection_scenario", "")),
        )


def _dataclass_to_plain(obj: Any) -> dict[str, Any]:
    """dataclass -> 순수 dict (tuple 등을 YAML/JSON 친화 형태로)."""
    d = asdict(obj)
    return {k: (list(v) if isinstance(v, tuple) else v) for k, v in d.items()}


# 조합 파일에서 그룹 참조가 아닌 예약 키.
#   plant     : 플랜트 모델 종류. Phase 8 이전엔 스크립트가 주입했으나, 케이스 정의는
#               config 로만 한다는 규칙(sim-experiment.md)에 따라 config 로 올렸다.
#   overrides : 시나리오 정의값. "group.field: value" 점표기.
_PLANT_KINDS = ("linear", "nonlinear")
_RESERVED_KEYS = ("plant", "overrides")


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
    gp: "GpConfig | None" = None
    sensor: "SensorConfig | None" = None
    state_kf: "StateKfConfig | None" = None
    plant: str = "linear"
    # 조합 파일이 참조한 그룹 이름 -> config 이름 (재현성 스냅샷에 남긴다).
    raw_refs: dict[str, str] = field(default_factory=dict)
    # 적용된 시나리오 override ("group.field" -> 값). 스냅샷에 그대로 남긴다.
    overrides: dict[str, Any] = field(default_factory=dict)

    def to_snapshot(self) -> dict[str, Any]:
        """재현성 스냅샷용 dict 덤프. 참조가 아니라 조립된 값 전체를 담는다
        (.claude/rules/sim-experiment.md "재현성").

        override 는 **적용된 뒤의 값**이 그룹 덤프에 반영되고, 무엇을 덮었는지는
        `overrides` 키에 따로 남는다. 통제변수 diff 는 그룹 덤프끼리 비교하면 된다.
        """
        snap = {
            "name": self.name,
            "refs": dict(self.raw_refs),
            "plant": self.plant,
            "overrides": dict(self.overrides),
            "vehicle": _dataclass_to_plain(self.vehicle),
            "sim": _dataclass_to_plain(self.sim),
        }
        if self.path is not None:
            snap["path"] = _dataclass_to_plain(self.path)
        if self.mpc is not None:
            snap["mpc"] = _dataclass_to_plain(self.mpc)
        if self.ekf is not None:
            snap["ekf"] = _dataclass_to_plain(self.ekf)
        if self.gp is not None:
            snap["gp"] = _dataclass_to_plain(self.gp)
        if self.sensor is not None:
            snap["sensor"] = _dataclass_to_plain(self.sensor)
        if self.state_kf is not None:
            snap["state_kf"] = _dataclass_to_plain(self.state_kf)
        return snap


# 그룹 이름 -> 해당 dataclass 로더 매핑.
# 새 그룹을 붙일 때 여기에만 추가한다 (if 분기 확산 방지).
_GROUP_LOADERS = {
    "vehicle": VehicleConfig.from_dict,
    "sim": SimConfig.from_dict,
    "path": PathConfig.from_dict,
    "mpc": MpcConfig.from_dict,
    "ekf": EkfConfig.from_dict,
    "gp": GpConfig.from_dict,
    "sensor": SensorConfig.from_dict,
    "state_kf": StateKfConfig.from_dict,
}


def load_group(group: str, name: str, configs_dir: Path = CONFIGS_DIR) -> Any:
    """단일 그룹 config를 로드한다. 예: load_group("vehicle", "sedan")."""
    if group not in _GROUP_LOADERS:
        raise KeyError(f"알 수 없는 config 그룹: {group!r}. 지원: {sorted(_GROUP_LOADERS)}")
    path = configs_dir / group / f"{name}.yaml"
    return _GROUP_LOADERS[group](_load_yaml(path))


def _apply_overrides(groups: dict[str, Any], overrides: dict[str, Any],
                     exp_name: str) -> dict[str, Any]:
    """`"group.field": value` 형태의 override 를 로드된 그룹 dataclass 에 적용한다.

    frozen dataclass 이므로 `replace` 로 새 인스턴스를 만든다. `replace` 는
    `__init__` 을 다시 타므로 각 그룹의 `__post_init__` 유효성 검사가 그대로 걸린다
    (예: `a_y_max > 0`). 검사를 우회하는 경로를 만들지 않는 것이 요점이다.
    """
    out = dict(groups)
    for dotted, value in overrides.items():
        if not isinstance(dotted, str) or dotted.count(".") != 1:
            raise ValueError(
                f"experiment '{exp_name}' 의 override 키는 'group.field' 형태여야 한다 "
                f"(got {dotted!r})."
            )
        group, fname = dotted.split(".")
        if group not in out or out[group] is None:
            raise ValueError(
                f"experiment '{exp_name}' 가 override 하려는 그룹 '{group}' 를 "
                f"참조하지 않는다 ({dotted})."
            )
        cfg = out[group]
        valid = {f.name for f in fields(cfg)}
        if fname not in valid:
            raise ValueError(
                f"experiment '{exp_name}' override '{dotted}': '{group}' 에 그런 필드가 "
                f"없다. 가능한 필드: {sorted(valid)}"
            )
        out[group] = replace(cfg, **{fname: value})
    return out


def load_experiment(name: str, configs_dir: Path = CONFIGS_DIR) -> ExperimentConfig:
    """configs/experiment/<name>.yaml 조합 파일을 로드한다.

    조합 파일은 그룹 참조(`group: name`)를 담는다. 그룹 값을 직접 담으면 잘못된 것이다.
    현재 vehicle, sim 은 필수. 나머지 그룹은 Phase 진행에 따라 추가된다.

    예약 키 두 개는 그룹 참조가 아니다 (Phase 8):
      plant: linear | nonlinear      # 플랜트 모델. 스크립트 주입 금지, config 로만.
      overrides: {"path.a_y_max": 6.0}  # 시나리오 정의값

    override 를 둔 이유: 분포이동 시나리오(a_y=4/6, dlc)를 표현하려면 path 그룹을
    통째로 복제해야 하는데, 그러면 세그먼트 정의가 복붙되어 sim-experiment.md 의
    "같은 값이 두 파일에 복붙되어 있으면 잘못된 것" 을 위반한다. 시나리오를 정의하는
    **한 개 값**만 조합 파일에 두는 쪽이 복제보다 낫다는 판단이다 (Phase 8 결정).
    """
    exp_path = configs_dir / "experiment" / f"{name}.yaml"
    refs_raw = _load_yaml(exp_path)

    plant = refs_raw.get("plant", "linear")
    if plant not in _PLANT_KINDS:
        raise ValueError(
            f"experiment '{name}' 의 plant 는 {_PLANT_KINDS} 중 하나여야 한다 (got {plant!r})."
        )

    overrides = refs_raw.get("overrides", {}) or {}
    if not isinstance(overrides, dict):
        raise ValueError(f"experiment '{name}' 의 overrides 는 매핑이어야 한다 (got {overrides!r}).")

    # 예약 키를 뺀 나머지는 전부 그룹 참조(문자열)여야 한다 (값 복붙 방지).
    refs: dict[str, str] = {}
    for group, ref in refs_raw.items():
        if group in _RESERVED_KEYS:
            continue
        if not isinstance(ref, str):
            raise ValueError(
                f"experiment '{name}' 의 '{group}' 는 그룹 참조(문자열)여야 한다. "
                f"값을 직접 적지 마라 (그룹 값을 바꾸려면 overrides 를 써라, got {ref!r})."
            )
        refs[group] = ref

    for required in ("vehicle", "sim"):
        if required not in refs:
            raise ValueError(f"experiment '{name}' 에 필수 그룹 '{required}' 참조가 없다.")

    groups = {g: (load_group(g, refs[g], configs_dir) if g in refs else None)
              for g in _GROUP_LOADERS}
    groups = _apply_overrides(groups, overrides, name)

    return ExperimentConfig(
        name=name,
        vehicle=groups["vehicle"],
        sim=groups["sim"],
        path=groups["path"],
        mpc=groups["mpc"],
        ekf=groups["ekf"],
        gp=groups["gp"],
        sensor=groups["sensor"],
        state_kf=groups["state_kf"],
        plant=plant,
        raw_refs=refs,
        overrides=dict(overrides),
    )
