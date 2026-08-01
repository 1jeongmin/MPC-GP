"""GP 잔차 데이터셋 (.claude/rules/gp-residual.md).

특징 z = [v_y, gamma, delta], 타깃 r = [r_vy, r_gamma] (명목 대비 잔차, 동적 채널만).
학습 궤적과 평가 궤적을 분리한다 (같은 로그로 학습·평가 동시 금지).
데이터셋에 생성 config/git hash/seed 를 동봉해 저장한다 (data/ 아래, git 제외).

잔차 정의는 runner 와 동일 (명목 이산 전이 대비). 여기서 재구현하지 않고
run_closed_loop 이 로깅한 residual 을 그대로 쓴다.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.config import SimConfig, VehicleConfig
from src.control.mpc_base import build_step_function
from src.control.mpc_nominal import NominalStepModel, make_nominal_mpc
from src.models.nonlinear_bicycle import make_nonlinear_rhs_np
from src.path.reference import Reference


@dataclass
class Standardizer:
    """열별 표준화 (z-score). 통계를 저장·재적용한다."""
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, X: np.ndarray) -> "Standardizer":
        X = np.atleast_2d(X)
        mean = X.mean(axis=0)
        scale = X.std(axis=0)
        scale = np.where(scale < 1e-12, 1.0, scale)   # 상수 열 방어
        return cls(mean=mean, scale=scale)

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (np.atleast_2d(X) - self.mean) / self.scale

    def inverse(self, Xs: np.ndarray) -> np.ndarray:
        return np.atleast_2d(Xs) * self.scale + self.mean


@dataclass
class ResidualDataset:
    """잔차 데이터셋. Z:(N,3) 특징, R:(N,2) 타깃.

    s/t 는 같은 궤적의 호길이[m]·시간[s] (선택). 학습 진단 그림
    (`make_gp_training_figure`)이 요구하므로 데이터셋이 스스로 들고 있게 한다 —
    없으면 그림을 그리려고 학습 주행을 다시 해야 한다.

    delta_applied 는 그 스텝에 **적용된** 입력 delta_k (선택). 특징 `Z[:,2]` 는
    배포 규약에 맞춰 delta_{k-1} 이므로 여기에 들어가지 않는다. 옛 규약(delta_k)으로
    학습했을 때와 대조하는 A/B 진단에만 쓴다 — 이것 없이 비교하려면 수집(수 분)을
    다시 해야 한다. **학습 경로는 이 배열을 쓰지 않는다.**
    """
    Z: np.ndarray
    R: np.ndarray
    meta: dict = field(default_factory=dict)
    s: np.ndarray | None = None
    t: np.ndarray | None = None
    delta_applied: np.ndarray | None = None

    def __len__(self) -> int:
        return self.Z.shape[0]

    def _take(self, idx, meta: dict) -> "ResidualDataset":
        """행 인덱스로 부분집합을 만든다 (선택 배열들을 같은 인덱스로 실어 나른다)."""
        pick = lambda a: None if a is None else a[idx]      # noqa: E731
        return ResidualDataset(self.Z[idx], self.R[idx], meta,
                               s=pick(self.s), t=pick(self.t),
                               delta_applied=pick(self.delta_applied))

    def subsample(self, M: int) -> "ResidualDataset":
        """균일 subsample 로 M개 점 선택 (딕셔너리 후보). N<=M 이면 그대로."""
        n = len(self)
        if M >= n:
            return self._take(np.arange(n), dict(self.meta))
        idx = np.unique(np.linspace(0, n - 1, M).astype(int))
        return self._take(idx, {**self.meta, "subsampled_from": n, "M": len(idx)})

    def take(self, idx: np.ndarray) -> "ResidualDataset":
        """임의 인덱스로 부분집합 (블록 분할 홀드아웃 등). meta 에 크기를 남긴다."""
        idx = np.asarray(idx)
        return self._take(idx, {**self.meta, "selected_from": len(self), "n": len(idx)})


def collect_residual_data(vehicle: VehicleConfig, mpc_cfg, sim: SimConfig,
                          path, x0: np.ndarray | None = None,
                          plant: str = "nonlinear", sensor=None,
                          state_estimator=None, rng=None) -> ResidualDataset:
    """명목 MPC 로 (선택적으로 비선형) 플랜트를 주행해 잔차 데이터를 모은다.

    특징 z_i = [v_y_i, gamma_i, delta_{i-1}], 타깃 r_i = residual_i[:2] (동적 채널).
    plant: "nonlinear"(Fiala, 기본) 또는 "linear"(잔차 0 검증용).

    sensor / state_estimator: 주면 **배포와 같은 조건**(측정잡음 + 공통 상태추정기)
    으로 주행해 데이터를 모은다. 안 주면 종전대로 이상적 센서다.

    ## 특징은 `x_fb`(제어기가 실제로 받는 상태)에서 뽑는다 — train/deploy 정합

    배포에서 GP 는 `mpc_gp.GPStepModel.set_operating_point(x_fb, u_prev)` 로
    **필터링된 상태**에서 z 를 만든다. 학습을 참 상태로 하면 특징 분포가 어긋나
    (같은 z 라도 실제로는 잡음이 실린 값이 들어옴) GP 가 산포를 과소평가한다.
    `x_fb` 는 runner 가 항상 로깅하고 센서·필터가 없으면 `x_fb == x` 이므로,
    이상적 센서 경로의 결과는 종전과 **완전히 동일**하다.

    타깃은 **참 상태 기준 잔차 그대로**다(`gp-residual.md` 의 정의를 바꾸지 않는다).
    즉 GP 는 `E[r_true | z_hat]` 를 배운다. 상태추정 오차 전파는 상태추정기의
    책임이지 모델 보정자의 몫이 아니다 (`ekf-baseline.md` 역할 혼동 금지).

    ## delta 성분은 `delta_{k-1}` 이다 — 같은 이유의 두 번째 정합

    배포에서 GP 는 solve **전에** 평가되므로 `delta_k` 를 알 수 없고 `u_prev` 를 쓴다
    (`mpc_gp.set_operating_point`). 학습을 `delta_k`(적용된 입력)로 짝지으면 배포와
    다른 특징을 배운다. 룰(`gp-residual.md` 「입력」)은 `z=[v_y, gamma, delta]` 라고만
    하고 시점을 규정하지 않으므로, 배포가 실현 가능한 쪽으로 학습을 맞춘다.

    타깃 `r_k` 는 `delta_k` 로 생성된 값 그대로 둔다(잔차 정의를 바꾸지 않는다).
    즉 GP 는 `E[r_k | v_y_k, gamma_k, delta_{k-1}]` 를 배운다 — 배포가 실제로 던지는
    질문과 정확히 같다.
    """
    from src.models.integrators import make_rhs_np
    from src.sim.runner import run_closed_loop

    reference = Reference(path, vehicle.vx_range)
    controller = make_nominal_mpc(vehicle, mpc_cfg, sim.dt_ctrl)
    nominal_step = build_step_function(NominalStepModel(vehicle, sim.dt_ctrl))
    plant_rhs = make_nonlinear_rhs_np(vehicle) if plant == "nonlinear" else make_rhs_np(vehicle)

    log = run_closed_loop(controller, reference, plant_rhs, nominal_step, sim, x0=x0,
                          rng=rng, sensor=sensor, state_estimator=state_estimator)
    Z = np.column_stack([log["x_fb"][:, 0], log["x_fb"][:, 1], log["delta_prev"]])
    R = log["residual"][:, 0:2]                                            # r_vy, r_gamma
    meta = {"plant": plant, "n": Z.shape[0], "seed": sim.seed,
            "filtered_features": state_estimator is not None,
            "noisy_sensor": sensor is not None,
            "delta_feature": "prev"}      # z[2] 규약. "applied"(옛 규약)와 구분한다.
    return ResidualDataset(Z, R, meta, s=log["s"].copy(), t=log["t"].copy(),
                           delta_applied=log["delta"].copy())


BASE_DIM = 3     # 한 시점의 기본 특징 [v_y, gamma, delta_prev]


def lap_block_split(ds: ResidualDataset, total_length: float,
                    n_holdout_laps: int = 1) -> tuple[ResidualDataset, ResidualDataset]:
    """폐루프 주행을 **랩 경계**로 갈라 (학습용, 홀드아웃) 를 만든다.

    total_length: 한 바퀴 호길이[m] (`Reference.total_length`).
    n_holdout_laps: 뒤에서 몇 바퀴를 홀드아웃으로 뗄지 (기본 1 = 마지막 한 바퀴).

    ## 왜 무작위 분할이면 안 되는가

    `gp-residual.md` 「데이터 위생」이 **학습 궤적과 평가 궤적을 분리하라**고 못박는다.
    같은 주행에서 무작위로 점을 빼서 평가하면 홀드아웃 점이 항상 학습점의 시간축
    이웃이 된다 — 52,665스텝에서 2000점을 균일 subsample 하면 간격이 26스텝(0.53 s)
    이라, 50 Hz 시계열에서는 사실상 보간만 하면 맞는 문제가 된다. 그렇게 잰
    캘리브레이션은 낙관적으로 편향된다(2026-08-01 진단).

    랩 단위로 자르면 홀드아웃 구간의 어떤 점도 학습 구간과 시간축으로 인접하지 않는다.
    같은 맵을 다시 도는 것이라 **분포는 같고 궤적만 다르다** — 이것이 재려는 조건이다.

    ## ★ 마지막 **미완성 바퀴 조각은 버린다** (2026-08-01 버그 수정)

    `run_closed_loop` 은 `s > total_length*n_laps + stop_margin` 에서 멈추므로 주행이
    목표를 몇 미터 넘겨서 끝난다(측정: 22774.7 m vs 목표 22770.0 m). 바퀴 수를
    `floor(s/L).max()+1` 로 세면 그 꼬리 4.7 m 가 "한 바퀴 더"로 잡혀서, 홀드아웃이
    **마지막 10개 점**만 남는 사고가 난다. 그 구간이 하필 직선이면 잔차가 두 자릿수
    작아서 z_std 가 0 에 가깝게 나오고 (실제로 그렇게 나왔다) 캘리브레이션 수치가
    통째로 무의미해진다. 그래서 **완주한 바퀴만** 센다.
    """
    if ds.s is None:
        raise ValueError("lap_block_split 은 호길이 s 가 필요하다 (옛 데이터셋에는 없다).")
    if not (total_length > 0.0):
        raise ValueError(f"total_length 는 양수여야 한다 (got {total_length}).")
    s = np.asarray(ds.s, float)
    n_complete = int(np.floor(s.max() / total_length))      # 완주한 바퀴 수
    if n_complete <= n_holdout_laps:
        raise ValueError(f"완주 바퀴 수({n_complete})가 홀드아웃({n_holdout_laps})보다 "
                         "많아야 한다.")
    cut = (n_complete - n_holdout_laps) * total_length
    end = n_complete * total_length                          # 이 뒤는 미완성 조각 -> 버림
    fit = ds.take(np.flatnonzero(s < cut))
    held = ds.take(np.flatnonzero((s >= cut) & (s < end)))
    return fit, held


def lagged_dim(n_lags: int, lag_mode: str = "full") -> int:
    """지연 적용 후 특징 차원. `full` -> 3*(L+1), `delta` -> 3+L."""
    if lag_mode == "full":
        return BASE_DIM * (n_lags + 1)
    if lag_mode == "delta":
        return BASE_DIM + n_lags
    raise ValueError(f"알 수 없는 lag_mode: {lag_mode!r} ('full' 또는 'delta').")


def apply_lags(ds: ResidualDataset, n_lags: int,
               lag_mode: str = "full") -> ResidualDataset:
    """특징에 시간지연 사본을 이어 붙인다. `Z:(N,3)` -> `(N, lagged_dim(...))`.

        b_k = [v_y_k, gamma_k, delta_{k-1}]                (기본 특징, 한 시점)

    `lag_mode="full"` — 상태·입력을 **통째로** 지연시킨다 (차원 3*(L+1)):

        z_k = [b_k, b_{k-1}, ..., b_{k-n_lags}]

    `lag_mode="delta"` — **입력 이력만** 붙인다 (차원 3+L). 상태는 현재만 쓴다:

        z_k = [v_y_k, gamma_k, delta_{k-1}, delta_{k-2}, ..., delta_{k-1-n_lags}]

    두 모드를 다 두는 이유: "과거를 준다"에 두 해석이 있고 어느 쪽이 잔차를 설명하는지
    미리 알 수 없다. `delta` 모드는 차원이 훨씬 덜 늘어(9D vs 5D) `gp-residual.md`
    「차원 최소 유지」와 덜 충돌하므로, 같은 효과가 나면 이쪽이 낫다.

    ★ 어느 모드든 `delta` 성분의 가장 최신 값은 `delta_{k-1}` 이다. `delta_k` 는 solve
    전이라 배포가 알 수 없어서 쓸 수 없다(그걸 쓰면 재현 불가능한 GP 가 된다).

    `n_lags=0` 이면 아무것도 하지 않고 원본을 그대로 돌려준다(기존 경로 무변경).

    ## 왜 0 으로 패딩하는가 — 배포와 정확히 같게 만들기 위해

    `k < n_lags` 인 초반 스텝에는 과거가 없다. 배포(`mpc_gp.GPStepModel`)도 주행
    시작 시점에 이력이 없고, runner 가 `x=0`, `u_prev=0` 에서 출발하므로
    (`runner.py`) **없는 과거를 0 으로 두는 것이 물리적으로도 실제 초기조건과 같다.**
    학습·배포가 같은 규약을 쓰면 초반 몇 스텝도 특별취급 없이 일치한다.

    ## 반드시 **subsample 전에** 적용하라

    지연은 행 순서가 시간순이고 빠진 스텝이 없다는 전제 위에서만 의미가 있다.
    딕셔너리로 솎아낸 뒤 적용하면 "직전 스텝"이 실제로는 26스텝 전이 되어버린다.
    그래서 부분집합 데이터셋이면 예외를 던진다.
    """
    if n_lags < 0:
        raise ValueError(f"n_lags 는 0 이상이어야 한다 (got {n_lags}).")
    lagged_dim(n_lags, lag_mode)           # lag_mode 유효성 검사
    if n_lags == 0:
        return ds
    for key in ("subsampled_from", "selected_from"):
        if key in ds.meta:
            raise ValueError(
                f"apply_lags 는 원본(시간순 연속) 데이터셋에만 쓸 수 있다 — meta 에 "
                f"'{key}' 가 있어 이미 솎아진 데이터다. subsample 전에 적용하라.")
    if ds.Z.shape[1] != BASE_DIM:
        raise ValueError(f"기본 특징 차원이 {BASE_DIM} 이 아니다 (got {ds.Z.shape[1]}) — "
                         "지연이 이미 적용된 데이터셋에 다시 적용하려는 것 같다.")

    blocks = [ds.Z]
    for lag in range(1, n_lags + 1):
        shifted = np.zeros_like(ds.Z)
        shifted[lag:] = ds.Z[:-lag]        # 앞쪽 lag 개 행은 0 패딩 (이력 없음)
        blocks.append(shifted if lag_mode == "full" else shifted[:, 2:3])
    Z = np.hstack(blocks)
    assert Z.shape[1] == lagged_dim(n_lags, lag_mode)
    meta = {**ds.meta, "n_lags": n_lags, "lag_mode": lag_mode,
            "feature_dim": Z.shape[1]}
    return ResidualDataset(Z, ds.R.copy(), meta, s=ds.s, t=ds.t,
                           delta_applied=ds.delta_applied)


def save_dataset(path: Path, ds: ResidualDataset, config_snapshot: dict,
                 seed: int | None) -> Path:
    """데이터셋을 npz 로 저장하고 config/git/seed 메타를 함께 남긴다."""
    from src.sim.logger import _git_info, _lib_versions
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {**ds.meta, "config": config_snapshot, "seed": seed,
            "git": _git_info(), "libs": _lib_versions()}
    extra = {name: arr for name, arr in
             (("s", ds.s), ("t", ds.t), ("delta_applied", ds.delta_applied))
             if arr is not None}
    np.savez(path, Z=ds.Z, R=ds.R, meta=np.array(meta, dtype=object), **extra)
    return path


def load_dataset(path: Path) -> ResidualDataset:
    """npz 에서 복원. 선택 배열은 옛 파일에 없을 수 있으므로 없으면 None (하위호환)."""
    d = np.load(path, allow_pickle=True)
    get = lambda k: d[k] if k in d.files else None      # noqa: E731
    return ResidualDataset(d["Z"], d["R"], d["meta"].item(),
                           s=get("s"), t=get("t"),
                           delta_applied=get("delta_applied"))
