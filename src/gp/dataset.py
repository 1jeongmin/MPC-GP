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
    """
    Z: np.ndarray
    R: np.ndarray
    meta: dict = field(default_factory=dict)
    s: np.ndarray | None = None
    t: np.ndarray | None = None

    def __len__(self) -> int:
        return self.Z.shape[0]

    def subsample(self, M: int) -> "ResidualDataset":
        """균일 subsample 로 M개 점 선택 (딕셔너리 후보). N<=M 이면 그대로."""
        n = len(self)
        if M >= n:
            return ResidualDataset(self.Z.copy(), self.R.copy(), dict(self.meta),
                                   None if self.s is None else self.s.copy(),
                                   None if self.t is None else self.t.copy())
        idx = np.unique(np.linspace(0, n - 1, M).astype(int))
        return ResidualDataset(self.Z[idx], self.R[idx],
                               {**self.meta, "subsampled_from": n, "M": len(idx)},
                               None if self.s is None else self.s[idx],
                               None if self.t is None else self.t[idx])


def collect_residual_data(vehicle: VehicleConfig, mpc_cfg, sim: SimConfig,
                          path, x0: np.ndarray | None = None,
                          plant: str = "nonlinear", sensor=None,
                          state_estimator=None, rng=None) -> ResidualDataset:
    """명목 MPC 로 (선택적으로 비선형) 플랜트를 주행해 잔차 데이터를 모은다.

    특징 z_i = [v_y_i, gamma_i, delta_i], 타깃 r_i = residual_i[:2] (동적 채널).
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
    """
    from src.models.integrators import make_rhs_np
    from src.sim.runner import run_closed_loop

    reference = Reference(path, vehicle.vx_range)
    controller = make_nominal_mpc(vehicle, mpc_cfg, sim.dt_ctrl)
    nominal_step = build_step_function(NominalStepModel(vehicle, sim.dt_ctrl))
    plant_rhs = make_nonlinear_rhs_np(vehicle) if plant == "nonlinear" else make_rhs_np(vehicle)

    log = run_closed_loop(controller, reference, plant_rhs, nominal_step, sim, x0=x0,
                          rng=rng, sensor=sensor, state_estimator=state_estimator)
    Z = np.column_stack([log["x_fb"][:, 0], log["x_fb"][:, 1], log["delta"]])
    R = log["residual"][:, 0:2]                                            # r_vy, r_gamma
    meta = {"plant": plant, "n": Z.shape[0], "seed": sim.seed,
            "filtered_features": state_estimator is not None,
            "noisy_sensor": sensor is not None}
    return ResidualDataset(Z, R, meta, s=log["s"].copy(), t=log["t"].copy())


def save_dataset(path: Path, ds: ResidualDataset, config_snapshot: dict,
                 seed: int | None) -> Path:
    """데이터셋을 npz 로 저장하고 config/git/seed 메타를 함께 남긴다."""
    from src.sim.logger import _git_info, _lib_versions
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {**ds.meta, "config": config_snapshot, "seed": seed,
            "git": _git_info(), "libs": _lib_versions()}
    extra = {}
    if ds.s is not None:
        extra["s"] = ds.s
    if ds.t is not None:
        extra["t"] = ds.t
    np.savez(path, Z=ds.Z, R=ds.R, meta=np.array(meta, dtype=object), **extra)
    return path


def load_dataset(path: Path) -> ResidualDataset:
    """npz 에서 복원. s/t 는 옛 파일에 없을 수 있으므로 없으면 None (하위호환)."""
    d = np.load(path, allow_pickle=True)
    return ResidualDataset(d["Z"], d["R"], d["meta"].item(),
                           s=d["s"] if "s" in d.files else None,
                           t=d["t"] if "t" in d.files else None)
