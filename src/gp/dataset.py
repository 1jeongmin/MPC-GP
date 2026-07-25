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
    """잔차 데이터셋. Z:(N,3) 특징, R:(N,2) 타깃."""
    Z: np.ndarray
    R: np.ndarray
    meta: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return self.Z.shape[0]

    def subsample(self, M: int) -> "ResidualDataset":
        """균일 subsample 로 M개 점 선택 (딕셔너리 후보). N<=M 이면 그대로."""
        n = len(self)
        if M >= n:
            return ResidualDataset(self.Z.copy(), self.R.copy(), dict(self.meta))
        idx = np.unique(np.linspace(0, n - 1, M).astype(int))
        return ResidualDataset(self.Z[idx], self.R[idx],
                               {**self.meta, "subsampled_from": n, "M": len(idx)})


def collect_residual_data(vehicle: VehicleConfig, mpc_cfg, sim: SimConfig,
                          path, x0: np.ndarray | None = None,
                          plant: str = "nonlinear") -> ResidualDataset:
    """명목 MPC 로 (선택적으로 비선형) 플랜트를 주행해 잔차 데이터를 모은다.

    특징 z_i = [v_y_i, gamma_i, delta_i] (스텝 i 상태·입력),
    타깃 r_i = residual_i[:2] (동적 채널 v_y, gamma).
    plant: "nonlinear"(Fiala, 기본) 또는 "linear"(잔차 0 검증용).
    """
    from src.models.integrators import make_rhs_np
    from src.sim.runner import run_closed_loop

    reference = Reference(path, vehicle.vx_range)
    controller = make_nominal_mpc(vehicle, mpc_cfg, sim.dt_ctrl)
    nominal_step = build_step_function(NominalStepModel(vehicle, sim.dt_ctrl))
    plant_rhs = make_nonlinear_rhs_np(vehicle) if plant == "nonlinear" else make_rhs_np(vehicle)

    log = run_closed_loop(controller, reference, plant_rhs, nominal_step, sim, x0=x0)
    Z = np.column_stack([log["x"][:, 0], log["x"][:, 1], log["delta"]])   # vy, gamma, delta
    R = log["residual"][:, 0:2]                                            # r_vy, r_gamma
    meta = {"plant": plant, "n": Z.shape[0], "seed": sim.seed}
    return ResidualDataset(Z, R, meta)


def save_dataset(path: Path, ds: ResidualDataset, config_snapshot: dict,
                 seed: int | None) -> Path:
    """데이터셋을 npz 로 저장하고 config/git/seed 메타를 함께 남긴다."""
    from src.sim.logger import _git_info, _lib_versions
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {**ds.meta, "config": config_snapshot, "seed": seed,
            "git": _git_info(), "libs": _lib_versions()}
    np.savez(path, Z=ds.Z, R=ds.R, meta=np.array(meta, dtype=object))
    return path


def load_dataset(path: Path) -> ResidualDataset:
    d = np.load(path, allow_pickle=True)
    return ResidualDataset(d["Z"], d["R"], d["meta"].item())
