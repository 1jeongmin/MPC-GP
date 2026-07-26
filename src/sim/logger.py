"""제어 스텝별 로깅 + 재현성 메타 (.claude/rules/sim-experiment.md).

로깅 스키마 (제어 스텝마다):
    t, x(4), delta, vx, kappa, s
    solve_time, ipopt_iter, converged
    residual r(4)                      # 4채널 전부. 3,4번(e_psi,e_y)은 진단용
    gp_mean(2), gp_var(2)              # GP 케이스 — MpcBase.step_log 훅 (Phase 8~)
    ekf_x_hat(6), ekf_d_hat(2),        # EKF 케이스 (Phase 6~)
    ekf_P_diag(6), ekf_P_d_block(2,2), ekf_innovation(4)

케이스별 필드는 runner 가 **분기 없이** 붙인다: estimator 가 있으면 ekf_*,
컨트롤러가 step_log() 를 내놓으면 그 dict 를 그대로 합류시킨다.

results/ 에는 npz 로 저장한다. 재현성 메타(config 스냅샷, git hash, seed,
라이브러리 버전)가 없으면 결과를 만들지 마라.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


class Log:
    """제어 스텝 로그 누적기. append 로 행을 쌓고 finalize 로 배열 dict 반환."""

    def __init__(self) -> None:
        self._rows: list[dict[str, Any]] = []

    def append(self, **kwargs: Any) -> None:
        self._rows.append(kwargs)

    def finalize(self) -> dict[str, np.ndarray]:
        """수집된 행을 채널별 numpy 배열로 스택한다. 빈 로그는 예외."""
        if not self._rows:
            raise RuntimeError("로그가 비어 있다.")
        keys = self._rows[0].keys()
        out: dict[str, np.ndarray] = {}
        for k in keys:
            vals = [r[k] for r in self._rows]
            # None(예: ipopt_iter 미수렴)은 nan 으로.
            if any(v is None for v in vals):
                vals = [np.nan if v is None else v for v in vals]
            out[k] = np.array(vals)
        return out


def save_npz(out_dir: Path, log: dict[str, np.ndarray], name: str = "log.npz") -> Path:
    """로그 배열 dict 를 npz 로 저장."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    np.savez(path, **log)
    return path


def _git_info() -> dict:
    """git commit hash 와 dirty 여부. 재현성 필수 항목."""
    def _run(args: list[str]) -> str:
        return subprocess.check_output(args, cwd=ROOT, text=True).strip()
    try:
        return {"commit": _run(["git", "rev-parse", "HEAD"]),
                "dirty": _run(["git", "status", "--porcelain"]) != ""}
    except Exception as e:  # noqa: BLE001
        return {"commit": None, "dirty": None, "error": str(e)}


def _lib_versions() -> dict:
    import casadi
    import scipy
    return {"python": sys.version.split()[0], "numpy": np.__version__,
            "scipy": scipy.__version__, "casadi": casadi.__version__}


def write_run_meta(out_dir: Path, run_id: str, config_snapshot: dict,
                   seed: int | None, extra: dict | None = None) -> Path:
    """재현성 메타를 meta.json 으로 저장한다.

    config 스냅샷(참조가 아닌 조립된 값 복사본), git hash/dirty, seed,
    라이브러리 버전을 담는다. 하나라도 없으면 결과를 만들지 마라
    (.claude/rules/sim-experiment.md 재현성).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "run_id": run_id,
        "seed": seed,
        "git": _git_info(),
        "libs": _lib_versions(),
        "config": config_snapshot,
    }
    if extra:
        meta.update(extra)
    path = out_dir / "meta.json"
    with path.open("w", encoding="utf-8") as fp:
        json.dump(meta, fp, ensure_ascii=False, indent=2, default=str)
    return path
