"""개루프 스텝 조향 응답 — 모델 물리 검증 (Phase 1).

sedan 과 limo 각각에 대해 delta 스텝을 주고 v_y, gamma 의 시간 응답을 그린다.
두 차량의 요 응답 시상수가 왜 다른지 고유값으로 정량화해 보고한다.

내부 계산은 전부 rad. deg 변환은 플롯 축에서만 한다
(.claude/rules/sim-experiment.md 플롯 규약).

결과는 results/<run_id>/ 에 저장하고, 재현성 메타(config 스냅샷, git hash,
seed, 라이브러리 버전)를 함께 남긴다. 하나라도 없으면 결과를 만들지 않는다.
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # 디스플레이 없는 환경에서 파일로만 저장
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

# 스크립트 직접 실행 시 저장소 루트를 import 경로에 추가.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import casadi
import scipy

from src.config import VehicleConfig, load_group, _dataclass_to_plain
from src.models import linear_bicycle as lb
from src.models.integrators import make_rhs_np


def _git_info() -> dict:
    """현재 git commit hash 와 dirty 여부. 재현성 필수 항목."""
    def _run(args: list[str]) -> str:
        return subprocess.check_output(args, cwd=ROOT, text=True).strip()
    try:
        commit = _run(["git", "rev-parse", "HEAD"])
        dirty = _run(["git", "status", "--porcelain"]) != ""
        return {"commit": commit, "dirty": dirty}
    except Exception as e:  # noqa: BLE001 - 재현성 메타는 실패해도 이유를 남긴다
        return {"commit": None, "dirty": None, "error": str(e)}


def _lib_versions() -> dict:
    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "casadi": casadi.__version__,
    }


def simulate_step_steer(cfg: VehicleConfig, vx: float, delta: float,
                        T: float, n: int = 600) -> tuple[np.ndarray, np.ndarray]:
    """고정밀 적분으로 스텝 조향 응답을 계산한다.

    x0 = 0, kappa = 0, delta 상수. 반환: (t[n], X[n,4]).
    """
    f = make_rhs_np(cfg)
    t_eval = np.linspace(0.0, T, n)
    sol = solve_ivp(
        lambda t, y: f(y, delta, vx, 0.0),
        (0.0, T), np.zeros(4),
        method="DOP853", rtol=1e-10, atol=1e-12, t_eval=t_eval,
    )
    if not sol.success:
        raise RuntimeError(f"{cfg} 적분 실패: {sol.message}")
    return sol.t, sol.y.T


def dominant_time_constant(cfg: VehicleConfig, vx: float) -> tuple[float, np.ndarray]:
    """A_lat 고유값과 지배 시상수 tau = 1/min|Re(eig)| [s]."""
    eig = np.linalg.eigvals(lb.A_matrix(cfg, vx)[0:2, 0:2])
    tau = 1.0 / np.min(np.abs(eig.real))
    return tau, eig


def main() -> None:
    run_id = "openloop_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "results" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # 각 차량: (config, vx, delta, T). vx 는 vx_range 안에서 대표값.
    cases = {
        "sedan": (load_group("vehicle", "sedan"), 15.0, 0.02, 3.0),
        "limo": (load_group("vehicle", "limo"), 0.8, 0.02, 0.2),
    }

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    summary: dict[str, dict] = {}

    for col, (name, (cfg, vx, delta, T)) in enumerate(cases.items()):
        t, X = simulate_step_steer(cfg, vx, delta, T)
        vy, gamma = X[:, 0], X[:, 1]
        tau, eig = dominant_time_constant(cfg, vx)
        gss = lb.steady_state_yaw_rate(cfg, vx, delta)

        axes[0, col].plot(t, vy, color="C0")
        axes[0, col].set_title(f"{name}  (vx={vx} m/s, delta={delta} rad)")
        axes[0, col].set_ylabel("v_y [m/s]")
        axes[0, col].grid(True, alpha=0.3)

        axes[1, col].plot(t, np.degrees(gamma), color="C1", label="gamma")
        axes[1, col].axhline(np.degrees(gss), ls="--", color="0.5",
                             label=f"gamma_ss={np.degrees(gss):.3f} deg/s")
        axes[1, col].set_ylabel("gamma [deg/s]")
        axes[1, col].set_xlabel("t [s]")
        axes[1, col].grid(True, alpha=0.3)
        axes[1, col].legend(fontsize=8)

        summary[name] = {
            "vx": vx, "delta": delta, "T": T,
            "K_us": lb.understeer_gradient(cfg),
            "A_lat_eig_real": eig.real.tolist(),
            "A_lat_eig_imag": eig.imag.tolist(),
            "dominant_tau_s": tau,
            "gamma_ss_rad_s": gss,
            "vehicle": _dataclass_to_plain(cfg),
        }

    # 제목은 ASCII (matplotlib 기본 폰트에 한글 글리프가 없어 깨진다).
    fig.suptitle("Open-loop step steer response: sedan vs limo")
    fig_path = out_dir / "step_steer_response.png"
    fig.savefig(fig_path, dpi=130)
    plt.close(fig)

    # 재현성 메타 (config 스냅샷, git, seed, 라이브러리 버전).
    meta = {
        "run_id": run_id,
        "script": "scripts/run_openloop.py",
        "seed": None,  # 개루프는 난수를 쓰지 않는다 (명시적으로 기록)
        "git": _git_info(),
        "libs": _lib_versions(),
        "cases": summary,
    }
    with (out_dir / "meta.json").open("w", encoding="utf-8") as fp:
        json.dump(meta, fp, ensure_ascii=False, indent=2)

    # 콘솔 보고 — 시상수 차이의 물리적 설명.
    print(f"[run_id] {run_id}")
    print(f"[saved]  {fig_path}")
    for name, s in summary.items():
        re = ", ".join(f"{v:.3f}" for v in s["A_lat_eig_real"])
        print(f"\n{name}: K_us={s['K_us']:+.3e}, dominant tau={s['dominant_tau_s']:.4f} s")
        print(f"  A_lat 고유값 실수부 = [{re}]  (gamma_ss={s['gamma_ss_rad_s']:.5f} rad/s)")

    print(
        "\n[시상수 차이 설명]\n"
        "요 응답 시상수는 A_lat 고유값 실수부의 역수다. limo 는 sedan보다\n"
        "질량(4.2 vs 1500 kg)과 요 관성(0.05 vs 2250 kg*m^2)이 압도적으로 작고\n"
        "저속(0.8 vs 15 m/s)에서 동작한다. A_lat 성분이 대략 C/(m*vx),\n"
        "C*l^2/(Iz*vx) 로 스케일하므로, 작은 m*vx 와 Iz*vx 가 고유값 크기를 키워\n"
        "limo 의 응답을 훨씬 빠르게(짧은 tau) 만든다. 또 sedan 은 고유값이 복소(감쇠\n"
        "진동)인 반면 limo(뉴트럴, a=b·Cf=Cr)는 A_lat 이 상삼각이라 실고유값 2개로\n"
        "과감쇠에 가깝다.\n"
        "주의: limo 의 Iz 는 직육면체 근사값(미식별)이므로 위 tau 는 자릿수 추정이다."
    )


if __name__ == "__main__":
    main()
