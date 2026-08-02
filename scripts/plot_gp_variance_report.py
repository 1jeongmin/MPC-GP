"""GP 분산 영역 그림 일괄 생성 — 폐루프 로그에서 바로 그린다.

    python scripts/plot_gp_variance_report.py [out_dir]

## 무엇을 그리나

시나리오마다 두 장:

1. **예측구간 밴드** (`*_band.png`) — 시간축 위에 GP 평균 ± 95% 예측구간(음영)과
   실측 잔차를 겹쳐 그린다. 음영 밖으로 점이 자주 나가면 과신, 음영이 쓸데없이
   넓으면 과소신뢰. **`Var[y*] = Var[f*] + sigma_n^2`(관측 예측분산)** 을 쓴다 —
   비교 대상이 관측된 잔차이기 때문이다(2026-07-31 진단).
2. **불확실성 분해** (`*_split.png`) — 같은 구간에서 `Var[f*]`(epistemic, "여기
   데이터가 있었나")와 `sigma_n^2`(aleatoric, "원래 튀는 양")을 **나눠서** 보여준다.
   둘을 뭉쳐 그리면 "낯선 곳이라 모르는 것"과 "원래 잡음이 큰 곳"을 구별할 수 없다.

기존 `make_uncertainty_map_figure`(상태공간 지도)와 역할이 다르다 — 그쪽은 (v_y,gamma)
평면 위의 지도이고, 이쪽은 **시간축에서 실측과 대조**하는 그림이다.
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.calibration import calibration_metrics
from src.viz.plots import make_gp_calibration_band_figure

# (실행 이름, 그림 제목) — 4케이스 + 주요 변형.
# ★ 그림 라벨은 **영문**으로 쓴다 — matplotlib 기본 폰트(DejaVu Sans)에 한글 글리프가
#   없어 깨진다. 저장소의 다른 그림도 전부 영문 라벨이다(폰트 의존성 회피).
SCENARIOS = [
    ("part1_gp_lpf_ay4", "case1 lpf_ay4 (in-distribution)"),
    ("part1_gp_lpf_ay6", "case2 lpf_ay6 (intensity extrap.)"),
    ("part1_gp_rtf_ay4", "case3 rtf_ay4 (shape extrap.)"),
    ("part1_gp_rtf_ay6", "case4 rtf_ay6 (shape+intensity extrap.)"),
    ("part1_gp_rtf_ay6_warp", "rtf_ay6 + input warping"),
    ("part1_gp_rtf_ay6_lag2", "rtf_ay6 + state history lag2"),
    ("part1_gp_lpf_ay4_het", "lpf_ay4 + heteroscedastic noise"),
]


def latest_log(tag: str):
    g = sorted(glob.glob(str(ROOT / "results" / f"{tag}_2026*" / "log.npz")))
    return np.load(g[-1]) if g else None


def split_figure(L, out_dir: Path, tag: str, title: str) -> Path:
    """Var[f*](epistemic) 와 sigma_n^2(aleatoric) 를 나눠 그린다."""
    t = L["t"]
    vf = L["gp_var"]                                   # 잠재 = epistemic
    vo = L["gp_var_obs"] if "gp_var_obs" in L.files else vf
    vn = np.maximum(vo - vf, 0.0)                      # 관측 - 잠재 = sigma_n^2
    err = np.abs(L["residual"][:, 0:2] - L["gp_mean"])

    stride = max(1, len(t) // 4000)
    s = slice(None, None, stride)
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.6), constrained_layout=True)
    for j, (nm, unit) in enumerate([("v_y", "m/s"), ("gamma", "rad/s")]):
        a = ax[j]
        a.fill_between(t[s], 0, np.sqrt(vf[s, j]), color="#d62728", alpha=0.45,
                       label=r"$\sqrt{\mathrm{Var}[f^*]}$  epistemic (sparse data)")
        a.fill_between(t[s], np.sqrt(vf[s, j]), np.sqrt(vf[s, j] + vn[s, j]),
                       color="#1f77b4", alpha=0.35,
                       label=r"$\sigma_n$  aleatoric (irreducible noise)")
        a.plot(t[s], err[s, j], "k.", ms=1.5, alpha=0.35, label=r"actual $|r-\mu|$")
        a.set_xlabel("t [s]"); a.set_ylabel(f"{nm} residual [{unit}]")
        frac = float(np.mean(vf[:, j] / np.maximum(vo[:, j], 1e-300)))
        a.set_title(f"{nm}: epistemic share {frac:.1%}")
        a.legend(fontsize=8, loc="upper right"); a.grid(alpha=0.3)
    fig.suptitle(f"uncertainty decomposition — {title}", fontsize=12)
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"{tag}_split.png"
    fig.savefig(p, dpi=130); plt.close(fig)
    return p


def main() -> None:
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results" / "figures_gp_variance"
    made = []
    print(f"{'시나리오':<26} {'z_std':>7} {'95%커버':>8} {'epistemic 비중':>14}")
    print("-" * 60)
    for tag, title in SCENARIOS:
        L = latest_log(tag)
        if L is None or "gp_mean" not in L.files:
            print(f"{tag:<26} (로그 없음)"); continue
        r = L["residual"][:, 0:2]; mu = L["gp_mean"]
        var = L["gp_var_obs"] if "gp_var_obs" in L.files else L["gp_var"]
        made.append(make_gp_calibration_band_figure(
            L["t"], r, mu, var, out_dir, tag, tag=title))
        made.append(split_figure(L, out_dir, tag, title))
        p = calibration_metrics(r, mu, var)["pooled"]
        ep = float(np.mean(L["gp_var"] / np.maximum(var, 1e-300)))
        print(f"{tag:<26} {p['z_std']:>7.3f} {p['coverage'][0.95]:>7.1%} {ep:>13.1%}")
    print(f"\n[저장] {out_dir}  ({len(made)}장)")
    print("※ epistemic 비중 = Var[f*]/Var[y*]. 낮으면 '모르는 것'이 아니라 '원래 잡음'이")
    print("  불확실성의 대부분이라는 뜻 — OOD 에서 이 값이 안 오르면 GP 가 낯섦을 못 느낀 것이다.")


if __name__ == "__main__":
    main()
