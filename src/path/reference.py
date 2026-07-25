"""참조 경로 — 호길이 s에 대한 곡률 프로파일 kappa(s).

경로는 (x,y) 점열이 아니라 kappa(s)로 정의한다. 오차좌표계 MPC 에는 이 형태가 필요하다
(.claude/rules/sim-experiment.md 경로 표현).

- kappa(s): 세그먼트 끝점 곡률을 선형 보간한 조각선형 함수 (클로소이드 = 선형 증가).
- vx_max(s) = sqrt(a_y_max/|kappa|), vx_range 로 clip. kappa < kappa_eps 는 직선 취급.
- get_preview(s, N, dt): 호길이 진행 ds = vx*dt 로 N개 프리뷰 생성. 끝에서 마지막 값 유지.
- reconstruct_xy: (x,y) 복원 — **시각화 전용.** 제어 루프에서 호출하지 마라.

내부 계산은 전부 rad·SI. deg 변환은 플롯 축에서만.
"""
from __future__ import annotations

import numpy as np

from src.config import PathConfig


class Reference:
    """kappa(s) 곡률 프로파일과 속도 프로파일, 프리뷰 제공자.

    vx_range 는 차량 config 에서 온다(경로 자체 속성이 아니므로 주입받는다).
    """

    def __init__(self, path_cfg: PathConfig, vx_range: tuple[float, float]):
        self.cfg = path_cfg
        self.vx_lo, self.vx_hi = float(vx_range[0]), float(vx_range[1])
        if not (self.vx_hi > self.vx_lo > 0.0):
            raise ValueError(f"vx_range 가 유효하지 않다: {vx_range}")

        # 세그먼트 경계 호길이 s_nodes 와 각 경계의 곡률 k_nodes.
        # k_nodes[0] = 0 (직선에서 출발), k_nodes[i] = segments[i-1].kappa_end.
        # 구간 내부는 이 노드들의 선형 보간(np.interp)이 곧 조각선형 kappa(s)다.
        s = 0.0
        s_nodes = [0.0]
        k_nodes = [0.0]
        for seg in path_cfg.segments:
            s += seg.length
            s_nodes.append(s)
            k_nodes.append(seg.kappa_end)
        self._s_nodes = np.asarray(s_nodes, dtype=float)
        self._k_nodes = np.asarray(k_nodes, dtype=float)
        self.total_length = float(s_nodes[-1])

    # ------------------------------------------------------------------ #
    # 곡률 · 속도 프로파일                                                 #
    # ------------------------------------------------------------------ #
    def kappa_of_s(self, s):
        """곡률 kappa(s) [1/m]. 스칼라/배열 모두 지원.

        s 를 [0, total_length] 로 clamp 한다 -> 경로 끝에서 마지막 곡률(직선=0) 유지.
        """
        s_c = np.clip(s, 0.0, self.total_length)
        return np.interp(s_c, self._s_nodes, self._k_nodes)

    def vx_max_of_s(self, s):
        """속도 프로파일 vx_max(s) [m/s]. 스칼라/배열 모두 지원.

        vx_max = sqrt(a_y_max / |kappa|), vx_range 로 clip.
        |kappa| < kappa_eps 는 직선 취급 -> vx_hi. 분모를 만들기 전에 분기해
        0-division 과 inf 전파를 피한다 (sim-experiment.md).
        """
        kappa = np.abs(self.kappa_of_s(s))
        straight = kappa < self.cfg.kappa_eps
        # 분모가 0에 가까운 곳은 sqrt 를 아예 계산하지 않는다 (1.0 대체값 사용).
        kappa_safe = np.where(straight, 1.0, kappa)
        vx = np.where(straight, self.vx_hi, np.sqrt(self.cfg.a_y_max / kappa_safe))
        return np.clip(vx, self.vx_lo, self.vx_hi)

    # ------------------------------------------------------------------ #
    # 프리뷰                                                              #
    # ------------------------------------------------------------------ #
    def get_preview(self, s: float, N: int, dt: float) -> tuple[np.ndarray, np.ndarray]:
        """현재 호길이 s 에서 지평 N 스텝의 (kappa[N], vx[N]) 프리뷰.

        호길이 진행은 ds = vx*dt (예측 속도로 전진). 경로 끝을 넘어가면 clamp 되어
        마지막 값(직선: kappa=0, vx=vx_hi)이 유지된다. 반환 배열 길이는 항상 N.
        """
        kappa = np.empty(N, dtype=float)
        vx = np.empty(N, dtype=float)
        s_cur = float(s)
        for i in range(N):
            k = float(self.kappa_of_s(s_cur))
            v = float(self.vx_max_of_s(s_cur))
            kappa[i] = k
            vx[i] = v
            s_cur += v * dt          # ds = vx*dt
        return kappa, vx

    # ------------------------------------------------------------------ #
    # (x, y) 복원 — 시각화 전용                                            #
    # ------------------------------------------------------------------ #
    def reconstruct_xy(self, ds: float = 0.05, x0: float = 0.0, y0: float = 0.0,
                       theta0: float = 0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """곡률 프로파일을 적분해 (x, y, theta) 경로를 복원한다. **시각화 전용.**

        dtheta/ds = kappa(s), dx/ds = cos(theta), dy/ds = sin(theta).
        초기 위치·헤딩에서 출발. ds 는 복원 해상도 [m].
        반환: (x[M], y[M], theta[M]), M = floor(total/ds)+1.
        제어 루프에서 호출하지 마라 (sim-experiment.md).
        """
        n = int(np.floor(self.total_length / ds)) + 1
        s = np.linspace(0.0, self.total_length, n)
        kappa = self.kappa_of_s(s)
        # theta(s) = theta0 + ∫ kappa ds  (사다리꼴 적분).
        theta = theta0 + np.concatenate([[0.0], np.cumsum(0.5 * (kappa[1:] + kappa[:-1]) * np.diff(s))])
        # x,y 도 사다리꼴 적분.
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        x = x0 + np.concatenate([[0.0], np.cumsum(0.5 * (cos_t[1:] + cos_t[:-1]) * np.diff(s))])
        y = y0 + np.concatenate([[0.0], np.cumsum(0.5 * (sin_t[1:] + sin_t[:-1]) * np.diff(s))])
        return x, y, theta
