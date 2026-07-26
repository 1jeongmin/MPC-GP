"""상태 측정 센서 모델 (Phase 8c).

Phase 8 까지 세 케이스는 모두 **참 상태**를 피드백받았다(이상적 센서). 그것은 비교를
'모델오차 보정 능력' 하나에 집중시키기 위한 통제였지, 현실적이라서가 아니었다.
이 모듈은 그 가정을 걷어내고 잡음 섞인 측정값을 제어 루프에 넣는다.

## 왜 별도 모듈인가

runner 에 케이스 분기를 넣지 않기 위해서다. 센서는 estimator 와 같은 방식으로
**주입**된다: 있으면 쓰고 없으면 참 상태를 그대로 쓴다. 따라서 이 파일이 없어도
기존 실험은 한 글자도 안 바뀌고 재현된다.

## 차량에 센서는 한 벌뿐이다

KF 는 원래 `simulate_measurement` 로 자기 잡음을 따로 뽑았다. 센서가 주입되면 그
경로를 쓰지 않고 **센서가 만든 같은 측정값**을 받는다. 이유 두 가지:
  1. 물리적으로 센서는 한 벌이다. 제어기와 추정기가 다른 잡음을 볼 수 없다.
  2. 세 케이스가 **같은 잡음 실현(realization)** 을 봐야 비교가 성립한다
     (sim-experiment.md 「통제해야 할 변수 — 잡음 seed」).

## 측정 모델

    y_k = x_k + n_k + b * sin(2*pi*f*t_k)
    n_k ~ N(0, diag(sigma^2))      # 백색잡음 (기본)
    b, f                            # 결정론적 사인 바이어스 (기본 0)

**사인 바이어스 기본값이 0 인 이유**: KF 는 측정잡음이 영평균 백색이라고 가정하고
설계된 추정기다. 결정론적 바이어스를 넣으면 그 가정이 깨져 KF 만 일방적으로
불리해진다 — `ekf-baseline.md` 가 금지하는 '비교군 약화'에 해당한다. 켜고 싶으면
그 사실을 리포트에 명시하고 R_kf 재설계까지 함께 해야 한다.
"""
from __future__ import annotations

import numpy as np

from src.config import SensorConfig


class StateSensor:
    """상태 4채널에 잡음(+선택적 사인 바이어스)을 실어 측정값을 만든다.

    재현성: rng 를 주입받아 쓴다. 같은 seed 면 케이스가 달라도 **같은 잡음 수열**이
    나온다 — 단, 스텝당 정확히 한 번만 호출되어야 한다(호출 횟수가 다르면 어긋난다).
    runner 는 스텝당 1회만 호출한다.
    """

    def __init__(self, cfg: SensorConfig, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng
        self._std = np.asarray(cfg.noise_std, float)
        self._amp = np.asarray(cfg.bias_amp, float)
        self._omega = 2.0 * np.pi * float(cfg.bias_freq_hz)
        self._has_bias = bool(np.any(self._amp > 0.0) and cfg.bias_freq_hz > 0.0)

    def measure(self, x_true: np.ndarray, t: float) -> np.ndarray:
        """참 상태 (4,) -> 측정값 (4,). t [s] 는 사인 바이어스 위상용."""
        x_true = np.asarray(x_true, float).reshape(4)
        y = x_true + self.rng.normal(0.0, 1.0, size=4) * self._std
        if self._has_bias:
            y = y + self._amp * np.sin(self._omega * t)
        return y


def make_sensor(cfg: SensorConfig | None,
                rng: np.random.Generator | None) -> StateSensor | None:
    """센서 config 가 있으면 센서를, 없으면 None(이상적 센서)을 반환한다."""
    if cfg is None:
        return None
    if rng is None:
        raise ValueError("센서를 쓰려면 rng 가 필요하다 (재현성).")
    return StateSensor(cfg, rng)
