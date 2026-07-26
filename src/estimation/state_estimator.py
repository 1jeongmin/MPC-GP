"""공통 상태추정기 — 측정 잡음 필터링 전용 (Phase 8d).

## 역할 구분이 이 파일의 존재 이유다

`ekf.py` 의 `AugmentedKF` 는 **모델 보정자**다 — 외란 `d` 를 추정해 MPC 예측 모델을
고치는 것이 목적이고, `d_hat` 만 제어에 쓰인다(`mpc_kf.py`). 반면 이 모듈은
**상태추정기**다 — 잡음 낀 측정에서 상태를 복원해 MPC 피드백에 넣는 것이 전부이며
모델 보정에는 관여하지 않는다. `ekf-baseline.md` 「역할 혼동 금지」의 두 역할을
코드 배치로도 분리한다.

## 세 케이스가 **모두** 이걸 쓴다

Phase 8c 까지는 세 케이스 모두 생측정값을 그대로 피드백받았다. 그 결과 조향 변화율이
전체 시간의 ~47% 를 액추에이터 속도한계에 붙어 있었다(측정됨). 현실 차량은 측정을
그대로 제어기에 넣지 않으므로, 여기서는 **공통 상태추정기로 한 번 거른 x_hat** 을
세 케이스에 동일하게 준다. 상태추정 품질이 통제변수로 고정되고, 케이스 간 차이는
여전히 **모델 보정 방식 하나**로 남는다.

## 왜 외란증강이 아니라 명목 4상태인가 (사용자 확정)

외란증강 필터를 공통 추정기로 쓰면 `mpc_only` 기준선조차 내부에 외란관측기를 갖게 되어
"보정 없음" 이라는 정의가 흐려진다. 순수 명목 4상태 KF 를 쓰면 기준선이 외란 정보를
어디서도 받지 않는다. 전상태 측정이라 모델 불일치가 있어도 update 단계에서 보정되므로
편향이 제한적이다(그래도 존재한다 — §알려진 한계).

필터 코어는 `AugmentedKF` 를 그대로 재사용한다. 그 클래스는 평균함수·야코비안을
**주입**받는 범용 구현이므로(n_state 인자), 4상태 명목 모델을 넣으면 그대로 순수
상태추정 KF 가 된다. 코드를 복제하지 않는다.
"""
from __future__ import annotations

import casadi as ca
import numpy as np

from src.config import StateKfConfig, VehicleConfig
from src.estimation.ekf import AugmentedKF
from src.models.linear_bicycle import dynamics_ca

N_STATE = 4
N_MEAS = 4


def full_state_measurement_ca() -> ca.Function:
    """측정 h(x4) = x4 (전상태 직접 관측). 야코비안은 상수 H = I4."""
    x = ca.SX.sym("x", N_STATE)
    return ca.Function("h_state", [x], [x])


class StateKF(AugmentedKF):
    """명목 4상태 순수 칼만필터 — 측정 필터링만 한다.

    `AugmentedKF` 의 필터 코어(predict/update)를 그대로 쓰되 상태가 4개다.
    외란 블록이 없으므로 `d_hat` / `P_d_block` 은 정의되지 않는다 — 호출하면
    조용히 잘못된 값을 주는 대신 명시적으로 막는다.
    """

    @property
    def d_hat(self) -> np.ndarray:
        raise AttributeError(
            "StateKF 는 상태추정기다. 외란 추정(d_hat)은 모델 보정자인 "
            "AugmentedKF(ekf.py)의 역할이다 — 역할을 섞지 마라."
        )

    @property
    def P_d_block(self) -> np.ndarray:
        raise AttributeError("StateKF 에는 외란 블록이 없다 (순수 4상태).")

    def filter_step(self, y: np.ndarray, u_prev: float, vx_prev: float,
                    kappa_prev: float, first: bool) -> np.ndarray:
        """측정 y(4,) 를 받아 필터링된 상태 추정 x_hat(4,) 을 반환한다.

        first=True 인 첫 스텝은 predict 를 건너뛴다(직전 입력이 없다).
        `AugmentedKF` 와 같은 predict->update 순서를 유지한다.
        """
        if not first:
            self.predict(u_prev, vx_prev, kappa_prev)
        self.update(np.asarray(y, float).reshape(N_MEAS))
        return self.x_hat.copy()


def make_state_kf(vehicle: VehicleConfig, cfg: StateKfConfig, dt: float,
                  x0: np.ndarray | None = None) -> StateKF:
    """명목 4상태 상태추정 KF 를 조립한다.

    프로세스 모델 = 명목 선형 자전거(`dynamics_ca`) — MPC 예측·잔차 정의와 같은 명목
    모델이다. 복사하지 않고 같은 함수를 호출한다.
    """
    process = dynamics_ca(vehicle)
    meas = full_state_measurement_ca()
    Q = np.diag(cfg.Q_diag)
    R = np.diag(cfg.R_diag)
    P0 = np.diag(cfg.P0_diag)
    x0 = np.zeros(N_STATE) if x0 is None else np.asarray(x0, float).reshape(N_STATE)
    return StateKF(process, meas, dt, Q, R, P0, x0,
                   n_state=N_STATE, n_meas=N_MEAS)
