"""증강상태 칼만필터 — Part 1 비교군 (.claude/rules/ekf-baseline.md).

**명명 정직성**: 이 추정기의 프로세스 모델(명목 선형 자전거 + 외란)과 측정 모델
(상태 직접 관측)이 **둘 다 선형**이므로, 예측·측정 야코비안이 상수다. 따라서
엄밀히는 EKF가 아니라 **외란증강 LTV 칼만필터**다. 리포트에도 그렇게 쓴다.
(선형계에서 KF는 최적 추정기이므로 비교군을 약화하는 게 아니라 최강 형태를 준다.)

**역할**: KF는 상태추정기, GP는 모델 보정자다. Part 1은 "모델 오차와 그 불확실성을
다루는 두 접근"의 비교다 — 이 프레이밍을 흐리지 마라.

**전방호환**: 필터 코어(예측/갱신 공분산)는 평균함수·야코비안을 **주입**받는다.
선형 케이스는 야코비안이 상수인 특수경우일 뿐이다. 야코비안은 CasADi ca.jacobian
으로 자동 생성한다(수기 미분 금지). 실차(비선형 측정)로 갈 때 process/meas 함수만
바꾸면 되고 필터 수식은 안 건드린다.

증강상태 = [v_y, gamma, e_psi, e_y, d_vy, d_gamma] (6).
측정      = [v_y, gamma, e_psi, e_y] (4, 전상태).

**측정 모델 결정 (2026-07-25, Phase 6 진단으로 수정)**: 처음엔 현실적 부분관측
(v_y 미측정)으로 뒀으나, 그 경우 외란 d_vy 가 e_y 로부터 이중적분 뒤에야 관측되어
심하게 지연·과대추정되고 MPC+KF 가 MPC-only 보다 나빠졌다. 논문의 초점은 상태추정이
아니라 **모델오차 처리**이므로, 세 방법 모두 상태 정보를 갖게 하고 KF 는 깨끗한 외란
관측기가 되도록 **전상태 측정**으로 바꿨다. 이러면 MPC+KF 가 MPC-only 를 이긴다(+~7%).
(실차 단계에서 v_y 미측정 + a_y 측정으로 관측성을 회복하는 것은 별도 확장.)
"""
from __future__ import annotations

import casadi as ca
import numpy as np

from src.config import EkfConfig, VehicleConfig
from src.models.integrators import rk4_step
from src.models.linear_bicycle import dynamics_ca

# 측정 채널이 증강상태에서 뽑는 인덱스 (v_y, gamma, e_psi, e_y = 전상태).
MEAS_ROWS = (0, 1, 2, 3)


def augmented_process_ca(vehicle: VehicleConfig) -> ca.Function:
    """증강 연속 우변 f_aug(x6, u, vx, kappa) -> xdot6.

    명목 4상태 우변을 재사용하고, 외란 d 를 v_y, gamma 행에 더한다. d 동역학은 0
    (random walk, Q_kf 로 구동). 명목 우변은 복사하지 않고 dynamics_ca 를 호출한다.
    """
    f_nom = dynamics_ca(vehicle)
    x = ca.SX.sym("x", 6)
    u = ca.SX.sym("u")
    vx = ca.SX.sym("vx")
    kappa = ca.SX.sym("kappa")
    xd4 = f_nom(x[0:4], u, vx, kappa)              # 명목 4상태
    xd4 = xd4 + ca.vertcat(x[4], x[5], 0, 0)       # d_vy, d_gamma 주입
    xd = ca.vertcat(xd4, 0.0, 0.0)                 # d_dot = 0
    return ca.Function("f_aug", [x, u, vx, kappa], [xd])


def linear_measurement_ca() -> ca.Function:
    """측정 h(x6) = 전상태 [v_y, gamma, e_psi, e_y] (선형). 야코비안은 상수 H."""
    x = ca.SX.sym("x", 6)
    y = ca.vertcat(*[x[r] for r in MEAS_ROWS])
    return ca.Function("h", [x], [y])


class AugmentedKF:
    """외란증강 (LTV) 칼만필터. 평균함수·야코비안 주입으로 EKF 전방호환.

    process_fn: 연속 우변 f(x,u,vx,kappa)->xdot (CasADi Function).
    meas_fn: 측정 h(x)->y (CasADi Function).
    선형 케이스는 이 둘이 선형이라 야코비안 상수 -> KF. 비선형이면 -> EKF (코어 동일).
    """

    def __init__(self, process_fn: ca.Function, meas_fn: ca.Function, dt: float,
                 Q_cont: np.ndarray, R: np.ndarray, P0: np.ndarray, x0: np.ndarray,
                 n_state: int = 6, n_meas: int = 4):
        self.dt = float(dt)
        self.n_state = n_state
        self.n_meas = n_meas
        self.Q_cont = np.asarray(Q_cont, float)   # 연속 (predict 에서 *dt)
        self.R = np.asarray(R, float)
        self.P = np.asarray(P0, float).copy()
        self.x_hat = np.asarray(x0, float).reshape(n_state).copy()
        self.innovation = np.zeros(n_meas)

        # 이산 전이 + 야코비안 (rk4 + ca.jacobian). 명목이 선형이면 F 는 상수.
        x = ca.SX.sym("x", n_state)
        u = ca.SX.sym("u")
        vx = ca.SX.sym("vx")
        kappa = ca.SX.sym("kappa")
        x_next = rk4_step(process_fn, x, u, (vx, kappa), self.dt)
        self._F_step = ca.Function("Fstep", [x, u, vx, kappa], [x_next])
        self._F_jac = ca.Function("Fjac", [x, u, vx, kappa], [ca.jacobian(x_next, x)])
        y = meas_fn(x)
        self._h = ca.Function("h", [x], [y])
        self._H_jac = ca.Function("Hjac", [x], [ca.jacobian(y, x)])

    @property
    def d_hat(self) -> np.ndarray:
        """외란 추정 [d_vy, d_gamma] (2,) — GP mean 에 대응하는 모델 보정."""
        return self.x_hat[4:6].copy()

    @property
    def P_diag(self) -> np.ndarray:
        """공분산 대각 (6,). d-블록(4:6)이 모델오차 불확실성."""
        return np.diag(self.P).copy()

    @property
    def P_d_block(self) -> np.ndarray:
        """외란 공분산 부분행렬 P[4:6, 4:6] (2,2).

        대각만이 아니라 **전체 블록**이 필요하다: Phase 8 캘리브레이션에서 이 블록을
        이산 잔차 공간으로 G P_d G^T 로 환산하는데(d 는 연속 rate, 잔차는 이산 상태차이),
        G 가 비대각이면 대각만으로는 환산이 틀린다.
        """
        return self.P[4:6, 4:6].copy()

    def predict(self, u: float, vx: float, kappa: float) -> None:
        """예측 스텝: x <- F(x,u), P <- F P F^T + Q*dt."""
        F = np.array(self._F_jac(self.x_hat, u, vx, kappa))
        self.x_hat = np.array(self._F_step(self.x_hat, u, vx, kappa)).reshape(self.n_state)
        self.P = F @ self.P @ F.T + self.Q_cont * self.dt
        self.P = 0.5 * (self.P + self.P.T)   # 대칭 유지

    def update(self, y: np.ndarray) -> None:
        """갱신 스텝: 측정 y 로 보정. innovation 저장."""
        y = np.asarray(y, float).reshape(self.n_meas)
        H = np.array(self._H_jac(self.x_hat))
        y_pred = np.array(self._h(self.x_hat)).reshape(self.n_meas)
        self.innovation = y - y_pred
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x_hat = self.x_hat + K @ self.innovation
        I = np.eye(self.n_state)
        self.P = (I - K @ H) @ self.P
        self.P = 0.5 * (self.P + self.P.T)

    def select_measurement(self, y_full4: np.ndarray) -> np.ndarray:
        """이미 만들어진 측정 상태(4,)에서 이 필터의 측정 채널만 뽑는다.

        외부 센서(src/sim/sensor.py)가 주입된 경우 쓰는 경로다. 차량에 센서는 한 벌이므로
        제어기와 필터가 **같은 측정값**을 봐야 한다 — 여기서 잡음을 새로 만들지 않는다.
        (센서가 없을 때만 simulate_measurement 로 자체 잡음을 만든다.)
        """
        y = np.asarray(y_full4, float).reshape(4)
        return np.array([y[r] for r in MEAS_ROWS])

    def simulate_measurement(self, x_true4: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """참 플랜트 상태(4,)에서 잡음 섞인 측정 y를 생성한다.

        측정 채널은 물리 4상태에서 MEAS_ROWS 로 뽑는다 (전상태 = 4채널).
        """
        x_true4 = np.asarray(x_true4, float).reshape(4)
        clean = np.array([x_true4[r] for r in MEAS_ROWS])
        noise = rng.multivariate_normal(np.zeros(self.n_meas), self.R)
        return clean + noise

    def is_linear(self, tol: float = 1e-9) -> bool:
        """야코비안이 상태에 무관(상수)인지 확인 — KF≡EKF 검증용."""
        rng = np.random.default_rng(0)
        F0 = np.array(self._F_jac(np.zeros(self.n_state), 0.1, 15.0, 0.0))
        H0 = np.array(self._H_jac(np.zeros(self.n_state)))
        for _ in range(5):
            x = rng.normal(size=self.n_state)
            F = np.array(self._F_jac(x, 0.1, 15.0, 0.0))
            H = np.array(self._H_jac(x))
            if np.max(np.abs(F - F0)) > tol or np.max(np.abs(H - H0)) > tol:
                return False
        return True


def make_augmented_kf(vehicle: VehicleConfig, ekf_cfg: EkfConfig, dt: float,
                      x0: np.ndarray | None = None) -> AugmentedKF:
    """선형 명목 모델로 증강 KF 를 조립한다. 실차 전환 시 이 팩토리만 교체."""
    process = augmented_process_ca(vehicle)
    meas = linear_measurement_ca()
    Q = np.diag(ekf_cfg.Q_kf_diag)
    R = np.diag(ekf_cfg.R_kf_diag)
    P0 = np.diag(ekf_cfg.P0_diag)
    x0 = np.zeros(6) if x0 is None else np.asarray(x0, float).reshape(6)
    return AugmentedKF(process, meas, dt, Q, R, P0, x0, n_state=6, n_meas=len(MEAS_ROWS))
