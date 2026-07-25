---
paths:
  - "src/sim/**"
  - "src/eval/**"
  - "src/viz/**"
  - "src/path/**"
  - "scripts/**"
  - "configs/experiment/**"
  - "configs/path/**"
  - "configs/sim/**"
  - "results/**"
---

# 시뮬레이션·실험 프로토콜 룰

## 다중레이트 루프

- 플랜트 100 Hz (`dt_plant = 0.01`), 제어 50 Hz (`dt_ctrl = 0.02`)
- 제어 1스텝당 플랜트 2스텝, 그 사이 제어입력은 zero-order hold
- 플랜트 모델은 **주입 가능한 인자**로 받는다. 선형 ↔ 비선형 교체가 runner 코드
  수정 없이 되어야 한다.
- 플랜트는 고정밀 적분기, MPC 예측은 RK4(`dt_ctrl`). 이 차이는 의도된 것이다.

## 경로 표현

경로는 (x,y) 점열이 아니라 **호길이에 대한 곡률 프로파일 `kappa(s)`**로 정의한다.
- 계단형 `kappa` 금지. 전이 구간은 클로소이드(선형 증가) 등 부드러운 함수로.
- `vx_max(s) = sqrt(a_y_max / |kappa(s)|)`, `vx_range`로 clip. `a_y_max`는 config.
  **`kappa = 0`(직선)에서 0-division이 난다.** `inf`를 경유시키지 말고
  `np.where(|kappa| < kappa_eps, vx_range[1], sqrt(...))` 형태로 **분모를 만들기 전에**
  분기하라. `kappa_eps`는 config에서 읽는다. `inf`를 clip에 흘려보내면 값은 맞아도
  경고가 뜨고, CasADi 경로에서는 NaN으로 전파될 수 있다.
- `get_preview(s, N, dt) -> (kappa[N], vx[N])`. 호길이 진행은 `ds = vx*dt`.
- (x,y) 복원 함수는 **시각화 전용**이다. 제어 루프에서 호출하지 마라.

## 케이스 정의는 config로만

Part 1(`mpc_only` / `mpc_ekf` / `mpc_gp`)과 Part 2(`offline` / `online` /
`online_sparse` / `sliding_window`)는 **`configs/experiment/*.yaml`로만 구분**한다.
스크립트 안에 케이스 분기를 하드코딩하지 마라.

`configs/experiment/*.yaml`는 **조합 파일**이다. 값을 직접 적지 말고 그룹
(`vehicle/`, `path/`, `mpc/`, `gp/`, `ekf/`, `sim/`)을 참조해 조립한다.
같은 값이 두 experiment 파일에 복붙되어 있으면 잘못된 것이다 — 공통 그룹으로 빼라.
케이스 간 통제변수가 실제로 동일한지는 **조립된 config를 diff**해서 확인한다.

## 통제해야 할 변수 (케이스 간 반드시 동일)

경로 형상 · 초기조건 · `v_x` 프로파일 · MPC 지평 `N` · MPC 가중치 ·
IPOPT 옵션 · 플랜트 모델과 적분기 설정 · 잡음 seed.

이 중 하나라도 다르게 돌려야 한다면 그 사실을 결과에 명시적으로 기록하라.

## 재현성 (필수)

매 실행마다 `results/<run_id>/`를 만들고 다음을 저장한다:
- 사용한 config 전체 스냅샷 (참조가 아니라 복사본)
- git commit hash, dirty 여부
- 난수 seed 전부
- 라이브러리 버전 (`pip freeze` 또는 최소한 casadi/numpy/gpytorch 버전)

이 중 하나라도 없으면 결과를 생성하지 마라.

## 로깅 스키마 (제어 스텝마다)

```
t, x(4), delta, vx, kappa, s
solve_time, ipopt_iter, converged
residual r(4)                      # 4채널 전부. 3,4번은 진단용
gp_mean(2), gp_var(2)              # GP 케이스
ekf_P_diag(4), ekf_innovation      # EKF 케이스
```

`results/`에는 CSV 또는 npz로 저장한다.

## 지표

- 추종: RMS `e_y`, max `|e_y|`, RMS `e_psi`, 정착 거동
- 제어 노력: RMS `delta`, RMS `delta_rate`, 제약 접촉 비율
- 실시간성: solve time mean / p95 / max, `dt_ctrl` 초과율, 수렴 실패율
- 모델: 잔차 RMS (채널별), noise floor 대비 배율
- UQ 캘리브레이션: 표준화 잔차 분포, 예측구간 커버리지(예: 95% 구간의 실제 포함률),
  가능하면 NLPD. **Part 1의 주 지표는 추종오차가 아니라 이 캘리브레이션이다.**

## 검증 연쇄 (건너뛰지 마라)

1. Phase 1에서 측정한 RK4 vs 고정밀 적분기 차이 = 잔차의 noise floor
2. 모델 완전일치 케이스의 잔차는 이 noise floor와 **같은 자릿수**여야 한다
3. 비선형 플랜트로 교체한 뒤 잔차가 noise floor의 몇 배인지 정량화
4. 이 배율이 GP가 학습할 신호의 SNR이다. 숫자로 보고하라

2번이 어긋나면 GP 작업을 진행하지 말고 원인부터 찾아라.

## 플롯 규약

- 내부 계산은 전부 rad. **deg 변환은 플롯 축에서만.**
- 필수 그림: 복원 경로 위 실제 궤적 / `e_y`·`e_psi` vs t / `delta` vs t(제약선 표시) /
  `v_y`·`gamma` vs t / solve time 히스토그램(`dt_ctrl` 기준선) / 잔차 4채널 vs t /
  GP 사후 표준편차 지도
- 케이스 비교 그림은 색·선종을 케이스별로 고정한다(config에 매핑 저장).
- 그림 파일명에 `run_id`를 포함한다.
