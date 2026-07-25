---
paths:
  - "src/models/**"
  - "configs/vehicle/**"
  - "tests/test_model*.py"
  - "tests/test_integrator*.py"
---

# 차량 모델 룰

## 확정 정식화 — 변경 금지

상태 `x = [v_y, gamma, e_psi, e_y]^T`, 입력 `u = delta`,
외생 파라미터 `v_x`, `kappa`.

```
xdot = A(v_x) x + B(v_x) u + E(v_x) kappa

A(v_x) =
[ -(Cf+Cr)/(m*vx),      (b*Cr-a*Cf)/(m*vx) - vx,   0,   0 ]
[ (b*Cr-a*Cf)/(Iz*vx), -(a^2*Cf+b^2*Cr)/(Iz*vx),   0,   0 ]
[ 0,                     1,                        0,   0 ]
[ 1,                     0,                       vx,   0 ]

B(v_x) = [ Cf/m, a*Cf/Iz, 0, 0 ]^T
E(v_x) = [ 0, 0, -vx, 0 ]^T
```

## 부호 규약 — 이 정의에서 벗어나지 마라

```
alpha_f = delta - (v_y + a*gamma)/v_x
alpha_r = -(v_y - b*gamma)/v_x
F_yf = Cf * alpha_f,   F_yr = Cr * alpha_r      # Cf, Cr = 축당 코너링 강성, 양수
m*(v_y_dot + v_x*gamma) = F_yf + F_yr
Iz*gamma_dot = a*F_yf - b*F_yr
```

오차 기구학:
```
e_psi_dot = gamma - v_x*kappa
e_y_dot   = v_y + v_x*e_psi
```

위 A/B/E는 이 규약에서 대수적으로 유도되는 결과와 일치함이 확인되었다.
행렬 성분을 바꾸고 싶다면 **먼저 유도를 다시 보이고 사용자에게 확인받아라.**

## 구조 원칙

1. **`v_x`, `kappa`는 절대 상태(결정변수)가 아니다.** 파라미터로만 들어간다.
   상태로 승격하면 `v_x*gamma` 항이 쌍선형이 되어 LTV 구조가 깨진다.
2. **오차 기구학(3, 4행)은 어떤 경우에도 수정하지 않는다.** 비선형 플랜트로
   교체할 때도 마찬가지다. 잔차의 출처를 타이어 비선형성 하나로 통제해야 한다.
3. 비선형 플랜트(`nonlinear_bicycle.py`)는 **타이어 모델만** 교체한다
   (Fiala 또는 단순화 Pacejka). 나머지 구조는 선형 버전과 동일하게 둔다.

## 이중 경로 구현 의무

모든 모델 함수는 두 경로를 **모두** 제공한다.

- numpy: `A_matrix(cfg, vx)`, `B_matrix(cfg, vx)`, `E_matrix(cfg, vx) -> np.ndarray`
- CasADi: 심볼릭 `x, u, vx, kappa`를 받아 `xdot`을 반환하는 `casadi.Function`

두 경로는 같은 입력에 같은 출력을 내야 한다(1e-10 이내). 이 대조 테스트 없이
새 모델을 추가하지 마라.

## 해석 검증 함수 (필수)

```
L = a + b
K_us(cfg)                    = m*(b*Cr - a*Cf) / (L*Cf*Cr)
steady_state_yaw_rate(...)   = delta*vx / (L + K_us*vx^2)
```

- `K_us`는 위 정의(= `m/L * (b/Cf - a/Cr)`)를 쓴다. 단위는 rad·s²/m 계열이며,
  중력가속도 `g`로 나눈 rad/g 규약과 **혼용 금지**. docstring에 규약을 명시하라.
- `K_us > 0` → 언더스티어, `= 0` → 뉴트럴, `< 0` → 오버스티어.

## 안정성 — 4x4 전체 A에 Hurwitz 판정을 하지 마라

`A(v_x)`는 우측상단 블록이 0이므로 **블록 하삼각**이고,
`eig(A) = eig(A_lat) ∪ eig(A_22)` 이다. 여기서

```
A_lat = A[0:2, 0:2]        # 횡동역학 (v_y, gamma)
A_22  = [[0, 0], [vx, 0]]  # 오차 기구학 → 고유값 0 (중근)
```

**원점 고유값 2개는 정상이다.** `e_psi`, `e_y`가 순수 적분기이기 때문이며,
개루프 4x4가 Hurwitz로 나오면 오히려 모델이 틀린 것이다.
안정성 판정은 반드시 `A_lat`에 대해서만 한다.

`A_lat`의 특성:
```
trace(A_lat) < 0                                          # 항상
det(A_lat)   = Cf*Cr*L*(L + K_us*vx^2) / (m*Iz*vx^2)
```
(유도 중 `(Cf+Cr)(a^2*Cf + b^2*Cr) - (bCr - aCf)^2 = Cf*Cr*L^2` 로 정리된다.)

따라서 **`A_lat` Hurwitz ⟺ `L + K_us*vx^2 > 0`**.

- `K_us >= 0` (언더/뉴트럴): 모든 `v_x`에서 안정
- `K_us < 0` (오버스티어): `v_x < v_crit = sqrt(-L/K_us)` 에서만 안정

이 판별식은 `steady_state_yaw_rate`의 분모와 **같은 식**이다. 두 검증 항목이
같은 양을 가리키므로 서로 교차검증이 되며, 오버스티어에서 임계속도를 넘으면
요레이트 해석해가 발산하는 이유도 여기서 설명된다.
`critical_speed(cfg)`를 구현하고, `K_us >= 0`이면 `inf`를 반환하라.

## 특이점

`v_x -> 0`에서 A 행렬이 발산한다. config 유효성 검사에서 `vx_range` 하한이
0보다 크도록 강제하고, 런타임에도 `v_x`가 하한 미만이면 예외를 던져라.
LIMO는 `vx_range: [0.2, 1.0]`으로 이 하한이 특히 빡빡하다.

## 적분기

- `rk4_step(f, x, u, params, dt)` — CasADi 심볼릭 호환. **MPC 예측 전용.**
- `plant_step(...)` — 플랜트 전용 고정밀 경로. `scipy.integrate.solve_ivp`
  (rtol=1e-10, atol=1e-12) 또는 CasADi CVODES. **어느 쪽을 왜 골랐는지 주석으로 남겨라.**
- 플랜트와 MPC 예측은 **의도적으로 다른 적분기**를 쓴다. 이 차이가 잔차의
  noise floor를 정의하며, 그 값은 반드시 숫자로 측정·보고되어야 한다.
- RK4 차수를 올리거나 내리기 전에 사용자에게 물어라. GP 결합 시 스텝당 우변
  평가 횟수가 그대로 solve time에 곱해진다.

## config 취급

LIMO 파라미터(`m, Iz, a, b, Cf, Cr, delta_max`)는 전부 미검증 추정치다.
YAML의 `TODO` 주석을 지우지 마라. LIMO 결과를 보고할 때는 파라미터가
식별되지 않았다는 사실을 항상 함께 적어라.
