# MPC-GP — GP 잔차 보정 기반 차량 횡방향 경로추종 MPC

명목 **선형 동역학 자전거 모델**(오차좌표계) 위에 **모델 오차 보정**을 얹은 횡방향
경로추종 MPC를 Python + CasADi로 구현하고 시뮬레이션으로 검증한다.

보정 방식 세 가지를 **완전히 동일한 조건**에서 비교한다.

| 케이스 | 보정 방식 | 불확실성의 정체 |
|---|---|---|
| `mpc_only` | 없음 (기준선) | — |
| `mpc_kf` | 외란증강 칼만필터가 외란 `d`를 **시간축**으로 추정 | `P` 대각 (aleatoric) |
| `mpc_gp` | GP가 상태→잔차 사상을 **상태공간**에서 학습 | 사후분산 (epistemic) |

핵심 논점은 "GP가 추종을 더 잘한다"가 아니라
**"GP의 불확실성은 상태공간 지도이고, KF의 것은 시간축 스칼라다"** 이다.

---

## 1. 지시받은 모델링 순서와 진행 현황

주신 4단계에 대한 대응이다. **1~4번 모두 완료**했고, 4번 이후로 추가 작업이 상당히
진행됐다(§2). 3번은 Python/MATLAB 두 선택지 중 **Python을 채택**해 구현했다(사유는 아래).

### 1번 — State equation formulation ✅

$$x = [v_y,\ \gamma,\ e_\psi,\ e_y]^\top,\qquad u = \delta,\qquad v_x,\ \kappa:\ \text{외생 신호}$$

레퍼런스를 그대로 쓰지 않고 **부호 규약부터 직접 유도**해 고정했다.

슬립각·타이어 힘:

$$\alpha_f = \delta - \frac{v_y + a\gamma}{v_x},\qquad \alpha_r = -\frac{v_y - b\gamma}{v_x}$$

$$F_{yf} = C_f\alpha_f,\qquad F_{yr} = C_r\alpha_r \qquad (C_f, C_r > 0)$$

뉴턴–오일러 + 오차 기구학:

$$m(\dot v_y + v_x\gamma) = F_{yf} + F_{yr},\qquad I_z\dot\gamma = aF_{yf} - bF_{yr}$$

$$\dot e_\psi = \gamma - v_x\kappa,\qquad \dot e_y = v_y + v_x e_\psi$$

정리하면 $\dot x = A(v_x)x + B(v_x)\delta + E(v_x)\kappa$:

$$A = \begin{bmatrix}
-\frac{C_f+C_r}{mv_x} & \frac{bC_r-aC_f}{mv_x}-v_x & 0 & 0\\
\frac{bC_r-aC_f}{I_zv_x} & -\frac{a^2C_f+b^2C_r}{I_zv_x} & 0 & 0\\
0 & 1 & 0 & 0\\
1 & 0 & v_x & 0
\end{bmatrix},\quad
B=\begin{bmatrix}C_f/m\\ aC_f/I_z\\0\\0\end{bmatrix},\quad
E=\begin{bmatrix}0\\0\\-v_x\\0\end{bmatrix}$$

**구현**: [`src/models/linear_bicycle.py`](src/models/linear_bicycle.py) — numpy / CasADi 이중 경로.
**규약 원본**: [`.claude/rules/vehicle-model.md`](.claude/rules/vehicle-model.md)

핵심 설계 결정 두 가지:

- **`v_x`, `κ`는 상태로 승격하지 않는다.** 파라미터로만 진입시켜야 LTV 구조가 유지되고,
  나중에 QP 전환 가능성을 논할 수 있다. 상태로 올리면 $v_x\gamma$ 항이 쌍선형이 된다.
- **오차 기구학(3·4행)은 어떤 경우에도 수정하지 않는다.** 비선형 플랜트로 바꿀 때도
  마찬가지다 — 잔차의 출처를 타이어 비선형성 하나로 통제하기 위한 실험 설계다.

검증용 해석해:

$$K_{us} = \frac{m(bC_r-aC_f)}{LC_fC_r},\qquad
\gamma_{ss} = \frac{\delta v_x}{L+K_{us}v_x^2},\qquad
v_{crit} = \sqrt{-L/K_{us}}$$

> ⚠️ **자주 틀리는 지점**: $A$는 블록 하삼각이라 $e_\psi, e_y$가 순수 적분기 →
> **원점 고유값 2개가 정상**이다. 4×4 전체에 Hurwitz 검사를 걸면 반드시 실패한다.
> 안정성 판정은 $A_{lat}=A[0{:}2,0{:}2]$에만 하며,
> $A_{lat}$ Hurwitz $\iff L+K_{us}v_x^2>0$ — $\gamma_{ss}$의 분모와 **같은 식**이라
> 두 검증이 서로 교차검증된다.

### 2번 — 잔차가 무엇으로 구성되는가 ✅

$$\dot x = f(x,u) + d(x,u)$$

**결론부터**: 잔차는 **타이어 비선형성 하나**이고, 4채널 중 **2채널만 독립 정보**다.

실험 설계상 명목 모델과 플랜트의 차이를 타이어 힘 법칙 하나로 통제했다.

| | 타이어 힘 |
|---|---|
| 명목 (MPC 예측) | 선형 $F_y = C_\alpha\alpha$ |
| 플랜트 (실제) | Fiala 브러시 (포화 있음) |

$$F_{y,max}=\mu F_z,\qquad \alpha_{sl}=\arctan\frac{3F_{y,max}}{C_\alpha},\qquad z=\tan(\mathrm{clip}(\alpha,\pm\alpha_{sl}))$$

$$F_y = C_\alpha z - \frac{C_\alpha^2}{3F_{y,max}}z|z| + \frac{C_\alpha^3}{27F_{y,max}^2}z^3$$

**구현**: [`src/models/tire.py`](src/models/tire.py), [`src/models/nonlinear_bicycle.py`](src/models/nonlinear_bicycle.py)

**측정 결과 — 잔차는 어느 채널에 실리는가** (sedan, $a_y$=4.0):

| 채널 | 잔차 RMS | noise floor 대비 | 성격 |
|---|---|---|---|
| $v_y$ | 1.30e-2 | ×1.36M | **독립 정보** — GP 학습 대상 |
| $\gamma$ | 1.21e-3 | ×1.33M | **독립 정보** — GP 학습 대상 |
| $e_\psi$ | 1.26e-5 | ×68k | 종속 ($=0.5\Delta t\cdot r_\gamma$) |
| $e_y$ | 1.33e-4 | ×235k | 종속 ($=0.5\Delta t\cdot r_{v_y}$) |

기구학 채널의 잔차는 새 정보가 아니라 **동적 채널 잔차의 결정론적 $0.5\Delta t$ 이미지**다.
이 관계가 "GP는 $v_y,\gamma$만 학습하면 된다"는 설계를 검증한다.

$a_{y,max}$를 올리면 타이어가 비선형 영역으로 더 들어가 잔차가 커진다:

| $a_y$ | SNR($v_y$) | SNR($\gamma$) | max\|$\alpha_f$\| |
|---|---|---|---|
| 1.0 | ×70k | ×53k | 0.74° |
| 4.0 | ×1.36M | ×1.33M | 3.68° |
| 6.0 | ×3.86M | ×4.87M | 6.85° |
| 7.0 | ×6.56M | ×13.0M | 9.99° |

($\alpha_{sl}\approx15.4°$. $a_y$=4~6이 비선형–비포화 영역 → GP 실험 구간으로 채택)

### 3번 — 잔차 없이 path tracking control 구현 ✅ (Python 채택)

Python과 MATLAB 두 선택지 중 **Python으로 구현했다.** 이유:

- 4번(잔차 모델링)의 GP를 **MPC 예측식 안에 직접 결합**해야 하는데, CasADi로 GP 평균을
  심볼릭 표현으로 만들어 IPOPT NLP에 파라미터로 주입하면 이 결합이 자연스럽게 된다.
  MATLAB Vehicle Dynamics Blockset은 차량 모델은 편하지만 이 결합 경로가 매끄럽지 않다.
- 명목 모델을 직접 유도해 쓰기로 한 이상(1번), 블록셋의 기성 차량 모델을 쓰면 오히려
  **잔차의 출처를 타이어 하나로 통제**한다는 실험 설계가 깨진다.
- config 기반 실험 관리·재현성 스냅샷·물리 게이트 테스트를 한 스택에서 처리할 수 있다.

MATLAB 병행 구현은 계획에 없다. 필요하시면 말씀해 주시면 진행하겠다.

요구사항 전부 충족:

| 요구 | 구현 |
|---|---|
| vehicle model | [`linear_bicycle.py`](src/models/linear_bicycle.py) (명목) + [`nonlinear_bicycle.py`](src/models/nonlinear_bicycle.py) (플랜트) |
| **RK4 이상 적분기** | MPC 예측 = RK4, **플랜트 = DOP853**(8차, rtol 1e-10) — [`integrators.py`](src/models/integrators.py) |
| **plant 100 Hz / control 50 Hz** | `dt_plant=0.01`, `dt_ctrl=0.02`, 그 사이 ZOH — [`runner.py`](src/sim/runner.py) |

**적분기를 일부러 다르게 둔 이유**: 두 경로의 차이가 곧 잔차의 **noise floor**를 정의한다.
이 값을 먼저 숫자로 확보해야 "이 잔차가 진짜 모델 오차인가, 이산화 오차인가"를 판정할 수 있다.

| 채널 | noise floor (RMS) |
|---|---|
| $v_y$ | 9.57e-9 m/s |
| $\gamma$ | 9.09e-10 rad/s |
| $e_\psi$ | 1.84e-10 rad |
| $e_y$ | 5.67e-10 m |

**MPC 정식화** (multiple shooting + IPOPT, [`mpc_base.py`](src/control/mpc_base.py)):

$$\min_{X,U}\sum_{k=0}^{N-1}\Big[W_{e_y}e_{y,k}^2+W_{e_\psi}e_{\psi,k}^2+W_{v_y}v_{y,k}^2+W_\gamma\gamma_k^2+R_\delta\delta_k^2+R_{\Delta\delta}(\delta_k-\delta_{k-1})^2\Big]+\text{종단항}$$

$$\text{s.t.}\quad X_{k+1}=F(X_k,U_k,v_{x,k},\kappa_k),\quad |\delta|\le\delta_{max},\quad |\Delta\delta|\le\dot\delta_{max}\Delta t$$

- **참조 상태 $x_{ref}$·피드포워드 $u_{ref}$를 만들지 않는다.** $\kappa$ 프리뷰가 지평에
  채워지면 MPC가 필요한 요레이트를 스스로 만든다.
- 경로는 (x,y) 점열이 아니라 **호길이에 대한 곡률 프로파일 $\kappa(s)$** 로 정의한다 —
  오차좌표계 MPC가 요구하는 형태다. [`reference.py`](src/path/reference.py)
- warm start + IPOPT 수렴 실패 시 fallback(이전 해 shift) 구현.

**실시간성**: 폐루프 solve time mean **9.2 ms** (`dt_ctrl` = 20 ms), `dt_ctrl` 초과율 0%.

### 4번 — 잔차 모델링 ✅ (KF + GPR 모두 완료)

#### (a) Disturbance observer — 외란증강 칼만필터

[`src/estimation/ekf.py`](src/estimation/ekf.py)

$$x_{aug}=[v_y,\ \gamma,\ e_\psi,\ e_y,\ d_{v_y},\ d_\gamma]^\top\in\mathbb{R}^6$$

$$\dot v_y = f_{nom,v_y}+d_{v_y},\qquad \dot\gamma=f_{nom,\gamma}+d_\gamma,\qquad \dot d = 0\ (\text{random walk})$$

외란을 $v_y,\gamma$에만 붙인 근거는 위 2번의 **측정된 잔차 SNR**이다.

$$P^-=FPF^\top+Q_{kf}\Delta t,\qquad K=P^-H^\top(HP^-H^\top+R_{kf})^{-1},\qquad P^+=(I-KH)P^-$$

야코비안 $F,H$는 `ca.jacobian`으로 자동 생성한다(수기 미분 금지) → 실차 비선형 측정으로
갈 때 필터 코어를 건드리지 않고 모델 함수만 교체하면 된다.

> **정직한 명명**: 명목 모델과 측정이 둘 다 선형이라 $F,H$가 상수다. 따라서 엄밀히는
> EKF가 아니라 **외란증강 LTV 칼만필터**다. 선형계에서 KF는 최적 추정기이므로,
> 이건 비교군을 약화하는 게 아니라 **최강 형태**를 주는 것이다.

#### (b) GPR — 잔차 학습

[`src/gp/`](src/gp/) — 직접 구현한 exact GP (numpy/scipy)

**잔차 정의** (흔들리면 전부 무의미):

$$r_k = x_{k+1}^{plant} - f_{nom}^{d}(x_k,u_k,v_{x,k},\kappa_k)$$

$f_{nom}^d$는 **MPC가 예측에 쓰는 것과 정확히 같은** RK4 이산화여야 한다. 코드는
`build_step_function(NominalStepModel)`을 **재사용**한다(복사가 아니다).

**입력·커널·학습**:

$$z=[v_y,\ \gamma,\ \delta]\in\mathbb{R}^3,\qquad
k(z,z')=\sigma_f^2\exp\Big(-\tfrac12\textstyle\sum_d\frac{(z_d-z'_d)^2}{\ell_d^2}\Big)$$

- 채널 $v_y,\gamma$ 독립 GP 2개, ARD RBF
- Type-II ML (log-det 포함 표준 NLML)
- $v_x$ 제외 근거: $\partial\alpha_f/\partial v_x\approx5\times10^{-4}$ rad/(m/s) → 슬립각 기여 0.04°(무시)

**MPC 결합** (이산 형태 — 단위가 핵심):

$$x_{k+1}=\mathrm{rk4}(f_{nom},x_k,u_k,\cdot,\Delta t) + B_d\hat\mu,\qquad
B_d=\begin{bmatrix}1&0\\0&1\\0&0\\0&0\end{bmatrix}$$

> ⚠️ **가장 흔한 버그**: $r_k$는 상태 **차이**(m/s)이지 미분(m/s²)이 아니다. 연속 우변에
> 더하면 RK4 내부에서 적분되어 $\Delta t$배, 즉 **50배 축소**된다. 반드시 RK4 **밖**에서 더한다.
> KF의 $d$는 반대로 rate라 RK4 **안**에 넣는다 — 두 결합의 위치가 다른 이유다.

**사후분산은 제어에 넣지 않되 반드시 저장한다.** 제어는 mean-only(non-cautious)이고
분산을 제약·비용에 전파하지 않지만, 매 스텝 계산·로깅한다 — 논문의 주 증거물이다.

$$\sigma^2(z)=\sigma_f^2 - k(z,Z)K^{-1}k(Z,z)$$

---

## 2. 지시 범위를 넘어 추가로 진행한 것

4번까지 마친 뒤, "GP가 KF보다 낫다"를 **주장이 아니라 측정으로** 만들기 위해 아래를 추가했다.

| 항목 | 내용 |
|---|---|
| **3-way 통합 실험 인프라** | 케이스를 config로만 정의하고, 조립된 config를 diff해 **통제변수가 다르면 실행을 거부** |
| **UQ 캘리브레이션** | 커버리지·NLPD. KF의 연속 외란을 CasADi 야코비안으로 이산 잔차 공간에 환산해 **GP와 같은 대상**으로 비교 |
| **분포이동 시나리오** | in-distribution / 강도 외삽($a_y$ 4→6) / 형상 외삽(dlc) |
| **외부 레이싱 트랙 검증** | 강의 실습 배포 `.mat` 트랙(955.5 m)으로 재현 — 합성 경로 편향 배제 |
| **트랙 GP의 형상외삽 축 추가** | 방향 반전은 사전 검증에서 기각(원본 트랙이 이미 좌우 혼합이라 새 영역이 아님) → 강의 자료의 별도 트랙(right_turn, 76% 우회전 전용)으로 대체, 채택 전 사후분산 실측으로 확인 |
| **현실적 센서 잡음** | 이상적 센서 가정을 걷어내고 측정 잡음 하에서 재실험 |
| **물리 게이트 테스트 69개** | "코드가 도는지"가 아니라 "물리가 맞는지"를 본다 |

---

## 3. `results/` 로 본 진행 과정

각 실행은 `results/<run_id>/`에 **config 전체 스냅샷 + git hash + seed + 라이브러리 버전**과
함께 저장된다. 재현 불가능한 실험은 실험이 아니다.

> `results/`와 `data/`는 `.gitignore` 대상이라 GitHub에는 올라가지 않는다.
> §6의 실행 명령으로 그대로 재생성할 수 있다.

| # | run_id | 단계 | 무엇을 확인했나 |
|---|---|---|---|
| 1–2 | `openloop_20260725_2137*` | 3번 | step steer 응답. sedan $\tau\approx0.126$ s(감쇠진동) vs limo $\tau\approx0.010$ s(과감쇠). **정상상태 요레이트가 해석해와 1% 이내 일치** |
| 3 | `baseline_matched_20260725_2207` | 3번 | 플랜트=명목(완전일치) 폐루프. 잔차가 noise floor와 같은 자릿수인지 → **구현 검증** |
| 4 | `mismatch_sedan_20260725_2220` | 2번 | 플랜트만 Fiala로 교체. **잔차 SNR·커버리지 산점도·$a_y$ 스윕** → GP 설계의 입력 |
| 5 | `part1_20260726_163843` | 4번 | 3-way 첫 통합 실행 (ay4만, 파이프라인 검증) |
| 6 | `part1_20260726_163931` | 추가 | **합성 경로 9런** (3케이스 × ay4/ay6/dlc) |
| 7 | `part1_20260726_170758` | 추가 | **레이싱 트랙 6런** (3케이스 × rt_ay4/rt_ay6) |
| 8 | `part1_20260726_184536` | 추가 | 센서 잡음 첫 실행 → **KF 발산 발견** (설정 오류, §4 참조) |
| 9 | `part1_20260726_190107` | 추가 | $Q_d$ 재튜닝 후 **잡음 6런 재실행** (최종) |
| 10 | `part1_20260727_000727` | 추가 | **right_turn 형상외삽 3런** — 재학습 없이 기존 트랙 GP로 평가 |
| 11 | `part1_20260727_002844` | 추가 | **right_turn + 센서 잡음 3런** — 형상축 우위가 잡음에서도 유지되는지 확인 |

각 `part1_*` 폴더 구성:

```
results/part1_<timestamp>/
├── summary.csv / summary.json      # 전 케이스 비교 요약
├── meta.json                       # 배치 재현성 메타
└── <시나리오>/
    ├── part1_<s>_cases.png         # 케이스 비교 (e_y, δ, 잔차)
    ├── part1_<s>_uncertainty.png   # ★ 주 그림 — GP 상태공간 지도 vs KF 시간축
    └── <케이스>/
        ├── log.npz                 # 제어 스텝별 전체 로그
        ├── meta.json               # config 스냅샷 + git hash + seed
        └── *_report.png            # 단일 런 종합 그림
```

---

## 4. 주요 결과

### 추종 성능 (RMS $e_y$ [m], 괄호 = MPC-only 대비)

| 시나리오 | MPC only | MPC+KF | MPC+GP |
|---|---|---|---|
| ay4 (합성, in-dist) | 9.56e-3 | 9.01e-3 (−5.8%) | **8.39e-3 (−12.2%)** |
| ay6 (합성, 강도외삽) | 2.79e-2 | 1.89e-2 (−32.1%) | 1.90e-2 (−31.9%) |
| dlc (합성, 형상외삽) | 1.03e-2 | 1.20e-2 (+17.0%) | 1.29e-2 (+25.1%) |
| **rt_ay4** (트랙, in-dist) | 9.17e-3 | 8.22e-3 (−10.3%) | **7.71e-3 (−16.0%)** |
| **rt_ay6** (트랙, 강도외삽) | 2.84e-2 | **1.92e-2 (−32.5%)** | 2.47e-2 (−13.0%) |
| **rtn_ay4** (트랙+잡음, in-dist) | 1.45e-2 | 1.35e-2 (−6.4%) | **1.26e-2 (−13.1%)** |
| **rtn_ay6** (트랙+잡음, 외삽) | 3.29e-2 | **2.89e-2 (−12.1%)** | 2.92e-2 (−11.4%) |
| **rt_rturn** (트랙, 형상외삽) | 9.17e-3 | 8.25e-3 (−10.1%) | **7.52e-3 (−18.0%)** |
| **rtn_rturn** (트랙+잡음, 형상외삽) | 1.391e-2 | 1.355e-2 (−2.6%) | **1.279e-2 (−8.1%)** |

### UQ 캘리브레이션 (대상 = 1스텝 이산 잔차 $r_k$)

$z_{std}>1$ = 과신(위험), $<1$ = 보수적

| 시나리오 | KF $z_{std}$ / 95%커버 | GP $z_{std}$ / 95%커버 |
|---|---|---|
| rt_ay4 | 0.601 / 99.2% | 1.748 / 85.4% |
| rt_ay6 | 1.879 / 85.0% | **3.255 / 52.6%** |
| rtn_ay4 | 9.319 / 38.0% | 9.278 / 53.6% |
| rtn_ay6 | 17.099 / 24.5% | **4.864 / 34.5%** |
| **rt_rturn** | 0.606 / 99.2% | **1.080 / 94.1%** |
| **rtn_rturn** | 8.513 / 39.3% | **3.163 / 58.6%** |

### 정직한 결론 네 가지

**① in-distribution에서 GP 우위는 견고하다.** 합성 경로(−12.2%), 외부 트랙(−16.0%),
센서 잡음 하(−13.1%) 모두에서 GP > KF > MPC-only 순서가 유지된다.

**② 핵심 UQ 주장은 성립한다.** GP 사후 std가 분포 밖에서 **12.7~30배 증가**한다.
"GP는 안 가본 영역을 안다"는 주장은 데이터로 지지된다.

**③ 그러나 절대 캘리브레이션은 실패한다** — 원인 두 가지를 규명했다.

- **원인 ① 사전분산 상한**: $\sigma^2(z)\le\sigma_f^2$ 이므로 사후분산은 사전분산을
  절대 넘을 수 없다. rt_ay6에서 실제 오차가 상한의 **3.19배**, 랩의 **39.3%**가 상한에
  붙박여 커버리지가 52.6%로 붕괴했다. 튜닝 문제가 아니라 정상(stationary) 커널의 구조적 성질이다.
- **원인 ② 배치 조건 변화 무감지**: 센서 잡음 → 조향 변화율 **9배 증가**(0.088→0.798 rad/s)
  → 플랜트가 학습된 다양체를 벗어남. 그 오차가 **GP가 가장 자신 있는 구간에 집중**된다
  (캘리브레이션 실패의 **91.3%**가 사후std 하위 25% 구간에서 발생, 그 구간 \|오차\|/σ = **6.14**).
  즉 신뢰도가 오차와 **역상관**이라 경고 기능을 못 한다.

**④ KF의 외삽 우위는 완벽한 센서의 산물이었다.** 깨끗한 rt_ay6에서 KF는 −32.5%로 GP를
압도했으나, 현실적 잡음이 들어가자 −12.1%로 떨어져 GP(−11.4%)와 동률이 됐다.

**⑤ rt_rturn — GP가 추종·캘리브레이션 둘 다 KF를 이긴 첫 시나리오.** 트랙 GP에도 dlc
같은 형상외삽 축을 만들려고 방향 반전을 검토했으나, 원본 트랙이 이미 좌우 혼합이라
(딕셔너리 69/29% 분포) 반전해도 새 영역이 아니라고 판단해 기각했다. 대신 강의 자료의
별도 트랙(`right_turn`, 76%가 우회전 전용)으로 대체했고, 채택 전 별도 스크립트로 GP
사후분산이 실제로 오르는지 먼저 확인했다(평균 2~3배, 최대 14배). 그 결과 GP가 추종에서
자기 학습 트랙보다도 더 이기고(−18.0% vs −16.0%) 캘리브레이션도 개선됐다($z_{std}$
1.748→1.080, 커버리지 85.4%→94.1%, 이상값 1.0/95%에 근접). **같은 "형상외삽"인데 dlc는
정반대 결과(KF·GP 둘 다 MPC-only에 짐)라, 무엇이 이 차이를 가르는지는 아직 확인하지
못했다** — 원인 없이 좋은 결과만 보고하지 않기 위해 §8에 열린 질문으로 남긴다.

**⑥ 형상축의 GP 우위는 센서 잡음에서도 순위가 안 뒤집힌다.** rt_rturn에 잡음을 더한
rtn_rturn에서 GP-KF 격차가 깨끗한 조건(7.9pp)과 잡음 조건(5.5pp) 둘 다 유지된다. 이는
강도축(rt_ay6)과 대비된다 — 거기서는 KF의 압도적 우위(−32.5%)가 잡음에서 −12.1%로
무너져 GP(−11.4%)와 동률이 됐다. 다만 캘리브레이션은 형상축도 잡음에서 크게 무너진다
($z_{std}$ 1.080→3.163). GP가 KF보다는 덜 나쁘고(3.163 vs 8.513), rtn_ay4의 GP(9.278)
보다도 낫다 — 형상축이 강도축보다 잡음 하에서도 상대적으로 낫다는 패턴은 유지된다.

### 진행 중 발견하고 수정한 설계 오류

결과가 좋게 나오도록 덮지 않고 원인을 찾아 고친 것들이다.

| 단계 | 증상 | 원인 | 수정 |
|---|---|---|---|
| KF | MPC+KF가 MPC-only보다 **−18% 나쁨** | 부분관측($v_y$ 미측정)이면 $d_{v_y}$가 $e_y$로부터 **이중적분 뒤에야** 관측 → 심한 지연·과대추정 | 전상태 측정으로 변경 → **+5.8%** |
| GP | 격자 완벽적합에도 **−74% 불안정** | 상태의존 지평 결합 + mean-only → 최적화기가 GP 부정확 영역을 **악용** | 현재 작동점 1회 평가 후 **지평 상수** 주입 → **+12%** |
| 로깅 | — | GP 사후분산이 **계산은 되는데 기록되지 않고 있었음**(논문 주 증거물) | `MpcBase.step_log()` 훅 추가 |
| 센서 | rtn_ay6에서 KF가 **RMS 3454 m로 발산** | `R_kf`만 센서에 맞추고 `Q_kf`는 이상적 센서 시절 값 유지 → Q/R 비율 붕괴 → 잡음을 외란으로 오인하는 양성 피드백 | in-distribution만 보고 $Q_d$ 재튜닝(8점 그리드) → 해소 |

---

## 5. 저장소 구조

```
gpmpc/
├── CLAUDE.md              # 프로젝트 규약 (부호·구조·금지 사항)
├── handoff.md             # 세션 인계 + 미결 결정 단일 소스
├── prompts.md             # Phase별 작업 지시 이력
├── .claude/rules/         # 경로별 조건부 룰 7개
├── configs/               # 모든 수치의 단일 출처 (하드코딩 금지)
│   ├── vehicle/  path/  mpc/  ekf/  gp/  sim/  sensor/  viz/
│   └── experiment/        # 조합 파일 = 케이스 정의
├── src/
│   ├── models/            # linear_bicycle, tire, nonlinear_bicycle, integrators
│   ├── path/              # reference — κ(s) 프로파일·프리뷰
│   ├── control/           # mpc_base, mpc_nominal, mpc_kf, mpc_gp
│   ├── estimation/        # ekf — 외란증강 LTV KF
│   ├── gp/                # dataset, kernels, train_offline, casadi_export
│   ├── sim/               # runner(다중레이트), assemble(케이스 조립), sensor, logger
│   ├── eval/              # metrics, calibration
│   └── viz/               # plots
├── scripts/               # run_openloop, run_sim(범용), run_part1(배치)
├── tests/                 # 물리 게이트 69개
├── data/                  # 경로·GP 데이터셋 (git 제외)
└── results/               # run_id별 산출물 (git 제외)
```

### 설계의 중심 — `StepModel` 주입

세 케이스가 `if`문 없이 갈리는 이유가 여기 있다. **`MpcBase`는 어떤 케이스인지 모른다.**

```python
class StepModel(Protocol):
    extra_param_dim: int
    def step_sym(self, x, u, vx, kappa, p_extra): ...   # 이산 1스텝 전이
    def extra_param_values(self) -> np.ndarray: ...
```

| 구현체 | 이산 전이 | 결합 위치 |
|---|---|---|
| `NominalStepModel` | $\mathrm{rk4}(f_{nom})$ | — |
| `DisturbanceStepModel` (KF) | $\mathrm{rk4}(f_{nom}+B_c d)$ | RK4 **안** ($d$는 rate) |
| `GPStepModel` (GP) | $\mathrm{rk4}(f_{nom})+B_d\hat\mu$ | RK4 **밖** ($r$은 상태차이) |

케이스 선택도 문자열 분기가 아니라 **참조된 config 그룹**으로 한다
([`src/sim/assemble.py`](src/sim/assemble.py)) — `gp` 그룹이 있으면 GP 케이스,
`ekf`가 있으면 KF, 둘 다 없으면 명목. 새 케이스 추가 = 레지스트리 한 줄.

---

## 6. 실행 방법

```bash
pip install -r requirements.txt      # Python 3.12 기준

# 물리 게이트 전체 (약 80초)
pytest tests/ -q

# 개루프 step steer 응답 (지시 3번)
python scripts/run_openloop.py

# 단일 실험
python scripts/run_sim.py part1_gp_rt_ay4

# 3-way 배치 (통제변수 자동 검증 후 실행)
python scripts/run_part1.py rt_ay4 rt_ay6     # 레이싱 트랙
python scripts/run_part1.py rtn_ay4 rtn_ay6   # + 센서 잡음
python scripts/run_part1.py                   # 전체
```

시나리오: `ay4` `ay6` `dlc` (합성) / `rt_ay4` `rt_ay6` `rt_rturn` (트랙) /
`rtn_ay4` `rtn_ay6` `rtn_rturn` (트랙+잡음)

**트랙 데이터**는 `data/path/`에 두 파일을 두어야 한다(둘 다 강의 실습 배포본.
`data/`는 git 제외이므로 별도 복사 필요. config에 sha256을 고정해 파일이 바뀌면 즉시 중단):
- `path_data_racetrack.mat` — 실습 2차 배포본 (rt_ay4/rt_ay6/rtn_*)
- `path_data_right_turn.mat` — 실습 1차 배포본 (rt_rturn)

---

## 7. 실험 조건

| 항목 | 값 |
|---|---|
| 차량 | sedan ($m$=1500 kg, $I_z$=2250 kg·m², $a$=1.2 m, $b$=1.5 m, $C_f$=$C_r$=80000 N/rad, $\mu$=0.9) |
| 다중레이트 | plant 100 Hz (DOP853) / control 50 Hz (RK4), ZOH |
| MPC | $N$=30 (0.6 s), IPOPT, warm start |
| 경로 | 합성 single_curve(140 m)·dlc(126 m) / 외부 레이싱 트랙(955.5 m, $R_{min}$=32.4 m) |
| GP | $M$=100, ARD RBF, Type-II ML, $\sigma_n$ floor 1e-2 |
| 센서 잡음 | std [0.03 m/s, 0.002 rad/s, 0.003 rad, 0.03 m] — 센서 등급별 통상값 기준 **공학적 추정치** |
| seed | 0 (전 케이스 동일) |

**공정성 원칙**: KF와 GP에 동일한 튜닝 예산(폐루프 평가 20회 이내)을 배정하고,
**in-distribution 시나리오만 보고** 결정한 뒤 외삽 시나리오 결과를 본 후에는 어느 쪽도
수정하지 않는다. 튜닝 방식·예산은 config에 같은 형식으로 기록한다.

---

## 8. 남은 작업

### 착수 전 결정할 것

1. **dlc·rt_ay6 열세의 원인 판별** — 지평 상수 주입의 구조적 한계인가, 튜닝 문제인가.
   판별법: 오라클(참 1스텝 잔차 상수 주입)을 돌려 그것도 지면 구조적 한계.
   **튜닝으로 덮기 전에 이것부터.**
2. **GP 과신 처리** — 한계로 서술 / 비정상 커널·입력의존 잡음으로 상한 제거 /
   $z$에 $v_x$ 추가(상한 문제 자체는 남음)
3. **튜닝 목적함수 불일치** — 주 지표는 캘리브레이션인데 $Q_d$를 추종 RMS로 튜닝했다.
   NLPD 기준 재튜닝 여부
4. **dlc(과신 유지) vs rt_rturn(캘리브레이션 개선)** — 같은 "형상외삽"인데 정반대
   결과가 나온 원인. dlc는 좌우 교대라는 확실한 새 형태인 반면 right_turn은 원본
   트랙이 이미 어느 정도 우회전을 포함해 딕셔너리와 겹쳤을 가능성(**추측**, 미검증)

### 예정 작업

- **Part 2: GP 실시간화** — online / sparse / sliding-window GP.
  `src/gp/{online,sparse,sliding_window}.py`가 현재 스텁. 주 지표는 solve time 분포와
  `dt_ctrl` 초과율. 네 변형이 같은 인터페이스를 구현해 runner가 변형을 몰라야 한다.
- **LIMO 하드웨어** (선택) — 착수 전 $C_f, C_r, I_z, \delta_{max}$ **식별 실험 필수**.
  현재 `configs/vehicle/limo.yaml` 값은 전부 자릿수 추정치라 그대로 쓰면 결과 해석이 불가능하다.
  (시뮬레이션에서 limo는 속도 범위가 좁아 슬립 유도가 안 되는 것도 확인됨)

---

## 9. 알려진 한계

정직하게 적어 둔다.

- **센서 잡음 값은 특정 논문 수치가 아니라 센서 등급별 통상값 기준의 공학적 추정치**다.
  문헌 인용이 필요하면 실제 데이터시트로 대체해야 한다.
- $v_y$를 직접 측정한다고 가정했다. **양산차에는 $v_y$를 직접 재는 센서가 없다.**
  부분관측이 KF를 악화시키는 것을 확인한 뒤, 이 연구의 초점이 상태추정이 아니라
  모델오차 처리이므로 전상태 측정으로 통제한 결과다.
- Fiala 모델의 큐빅 형태는 여러 문헌에서 동일하게 확인되지만 **원 출처 DOI는 확인하지 못했다.**
  다만 선형 극한·포화·원점 대칭 성질은 수치미분으로 자체 검증했다.
- `limo` 차량 파라미터는 전부 미검증 추정치다(YAML의 `TODO` 주석 참조).
- GP는 mean-only(non-cautious)다. 분산을 제어에 전파하는 stochastic MPC는 이번 스코프 밖이다.
- **잡음 시나리오에서 조향 변화율이 전체 시간의 ~47%를 액추에이터 속도한계에서 보낸다**
  (rtn_ay4/ay6/rturn 공통, mpc_only 포함 세 케이스 균일). 버그가 아니라 "세 케이스
  모두 필터링 안 된 생측정값을 제어 피드백으로 받는다"는 설계 결정의 직접적 대가다 —
  KF가 있어도 그 필터링 결과는 외란 추정에만 쓰이고 피드백 경로 자체에는 반영되지
  않는다. solve time도 동반 상승한다(mean 13~17ms, `dt_ctrl` 초과 최대 6.7%).
- **잔차를 타이어 비선형성 하나로 통제한 것은 실차 잔차를 대표하지 않는다.** 확인해 본
  GP-MPC 문헌(Hewing et al. arXiv:1705.10702, GP-based overtaking MPC 2021)은 대부분
  잔차를 "model mismatch and unmodeled dynamics"로 뭉뚱그려 학습시키지, 물리 원인
  하나로 좁히지 않는다. 이 프로젝트의 단일원인 통제는 인과관계를 깨끗하게 검증하기
  위한 의도적 선택이며, combined slip·하중이동·액추에이터 지연·파라미터 불확실성 등
  실차에서 섞여 들어올 다른 잔차원은 다루지 않는다.
