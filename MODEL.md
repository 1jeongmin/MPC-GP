# MODEL — 모델 유도 및 정식화 (README §1 분리)

`README.md` 에서 분리했다(2026-07-30, 파일 크기 때문에 분리). 지시받은 4단계
(state equation / 잔차 정의 / path tracking 구현 / 잔차 모델링)에 대한 상세 답변이다.

> **⚠ 아래 MPC 비용함수 수식은 원래 답변 당시(Phase 3) 것이다.** 실제 현재
> 튜닝값(`W_vy=0`, `W_gamma` 참조 상대화, `N=50`+입력 블로킹)은
> `.claude/rules/mpc-solver.md` 가 단일 소스다. 여기 수식은 **정식화 구조**
> (multiple shooting, 결합 형태)를 설명하는 용도로만 봐라.

---

## 1. 지시받은 모델링 순서와 진행 현황

주신 4단계에 대한 대응이다. **1~4번 모두 완료**했고, 4번 이후로 추가 작업이 상당히
진행됐다(`README.md` §2). 3번은 Python/MATLAB 두 선택지 중 **Python을 채택**해
구현했다(사유는 아래).

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
