# Claude Code 프롬프트 — GP-MPC 본 저장소

> **이 문서는 규약을 반복하지 않는다.** 상태방정식·부호 규약·코딩 규칙·테스트 기준은
> 전부 `.claude/rules/`가 담당한다. 프롬프트에 같은 내용을 다시 적으면 관리 지점이
> 두 개가 되고 나중에 반드시 어긋난다. 프롬프트는 **"무엇을 만들지"**만 말하고
> **"어떻게 만들지"**는 룰에 맡긴다.
>
> 같은 이유로 **미결 결정은 여기에 적지 않는다.** 단일 소스는 `handoff.md`의
> 「미결 결정」 표다.

## 사용 규칙

1. 프롬프트를 **하나씩, 순서대로** 입력한다.
2. 각 Phase의 게이트를 통과했는지 확인하고 다음으로 넘어간다. 실패하면 멈춘다.
3. Phase가 바뀌면 `/clear`로 새 세션을 연다.
4. **각 Phase 시작 시 해당 룰 파일을 명시적으로 Read 시켜라.** `paths` 스코프 룰은
   매칭 파일을 읽을 때만 컨텍스트에 올라온다. 새 파일을 만들 때는 안 올라온다.
5. 컨텍스트 사용률과 로드된 룰은 `/context`로 확인한다. 60% 근처에서 `/compact`.
6. 세션을 끝내기 전에 `handoff.md`를 갱신한다.

---

## Phase -1 — 저장소 골격 · **완료됨 (2026-07-24)**

다시 실행하지 마라. `handoff.md`를 빈 파일로 덮어써서 「미결 결정」 표가 날아간다.

이미 존재하는 것: `git` 저장소, `.gitignore`, `requirements.txt`, `handoff.md`,
`configs/` 그룹 7개 디렉토리, `src/`·`scripts/` 모듈 스텁 25개 + `__init__.py`.

- 스텁 파일은 **룰 자동 로드 트리거 확보용**이다. 지우지 마라.
- `requirements.txt`에 `torch`/`gpytorch`가 **이미 들어 있다.** Phase 7 전까지는
  설치하지 않아도 무방하다. 빼지 마라 (뺐다 넣으면 재현성 스냅샷이 갈린다).

**확인 방법**: 새 세션에서 `/context` 실행.
**정상 상태는 7개 룰 중 0개가 로드된 것이다.** 전부 `paths` 스코프이므로 세션 시작
시점에는 올라오지 않는 것이 정상이다. `src/models/linear_bicycle.py`를 Read 시킨 뒤
`vehicle-model.md`가 올라오면 정상 동작하는 것이다.

---

## Phase 0 — config 시스템 + 선형 모델

```
Phase 0을 진행한다. config 로딩 체계와 선형 자전거 모델을 구현한다.

먼저 .claude/rules/vehicle-model.md 와 .claude/rules/python-conventions.md 를
Read 해라. 부호 규약과 코딩 규칙이 거기에 있다. 이 프롬프트에서 반복하지 않는다.

## configs/vehicle/sedan.yaml (검증 기준값)
m: 1500.0        # kg
Iz: 2250.0       # kg*m^2
a: 1.2           # m, CG -> 전축
b: 1.5           # m, CG -> 후축
Cf: 80000.0      # N/rad, 전축 코너링 강성 (축당)
Cr: 80000.0      # N/rad, 후축 코너링 강성 (축당)
delta_max: 0.5        # rad
delta_rate_max: 1.0   # rad/s
vx_range: [5.0, 25.0] # m/s

이 조합은 K_us > 0 (언더스티어)이고, vx=15, delta=0.02 에서
정상상태 a_y 가 약 1.4 m/s^2 로 선형 타이어 영역 안이다. 게이트 1의 기준값이다.

## configs/vehicle/limo.yaml (AgileX LIMO, Ackermann)
아래 값은 전부 미검증 추정치다. YAML의 TODO 주석을 반드시 유지하라.
m: 4.2           # kg      (TODO: 실측 필요)
Iz: 0.05         # kg*m^2  (TODO: 직육면체 근사 Iz = m*(l^2+w^2)/12,
                 #          l~0.32 w~0.22 기준. 실측/식별로 대체할 것)
a: 0.1           # m       (TODO: CG가 축간 중앙이라는 가정)
b: 0.1           # m       (TODO: 동일)
Cf: 200.0        # N/rad   (TODO: 근거 없음. 축하중 약 20N에 Ca ~ 10*Fz 경험칙을
                 #          적용한 자릿수 추정. 반드시 식별 실험으로 대체할 것)
Cr: 200.0        # N/rad   (TODO: 동일)
delta_max: 0.5        # rad (TODO: 실제 조향 한계 확인 필요)
delta_rate_max: 2.0
vx_range: [0.2, 1.0]

이 조합은 a == b, Cf == Cr 이므로 K_us 가 정확히 0(뉴트럴 스티어)이고
critical_speed 는 inf 다. 게이트 2의 뉴트럴 케이스로 그대로 쓸 수 있다.
단 K_us == 0 을 부동소수점 등호로 비교하지 마라. abs(K_us) < tol 로 판정한다.

## configs/sim/default.yaml
dt_plant: 0.01, dt_ctrl: 0.02, duration, seed

## src/config.py
- dataclass: VehicleConfig, MpcConfig, SimConfig (나머지는 나중에 추가)
- 그룹 YAML 로더 + configs/experiment/*.yaml 조합 로더
  조합 파일은 그룹 참조만 담고 값을 직접 갖지 않는다
- 유효성 검사는 로드 시점에 수행. 어떤 검사가 필요한지는
  .claude/rules/vehicle-model.md 의 "특이점" 절을 근거로 판단해라
- 조합된 config를 dict로 덤프하는 함수 (재현성 스냅샷용)

## src/models/linear_bicycle.py
numpy 경로와 CasADi 경로를 모두 제공한다. 상세는 vehicle-model.md 참조.
해석 함수: understeer_gradient, steady_state_yaw_rate, critical_speed

## tests/test_model.py
.claude/rules/testing.md 의 게이트 2b, 3, 4 를 구현한다.
게이트 2b는 4x4 전체가 아니라 부분행렬에 대한 검사라는 점을 특히 주의하라.
A_lat = A[0:2,0:2] 와 오차 기구학 블록 A[2:4,2:4] 는 서로 다른 검사 대상이다.

부호 규약이나 구조에 불명확한 점이 있으면 코드 전에 질문해라.
```

**게이트**: `testing.md` 게이트 2b, 3, 4 통과.

---

## Phase 1 — 적분기 + 개루프 물리 검증

```
Phase 1을 진행한다. 적분기를 만들고 개루프 시뮬레이션으로 모델이 물리적으로
올바른지 검증한다. **이 단계를 통과하지 못하면 MPC로 넘어가지 않는다.**

먼저 .claude/rules/vehicle-model.md 와 .claude/rules/testing.md 를 Read 해라.

## src/models/integrators.py
MPC 예측용 RK4와 플랜트용 고정밀 경로. 요구사항은 vehicle-model.md "적분기" 절 참조.
어느 고정밀 적분기를 왜 골랐는지 주석으로 남겨라.

## tests/test_integrator.py
testing.md 의 게이트 1, 2, 5 를 구현한다.

게이트 5(noise floor)는 테스트가 값을 삼키지 말고 반드시 숫자로 출력해야 한다.
**채널별로 따로 낸다** (v_y, gamma, e_psi, e_y). 이후 잔차 판정이 채널별
noise_floor 대비 배율로 이루어지기 때문이다 (gp-residual.md "진단 임계값").
이 값들을 handoff.md 에도 적어라.

## scripts/run_openloop.py
step steer 응답(v_y, gamma vs 시간)을 플롯한다.
sedan과 limo 둘 다 실행하고, 두 차량의 응답 시상수가 왜 다른지 설명하라.
limo 의 Iz 는 근사값이므로 시상수 해석에 그 불확실성을 함께 적어라.

게이트 1이 실패하면 MPC로 넘어가지 말고 원인을 먼저 찾아 보고하라.
```

**게이트**: 정상상태 요레이트 1% 이내 일치. **채널별** noise floor 수치 확보.

---

## Phase 2 — 참조 경로 (곡률 프로파일)

```
Phase 2를 진행한다. 경로를 (x,y) 점열이 아니라 호길이 s에 대한 곡률 프로파일
kappa(s)로 정의한다. 오차좌표계 MPC에는 이 형태가 필요하다.

먼저 .claude/rules/sim-experiment.md 의 "경로 표현" 절을 Read 해라.

## configs/path/single_curve.yaml
직선 -> 클로소이드 -> 일정곡률 -> 클로소이드 -> 직선.
각 구간 길이, 곡률 반경, a_y_max, kappa_eps 를 파라미터로 둔다.
기본값을 제안하되, 확정 전에 나에게 물어라.

a_y_max 에 대한 주의: 이 값은 단순 튜닝값이 아니다. 두 가지 의미가 겹친다.
1. **명목 모델의 유효 한계**. 우리 명목 모델은 *선형 타이어* 자전거이므로
   구속조건은 타이어 선형 영역이다 (건조 아스팔트에서 대략 a_y < 0.3~0.4 g).
   운동학 자전거 모델의 유효 한계(a_y < 0.5*mu*g, Polack et al. 2017,
   DOI 확인 안 됨)와 혼동하지 마라. 우리는 운동학 모델을 쓰지 않는다.
   숫자는 비슷할 수 있어도 근거가 다르고, 이 문장은 논문에 그대로 실린다.
2. 동시에 **GP가 학습할 잔차의 크기를 결정하는 손잡이**다. 너무 낮으면
   선형 타이어 영역만 밟아 잔차가 noise floor에 묻힌다.
기본값은 path config에 두되 experiment config에서 override 가능하게 하고,
YAML 주석에 위 두 의미를 모두 적어라.

kappa_eps: 직선 구간에서 vx_max = sqrt(a_y_max/|kappa|) 가 0-division 이 된다.
inf 를 경유시키지 말고 분모를 만들기 전에 분기하라 (sim-experiment.md 참조).

## configs/path/dlc.yaml
double lane change 형태의 두 번째 경로. Phase 5 이후 잔차를 크게 만드는 용도.
지금은 정의만 하고 검증은 single_curve 로 한다.

## src/path/reference.py
곡률 프로파일, 속도 프로파일, get_preview, (x,y) 복원 함수.

## tests/test_reference.py
testing.md 게이트 8 을 구현한다. kappa = 0 구간에서 vx_max 가 유한한지도 확인하라.
```

**게이트**: 원호 복원 일치, 프리뷰 길이 불변, `kappa=0`에서 NaN/inf 없음.

---

## Phase 3 — 명목 MPC

```
Phase 3을 진행한다. CasADi multiple shooting + IPOPT.

먼저 .claude/rules/mpc-solver.md 를 Read 해라. 정식화 상세가 전부 거기에 있다.
여기서 반복하지 않는다.

## configs/mpc/default.yaml
N, 상태 가중치, 입력 가중치, 종단 가중치, IPOPT 옵션.
가중치 이름은 KF 잡음 공분산과 헷갈리지 않게 짓는다 (mpc-solver.md "표기 충돌" 참조).
practice1에서 쓰던 R, R_d 라는 이름은 쓰지 마라.

## src/control/mpc_base.py
Controller 추상 인터페이스를 먼저 정의한다.

    solve(x0, preview, u_prev) -> (u, info)

여기서 핵심 설계 요구는 다음이다:
**이산 상태 전이 함수를 casadi.Function 으로 주입받는다.**

    F(x, u, p) -> x_next          # p = (vx, kappa) 등 스텝 파라미터

    명목 케이스: F = rk4_step(f_nom, x, u, p, dt_ctrl)
    GP 케이스  : F = rk4_step(f_nom, x, u, p, dt_ctrl) + B_d @ mu_GP(z)

**주입 대상은 연속 우변 f 가 아니라 이산 전이 F 다.** GP 잔차는 상태 차이
단위이므로 RK4 바깥에서 더해야 한다. 우변에 더하면 dt_ctrl 배 틀린다.
근거는 gp-residual.md "결합 형태는 이산으로 확정" 절. 이 설계로 잡아야
Phase 7에서 GP를 붙일 때 MPC 코드를 수정하지 않는다.
if 분기로 케이스를 늘리는 설계는 금지다 (CLAUDE.md 실험 구조 절).

파라미터 벡터 P 의 레이아웃은 명목/GP 케이스가 다르다. 인덱스를 손으로 세지 말고
레이아웃 정의를 한 곳에 두어라 (mpc-solver.md 참조).

## src/control/mpc_nominal.py
위 인터페이스의 구현체.

## tests/test_mpc.py
testing.md 게이트 6 을 구현한다.
IPOPT 수렴 실패 경로는 실패를 인위적으로 유발해서 실제로 동작하는지 확인하라.

인터페이스 설계에 애매한 점이 있으면 코드 전에 질문해라.
특히 preview 자료구조를 어떻게 넘길지 먼저 합의하자.
```

**게이트**: `testing.md` 게이트 6 전부 통과. solve time이 로그에 남는지 확인.

---

## Phase 4 — 다중레이트 폐루프 + 잔차 로깅

```
Phase 4를 진행한다. 폐루프 시뮬레이터와 로깅을 만든다.

먼저 .claude/rules/sim-experiment.md 와 .claude/rules/gp-residual.md 를 Read 해라.
다중레이트 구조, 로깅 스키마, 재현성 요구, 지표, 플롯 규약이 전부 거기에 있다.

## 만들 것
- src/sim/runner.py   (플랜트 모델 주입 가능)
- src/sim/logger.py
- src/eval/metrics.py
- src/viz/plots.py
- scripts/run_sim.py  (experiment config 경로를 인자로 받는 범용 실행기.
                       run_part1.py / run_part2.py 는 나중에 이 위의 배치 래퍼로 만든다)
- configs/experiment/baseline_matched.yaml
  = sedan + single_curve + mpc/default + sim/default, 플랜트도 선형 (모델 완전일치)

## 잔차 계산에서 절대 어기면 안 되는 것
잔차 정의는 .claude/rules/gp-residual.md 첫 절에 있다.
- **MPC가 예측에 쓰는 것과 같은 함수를 호출**해야 한다. 복사해서 다시 구현하지 마라.
- 잔차는 dt_ctrl 경계에서만 취한다. 플랜트가 dt_plant 로 돌아도 마찬가지다.

## 검증 (핵심)
모델 완전일치 케이스이므로 잔차는 Phase 1에서 측정한 noise floor와
같은 자릿수여야 한다. 자릿수가 다르면 어딘가 잘못된 것이다.

- 4채널 잔차의 RMS를 각각 출력하고, **채널별 noise floor 대비 배율**로 비교 표를
  만들어라. 채널끼리 직접 비교하지 마라 (m/s, rad/s, rad, m 으로 단위가 다르다).
- e_psi, e_y 채널의 판정 기준은 gp-residual.md 의 정량 임계값을 쓴다:
  RMS(r) <= 10 * noise_floor(해당 채널). 초과하면 구현 버그다.

sedan과 limo 둘 다 돌리고 결과를 비교 보고하라.
```

**게이트**: 4채널 모두 잔차 RMS ≤ 10 × 채널별 noise floor.
어긋나면 GP 작업으로 넘어가지 않는다.

---

## Phase 5 — 비선형 플랜트 + 잔차 특성화

```
Phase 5를 진행한다. 여기서 처음으로 실제 모델 불일치를 만든다.
**이 Phase의 산출물이 GP 설계의 입력이 된다.** 서두르지 마라.

먼저 .claude/rules/vehicle-model.md 와 .claude/rules/gp-residual.md 를 Read 해라.

## src/models/tire.py
비선형 타이어 모델을 구현한다. Fiala 브러시 모델을 권장한다.

주의: 아래는 내 기억 기준의 표준형이므로, 구현 전에 교재/논문으로 확인하고
어느 출처를 따랐는지 주석에 명시하라. 확인 못 하면 확인 못 했다고 보고하라.

  포화 슬립각:  alpha_sl = atan(3*mu*Fz / C_alpha)
  |alpha| < alpha_sl 이면 3차 다항식, 그 밖에서는 mu*Fz 로 포화

**부호는 우리 규약(F_y = C_alpha * alpha, vehicle-model.md)에 맞춰라.**
교재 공식은 alpha 부호 규약이 반대인 경우가 많다. 그대로 옮기면 잔차 부호가 뒤집힌다.

## tests/test_tire.py — 가장 중요한 게이트
alpha -> 0 에서 dFy/dalpha 가 C_alpha 로 수렴하는지 수치미분으로 확인하라.
이것이 통과하지 않으면, 잔차에 "타이어 비선형성"이 아니라
"코너링 강성 불일치"라는 다른 원인이 섞여 들어간다. 실험 설계가 오염된다.
추가로: alpha -> 무한대에서 |Fy| -> mu*Fz 포화, 원점 대칭성.

## src/models/nonlinear_bicycle.py
타이어만 교체한다. **오차 기구학(e_psi, e_y 식)은 절대 건드리지 마라.**
잔차의 출처를 타이어 비선형성 하나로 통제하기 위한 실험 설계다.
configs/vehicle/*.yaml 에 mu 와 축하중 Fz 를 추가한다 (limo는 TODO 표기).

## configs/experiment/mismatch_*.yaml
플랜트만 비선형으로 바꾸고 MPC 예측모델은 선형 그대로 두는 케이스.
a_y_max 를 여러 수준으로 스윕할 수 있게 구성한다.

## 잔차 특성화 — 이 Phase의 진짜 산출물
1. 채널별 잔차 RMS가 Phase 1의 채널별 noise floor 대비 몇 배인지 정량화
   -> 이것이 GP가 학습할 신호의 SNR이다
   -> e_psi, e_y 채널은 여전히 10배 이내여야 한다. 여기서 커지면
      타이어가 아니라 기구학 구현이 오염된 것이다
2. a_y_max 수준을 바꿔가며 SNR이 어떻게 변하는지 곡선으로 보고
   -> GP 실험을 어느 a_y 영역에서 돌려야 의미가 있는지 결정하는 근거
3. 잔차를 (v_y, gamma, delta) 공간에 산점도로 뿌려서
   구조가 보이는지, 어느 영역이 데이터로 덮이고 어느 영역이 비는지 시각화
   -> GP 입력 선택과 UQ 논증의 직접 근거
4. sedan과 limo 각각에 대해 위를 수행. limo는 vx_range가 좁아
   슬립 유도가 어려울 수 있다. 어려우면 어렵다고 보고하라.

결과를 results/ 에 저장하고, 요약을 handoff.md 에 남겨라.
```

**게이트**: 타이어 선형 극한 일치. 잔차 SNR·커버리지 시각화 확보.
`e_psi`, `e_y` 채널은 여전히 10 × noise floor 이내.

---

## Phase 5 이후 결정 지점 (2026-07-25 Phase 5 완료)

**Phase 6 이후 프롬프트는 Phase 5 숫자를 봐야 쓸 수 있었다.** 이제 잔차의 크기·분포를
측정했으므로(handoff.md 「Phase 5 특성화 결과」), Phase 6(증강 KF) 프롬프트는 아래에
**확정 기록**했다. Phase 7+(GP)는 Phase 7 착수 시 GP 입력·M·프레임워크를 정하고 쓴다.

결정 항목의 **단일 소스는 `handoff.md`의 「미결 결정」 표**다. 여기에 복제하지 않는다.

**Phase 5 근거로 확정된 Phase 6 설계 (2026-07-25):**
- 모델 오차가 `v_y`,`gamma`에만 실림(잔차 SNR) → 외란을 이 두 채널에 증강.
- 측정 모델 = 현실적 부분관측(`gamma`,`e_psi`,`e_y` 관측+잡음, `v_y` 미측정).
- 비교군 = 증강 KF 단독. Part 1 = MPC only / MPC+증강KF / MPC+GP 3-way.
- 추정기 모델이 선형 명목이라 **엄밀히는 EKF가 아니라 LTV KF**. 정직히 그렇게 명명.
  단 실차(비선형 측정) 대비 전방호환 구조로 짠다(평균함수+야코비안함수 주입).

Phase 5 산출물이 각 결정에 어떻게 연결되는지만 적어둔다:

| Phase 5 산출물 | 이걸로 정해지는 것 |
|---|---|
| 채널별 잔차 RMS 분포 | EKF 증강 상태를 무엇으로 둘지 |
| `(v_y, gamma, delta)` 산점도의 구조 | GP 입력 축 선택 (`v_y` vs `beta`, `v_x` 포함 여부) |
| 잔차 커버리지의 빈 영역 | dictionary 크기 `M` 후보 범위 |
| SNR vs `a_y_max` 곡선 | 학습 데이터 수집 시나리오 |
| solve time 여유분 | GP 프레임워크 (GPyTorch vs 직접 구현) |

**solve time 예산 산정 주의**: GP 결합이 이산 형태이므로 GP 평가는 RK4 바깥에서
일어난다. solve당 커널 평가는 **`N * M`** 회다 (`4*N*M` 이 아니다).
`M` 상한을 여기서 4배 낮게 잡지 마라.

---

## Phase 6 — 증강 KF 비교군 (Part 1)

```
Phase 6을 진행한다. Part 1의 비교군인 증강상태 칼만필터를 만든다.
먼저 .claude/rules/ekf-baseline.md 를 Read 해라. 공정성 요구가 거기에 있다.

## 명명 — 정직하게
파일·모듈 이름은 ekf.py 로 두되, 이 추정기는 명목(선형) 모델을 쓰므로 예측·측정
야코비안이 상수다. 즉 엄밀히는 EKF가 아니라 **외란증강 LTV 칼만필터(KF)**다.
주석·리포트에 이 사실을 명시하라. 없는 것을 EKF라 부르지 마라.
(선형계에서 KF는 최적 추정기이므로, KF를 쓰는 것이 비교군을 약화하는 게 아니라
오히려 최강 형태를 주는 것이다 — ekf-baseline.md 공정성.)

## 프레이밍 (코드 주석·리포트도 이대로)
KF는 상태추정기, GP는 모델 보정자다. Part 1은 "모델 오차와 그 불확실성을 다루는
두 접근"의 비교다. 추종오차 승부가 아니라 불확실성의 종류(KF 시간축 전파 vs
GP 상태공간 지도)를 보이는 것이 목적. 이 구분이 흐려지는 서술을 만들지 마라.

## 증강 구조 (Phase 5 근거)
Phase 5에서 모델 오차는 v_y, gamma 동역학에만 실림이 확인됐다(잔차 SNR).
증강상태 x_aug = [v_y, gamma, e_psi, e_y, d_vy, d_gamma]  (6)
  vy_dot    = f_nom_vy(x,u,vx)   + d_vy
  gamma_dot = f_nom_gamma(x,u,vx)+ d_gamma
  기구학 2식 불변, d_vy_dot = d_gamma_dot = 0 (random walk, Q_kf 구동)
명목 우변은 재사용한다(복사 금지). dt 이산화는 MPC 예측과 같은 dt_ctrl.

## 측정 모델 (현실적 부분관측)
y = [gamma, e_psi, e_y] + 가우스 잡음.  v_y 는 미측정(추정 대상).
H 는 해당 3행. R_kf = 측정 잡음 공분산. 잡음 seed 는 케이스 간 동일.

## 전방호환 설계 — 필수
나중에 실차(비선형 측정)로 가면 EKF가 필요하다. 지금 KF를 그 특수경우로 짜라:
필터 코어(P 예측, 칸 이득 K, 갱신)는 평균함수와 야코비안을 **주입받는다**.
  predict_mean(x,u,vx,kappa,dt) / predict_jac(...) -> (x_pred, F)
  meas_mean(x) / meas_jac(x)    -> (y_pred, H)
선형 케이스는 F,H 가 상수인 특수경우일 뿐이다. 야코비안은 CasADi ca.jacobian 으로
자동 생성한다(수기 미분 금지). 이렇게 하면 실차 전환 시 모델 객체만 바꾸면 되고
필터 수식은 안 건드린다. (MPC 의 StepModel 주입과 같은 철학.)

## src/estimation/ekf.py
- AugmentedKF: predict(u,vx,kappa,dt), update(y) -> (x_hat, P). d_hat 접근자.
- 매 스텝 P 대각(6), innovation, x_hat, d_hat 반환·로깅.

## src/control/ 결합 (StepModel 주입, if 분기 없음)
- EKFStepModel(가칭 DisturbanceStepModel): extra_param_dim=2,
  step_sym = rk4_step(f_nom + [d_vy,d_gamma,0,0] 를 우변에 더함),
  extra_param_values() 는 KF 의 현재 d_hat 반환. (연속 외란이므로 RK4 안에서 더한다.
  이는 GP 의 이산 잔차 주입(RK4 밖)과 대비된다 — 단위가 다르다.)
- MpcBase / mpc_nominal 은 수정하지 않는다.

## 제어 통합 정책 (통제변수 고정)
세 케이스 모두 **완전상태 피드백**을 쓴다(초기상태 정보 통제). 차이는 모델 보정뿐:
  - MPC only: 보정 없음.
  - MPC + KF: KF 가 부분관측 잡음측정으로 d_hat 추정 -> 예측에 주입. P 로깅.
  - MPC + GP: GP mean 주입(Phase 7). 사후분산 로깅.
이렇게 하면 비교가 '상태추정 품질'이 아니라 '모델오차 보정·불확실성'에 집중된다.
(KF 의 상태추정 부분은 v_y 가 미측정이라 내부적으로 필요하지만, 피드백에는
완전상태를 균일 사용해 케이스 간 통제한다.)

## src/sim/runner.py 확장 (주입식 유지)
선택적 estimator 를 받는다. 있으면: 측정 생성(true+잡음) -> KF predict/update ->
d_hat 를 컨트롤러 StepModel 이 프리뷰에 실어 예측 보정. estimator 없으면 기존 경로.
runner 에 케이스 분기(if case==...)를 넣지 마라.

## configs/ekf/default.yaml
Q_kf(6 대각, 핵심은 d_vy/d_gamma 프로세스 잡음), R_kf(3 대각), P0.
튜닝 방식(수동 그리드/자동)과 튜닝 예산을 config·리포트에 명시(공정성).
가중치 이름은 MPC 와 절대 겹치지 않게: Q_kf, R_kf (mpc-solver.md 표기 충돌).

## configs/experiment/part1_*.yaml
part1_mpc_only / part1_kf / part1_gp. sedan + single_curve + mpc + (ekf) + 비선형 플랜트.
통제변수(경로·IC·seed·N·가중치·IPOPT·플랜트·a_y_max) 세 케이스 동일.
a_y_max 는 Phase 7 GP 와 반드시 같은 값(조립된 config diff 로 확인).

## tests/test_ekf.py
- 가관측성: (A_aug, H) 가 수치적으로 가관측인지 확인. (v_y 미측정+외란 증강이
  가관측이 아니면 d_hat 이 발산하거나 안 잡힌다 — 여기서 멈추고 보고하라.)
- 잡음 없는 극한에서 d_hat 이 실제 잔차(연속화)를 추종하는지.
- P 대각·innovation 이 매 스텝 로깅되는지. innovation 이 화이트에 가까운지.
- 선형이므로 KF≡EKF 인지 대조(야코비안이 상수인지) — 전방호환 구조 검증.

## 핵심 산출물 (Part 1 주 그림 준비)
GP 사후 std vs KF P 의 d-블록 대각을 나란히 비교할 수 있도록, KF 의
(x_hat, P, d_hat)를 궤적 위에서 로깅한다. KF 의 P 는 시간축 값이라 상태공간
지도가 안 된다는 점을 그림으로 보이는 게 논점.

가관측성·수렴·튜닝·공정성에 애매한 점이 있으면 코드 전에 질문해라.
```

**게이트**: `(A_aug, H)` 가관측. 잡음 없는 극한에서 `d_hat`이 실제 잔차 추종.
`P` 대각·innovation 로깅. KF 가 선형(야코비안 상수)임을 대조로 확인.

---

### Phase 7~10 로드맵 (프롬프트는 해당 Phase 착수 시)

- **Phase 6** — 위 「Phase 6 — 증강 KF 비교군」 프롬프트로 확정.
- **Phase 7** — offline GP. 학습 → CasADi 변환 → `mpc_gp` 결합.
  게이트: GPyTorch 예측과 CasADi 변환본 일치, 학습 데이터 밖에서 사후 std 증가.
  착수 시 결정: GP 입력 축(`v_x` 포함 여부), `M`, 프레임워크, 학습 시나리오.
- **Phase 8** — **Part 1 실험 실행**. MPC only / MPC+증강KF / MPC+GP.
  주 산출물은 추종오차가 아니라 **GP 사후 std vs KF `P` d-블록 대각의 공간 지도**와
  예측구간 커버리지다.
- **Phase 9** — online / sparse / sliding-window GP.
  주 산출물은 solve time 분포와 `dt_ctrl` 초과율. `M` 고정 원칙(`mpc-solver.md`)과
  하이퍼파라미터 고정 원칙(`gp-residual.md`) 준수.
- **Phase 10** — **Part 2 실험 실행** 및 통합 리포트.

### (선택) LIMO 하드웨어

시뮬레이션 결과가 나온 뒤에만 검토한다. 착수 전 필수 선행 작업:
LIMO의 `Cf`, `Cr`, `Iz`, `delta_max` 식별 실험. 현재 config 값은 전부 자릿수 추정이라
그대로 하드웨어에 쓰면 결과 해석이 불가능하다.
