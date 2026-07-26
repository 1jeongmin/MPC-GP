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

**solve time 예산 산정 주의** ~~solve당 커널 평가는 `N * M` 회다~~
→ **Phase 7 진단으로 폐기됨.** 결합이 지평 상수(작동점 1회 평가)로 바뀌어 GP 평가는
**solve당 1회**, 커널 평가는 `M` 회다. 따라서 `M` 은 solve time 과 **무관**하고 적합
품질만 결정한다. 위 문단은 상태의존 결합을 전제한 옛 산정이므로 그대로 쓰지 마라.
(현행 기준: `mpc-solver.md` 「GP 결합 시」, `gp-residual.md` 「CasADi 결합」)

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

## 측정 모델 (전상태 — Phase 6 진단으로 확정)
y = [v_y, gamma, e_psi, e_y] + 가우스 잡음 (전상태). H = I4 해당 행.
R_kf = 측정 잡음 공분산(4). 잡음 seed 는 케이스 간 동일.

주의: 처음엔 현실적 부분관측(v_y 미측정)으로 뒀으나, 그 경우 외란 d_vy 가 e_y 로부터
이중적분 뒤에야 관측되어 지연·과대추정되고 **MPC+KF 가 MPC-only 보다 나빠졌다**.
논문 논지("KF 도 MPC 를 돕는다, 단 GP 가 더")가 성립하려면 KF 가 단독 MPC 를 이겨야
한다. 논문의 초점은 상태추정이 아니라 **모델오차 처리**이므로, 세 방법 모두 상태
정보를 갖게 하고 KF 는 깨끗한 외란관측기가 되도록 전상태 측정으로 확정했다. 이러면
MPC+KF 가 MPC-only 를 이긴다(오라클 상한 +7.3% 근접). 오라클(참 순간외란 상수외삽)이
MPC-only 를 이기는 것으로 개념 정상을 확인했고, GP 는 외란을 지평에서 d(x) 로 예측해
이 상수외삽 상한을 넘는다 — 이것이 GP 우위의 근거다.

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
**MPC+KF 가 MPC-only 를 이긴다** (test_kf_beats_mpc_only).

---

## Phase 7 — offline GP (Part 1)

```
Phase 7을 진행한다. offline GP 잔차 학습 -> CasADi 변환 -> mpc_gp 결합.
먼저 .claude/rules/gp-residual.md 를 Read 해라.

## 확정 결정 (2026-07-25, Phase 5 특성화·v_x 민감도 분석 근거)
- 입력 z = [v_y, gamma, delta] (3D). v_x 제외: 잔차 유의구간에서 v_x 가 변하지만
  슬립각 민감도 d(alpha_f)/d(vx)=(vy+a*gamma)/vx^2 ~= 5e-4 rad/(m/s) 라 v_x 의
  독립 기여가 슬립각 0.04°(전체 1.3~3.7° 대비 무시) 수준. 차원 최소로 UQ 논증 깔끔.
- 채널 = v_y, gamma 독립 GP 2개. e_psi, e_y 는 학습 안 함(기구학, 진단만).
- 프레임워크 = 직접 numpy/scipy exact GP (torch 미사용).
- 커널 = ARD RBF. 하이퍼파라미터 = Type-II ML (log-det 포함 표준 NLML).
- 결합 = 이산: x_{k+1} = rk4(f_nom) + B_d @ mu_GP(z), **RK4 밖** (Phase 1 확정).

## src/gp/dataset.py
잔차 데이터셋. 특징 z=[vy,gamma,delta], 타깃 [r_vy, r_gamma] (명목 대비 잔차).
- 학습 궤적과 평가 궤적을 분리한다 (같은 주행 로그로 학습·평가 동시 금지).
  제안: 학습=single_curve a_y=4, 평가=a_y=6 및 dlc (분포 밖 -> 사후분산 증가 시연).
- 표준화 통계(z_mean/std, r_mean/std) 계산·저장, 추론 시 재사용.
- 데이터셋에 생성 config/git hash/seed 동봉 (data/ 아래, git 제외).

## src/gp/kernels.py
ARD RBF. numpy 경로 + CasADi 경로. 같은 입력에 두 경로 일치 대조.

## src/gp/train_offline.py
- 딕셔너리 Z = 학습 궤적에서 선택한 M개 점 (config M, 기본 균일 subsample).
  M 고정 (mpc-solver.md, Part 2 sparse 와 짝). 제대로 된 sparsification 은 Phase 9.
- exact GP: K = RBF(Z,Z) + sigma_n^2 I, alpha = solve(K, y_std).
- Type-II ML: NLML = 0.5 y^T alpha + 0.5 logdet(K) + 0.5 n log(2pi).
  **log-det 생략 금지** (gp-residual.md). scipy.optimize 로 하이퍼파라미터 최적화.
- 2채널 독립 학습. Z/alpha/lengthscale/sigma_f/sigma_n/표준화통계 저장.

## src/gp/casadi_export.py
mu(z) = k(z_std, Z) @ alpha (표준화 공간 계산 후 역표준화해 r 반환).
Z/alpha/하이퍼파라미터/표준화통계를 CasADi 파라미터로 주입.
**numpy GP 예측과 CasADi 예측이 일치하는지 대조 (필수).**

## src/control/mpc_gp.py  (⚠ Phase 7 진단으로 결합 형태 정정 — 아래 「정정」 참조)
GPStepModel: 현재 작동점 상수 결합. extra_param_dim=2 (mu_hat).
  set_operating_point(x0,u_prev): mu_hat = mu_GP([x0[0],x0[1],u_prev]) 를 CasADi 표현으로
    1회 평가·저장 (MpcBase.solve 훅이 호출).
  step_sym = rk4_step(f_nom) + B_d @ mu_hat (지평 상수). **RK4 밖**.
  (KF 의 연속 외란(RK4 안)과 대비 — GP 잔차는 이산 상태차이 단위.)
  extra_param_values() = mu_hat. MpcBase/mpc_nominal 무수정(작동점 훅만 추가).

## 사후분산 (제어에 넣지 않되 반드시 저장)
mean-only 제어. 분산을 제약·비용에 전파하지 마라. 그러나 매 스텝
var(z) = k(z,z) - k(z,Z) K^-1 k(Z,z) 를 계산·로깅한다 (논문 주 증거물). 끄지 마라.

## configs/gp/default.yaml, configs/experiment/part1_gp.yaml
part1_gp 는 part1_mpc_only/part1_kf 와 통제변수(경로·IC·seed·N·가중치·IPOPT·
플랜트·a_y) 동일. 조립된 config diff 로 확인.

## tests/test_gp.py — 게이트 7
- 잔차 0 데이터로 학습 -> GP 평균 ~0, 분산 ~사전분산 수준.
- 학습 데이터에서 멀어질수록 사후 std 가 단조 증가 (핵심 UQ 주장. 실패하면 멈춤).
- numpy GP 예측과 CasADi 변환본 예측 일치.
- 표준화 통계 저장·재적용 시 예측 동일.
- 추가 게이트: MPC+GP 추종 < MPC+KF (GP 가 KF 를 이긴다. Phase 6 방식 재사용).

학습 데이터 수집·궤적 분리·M 선택에 애매하면 코드 전에 질문해라.
```

**게이트**: testing.md 게이트 7 전부 + 사후 std 단조 증가 + numpy↔CasADi 일치 +
**MPC+GP < MPC+KF < MPC-only** (추종오차).

### Phase 7 진단 정정 (2026-07-26) — 결합은 상태의존이 아니라 지평 상수

프롬프트 초안은 mu_GP(z_k)를 각 지평 예측상태에서 평가(상태의존 결합)했으나, 그러면
mean-only MPC 가 GP 부정확 영역을 최적화에서 악용해 **불안정**해진다(격자 완벽적합에도
-74%). 판별: 이산 오라클(참 1스텝잔차 상수) +13%, 상수-GP +12%, 상태의존-GP -45~-74%.
→ **GP 평균을 현재 작동점에서 1회 평가해 지평 상수로 주입**한다. 결과: 전 M 에서
안정적으로 MPC+GP +12% > MPC+KF +5.8% > MPC-only. GP 가 KF 를 이기는 이유도 재정립:
지평예측이 아니라 상태->외란 직접학습이라 현재상태에서 즉시 정확(KF는 필터 지연).
정칙화(sigma_n_floor)로 보간 과적합도 막는다. 룰 반영: gp-residual.md, mpc-solver.md.

---

## Phase 8 — Part 1 실험 실행 + UQ 캘리브레이션

```
Phase 8을 진행한다. Part 1(MPC only / +KF / +GP) 3-way 를 재현 가능한 실험으로
실행하고, 주 산출물인 UQ 캘리브레이션과 불확실성 공간지도를 만든다.

먼저 .claude/rules/sim-experiment.md 와 .claude/rules/gp-residual.md 를 Read 해라.
지표·로깅 스키마·재현성·플롯 규약이 전부 거기에 있다.

## 이 Phase 의 성격 (Phase 7 결과에서)
추종 3-way 승부는 이미 테스트에서 확인됐다. 다시 만드는 게 목적이 아니다.
Phase 7 정정으로 GP 와 KF 는 둘 다 "지평 상수 외란" 을 주입하므로 제어 구조가
같아졌다. 따라서 차별점은 추종 승부가 아니라 **불확실성의 종류**
(GP = 상태공간 지도 / KF = 시간축 스칼라)이고, 이 Phase 는 그것을 숫자와 그림으로
**검정**하는 단계다. 증명하는 단계가 아니다 — 결과가 논지에 불리하면 그대로 보고하라.

## 0. 먼저 막을 구멍 — GP 사후분산이 로깅되지 않는다
src/sim/runner.py 는 ekf_* 만 로깅하고 gp_mean/gp_var 를 로깅하지 않는다.
GPStepModel 은 self.gp 를 "분산 로깅용" 으로 들고만 있고 predict_var 를 아무도
호출하지 않는다. sim-experiment.md 로깅 스키마와 gp-residual.md 「사후분산은 반드시
저장한다」 위반이며, 이것이 논문의 주 증거물이다.

runner 에 케이스 분기를 넣지 말고 고쳐라: 컨트롤러가 선택적으로 스텝별 로그를
내놓는 훅(예: controller.step_log() -> dict | None)을 두고, runner 는 있으면
log.append 에 합류시킨다. estimator 훅과 대칭 구조. GPStepModel.set_operating_point
이 mu_hat 과 함께 var(z_now) 도 계산해 보관한다.

## 1. scripts/run_sim.py 를 케이스 무관 실행기로 (현재는 Phase 4 유물)
make_nominal_mpc + 선형 플랜트가 하드코딩되어 part1_kf / part1_gp 를 못 돌린다.
experiment config 로부터 컨트롤러·플랜트·추정기를 조립하게 고쳐라.
**if case == "..." 금지** (CLAUDE.md). 조립 규칙 제안: experiment 에 존재하는
그룹으로 결정(gp 그룹 -> GP 케이스, ekf -> KF, 둘 다 없으면 명목), 플랜트는
experiment 의 plant 키로 주입. 이 규칙이 부적절하면 코드 전에 물어라.

## 2. scripts/run_part1.py — 3-way 배치 래퍼
part1_mpc_only / part1_kf / part1_gp 를 하나의 run_id 우산 아래 돌린다.
- 통제변수 검증을 **자동화**하라. 조립된 세 config 를 diff 해서 vehicle/path/mpc/
  sim/seed/플랜트/a_y 가 동일하지 않으면 **실행을 거부**한다. 눈으로 확인 금지.
- results/<run_id>/<case>/ 별 로그·메타 + 최상위 비교 요약(CSV/JSON).

## 3. src/eval/calibration.py — Part 1 주 지표 (현재 스텁)
대상은 **1스텝 잔차 r_k** 다. 두 방법이 같은 대상을 예측하는지 리포트에 명시하라.
다른 대상을 비교하면 캘리브레이션 수치는 아무 의미가 없다.
- GP: 표준화 잔차 (r - mu_GP)/sigma_GP 의 분포, 68/95% 구간 실제 포함률, NLPD.
- KF: 같은 대상에 대한 KF 예측분포로 동일 지표. **어느 분포를 KF 의 예측구간으로
  볼지는 미결이다(handoff.md 미결 표). 임의로 정하지 말고 코드 전에 물어라.**
- 부수로 KF innovation 의 자기일관성(S = H P^- H^T + R_kf 대비 정규화)도 낸다.
  이건 KF 가 자기 모델 안에서 일관적인지 보는 것이지 GP 와의 비교가 아니다.

## 4. 분포이동 시나리오 — UQ 논증의 본체
GP 는 single_curve a_y=4 로 학습됐다(Phase 7). 평가는 학습 밖에서 한다:
  (a) in-distribution : single_curve a_y=4
  (b) 강도 외삽       : single_curve a_y=6
  (c) 형상 외삽       : dlc
세 시나리오 × 세 케이스 = 9 런. configs/experiment/ 에 조합으로 표현하고
스크립트에 하드코딩하지 마라.
기대: (b)(c)에서 GP 사후 std 가 커지고 커버리지가 유지되면 UQ 주장 성립.
커버리지가 무너지면 그 사실을 그대로 보고하라. **결과가 나쁘다고 학습 데이터를
몰래 늘리지 마라** (gp-residual.md 데이터 위생).

## 5. 주 그림 — GP 사후 std vs KF P d-블록
- (v_y, gamma) 평면 위 GP 사후 std 등고선 + 학습점 산점 + 세 시나리오 궤적 오버레이.
  "GP 는 안 가본 영역을 안다" 를 한 장으로 보이는 그림. delta 축은 고정 단면으로
  자르고 어느 값인지 캡션에 적어라.
- 같은 궤적의 KF P d-블록 대각을 시간축으로 그린다. 이것이 상태공간에 올라가지
  않는다는 것이 논점이다. 억지로 같은 축에 올리지 말고 대비가 보이게 배치하라.
- 케이스 색·선종 매핑은 config 에 저장 (sim-experiment.md 플롯 규약).

## 6. 공정성 — 튜닝 예산
KF(Q_kf, R_kf)와 GP(하이퍼파라미터)의 튜닝 예산을 같게 맞추고 그 사실을 config 와
리포트에 적는다. 튜닝은 (a) in-distribution 시나리오만 보고 하고, (b)(c) 결과를
본 뒤 어느 쪽도 사후에 손보지 마라 (ekf-baseline.md 공정성).
예산 정의는 미결이다(handoff.md 미결 표).

## 금지
- runner / MpcBase 에 케이스 분기 추가.
- 캘리브레이션이 안 맞는다고 사후분산을 스케일링해서 맞추기.
- 세 케이스의 통제변수를 하나라도 다르게 두고 "사소하다" 며 넘어가기.
- 9 런 중 잘 나온 것만 리포트에 싣기.

## tests/test_part1.py
- 조립된 세 config 의 통제변수 동일성 (2번 자동검증의 단위테스트).
- gp_var 가 매 스텝 로깅되고 전부 양수인지.
- 학습영역 안 < 밖 의 평균 사후 std (분포이동이 실제로 잡히는지).
- 커버리지 계산기 자체 검증: 참 N(0,1) 합성표본에서 95% 근처가 나오는지.

캘리브레이션 대상 정의, 튜닝 예산, 시나리오 조합에 애매하면 코드 전에 질문해라.
```

**게이트**: `gp_var` 전 스텝 로깅(양수) · 통제변수 자동검증 통과 ·
학습영역 밖에서 GP 사후 std 증가 · 커버리지 계산기 합성분포 검증 ·
9런 전부 `results/`에 재현성 메타 동봉.

---

### Phase 8~10 로드맵 (프롬프트는 해당 Phase 착수 시)

- **Phase 6** — 위 「Phase 6 — 증강 KF 비교군」 프롬프트로 확정.
- **Phase 7** — 위 「Phase 7 — offline GP」 프롬프트로 확정.
- **Phase 8** — 위 「Phase 8 — Part 1 실험 실행 + UQ 캘리브레이션」 프롬프트로 확정.
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
