# 프롬프트 아카이브 — Phase -1~2 (저장소 골격 · config · 참조 경로)

`prompts.md` 에서 분리된 완료 Phase 아카이브다(2026-07-30, 파일 크기 때문에 분리).
사용 규칙·현재 미결 결정은 `prompts.md`/`handoff.md` 를 봐라.

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

