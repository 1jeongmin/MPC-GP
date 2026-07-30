# 프롬프트 아카이브 — Phase 3~5 (명목 MPC · 폐루프 · 비선형 플랜트)

`prompts.md` 에서 분리된 완료 Phase 아카이브다(2026-07-30). 사용 규칙·현재
미결 결정은 `prompts.md`/`handoff.md` 를 봐라.

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

