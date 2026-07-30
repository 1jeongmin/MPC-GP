# 프롬프트 아카이브 — Phase 8 (Part 1 실험 실행 + UQ 캘리브레이션)

`prompts.md` 에서 분리된 완료 Phase 아카이브다(2026-07-30). 사용 규칙·현재
미결 결정은 `prompts.md`/`handoff.md` 를 봐라.

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
