# GP-MPC 차량 횡방향 제어 — 프로젝트 지침

## 이 저장소의 범위

명목 **선형 동역학 자전거 모델(오차좌표계)** 위에 **GP 잔차 보정**을 얹은 횡방향
경로추종 MPC를 **Python + CasADi**로 구현하고 시뮬레이션으로 검증한다.

이 저장소는 **구현 코드 전용**이다. 논문 원고, 강의 번역 노트, 문헌 조사 결과는
여기에 두지 않는다. 코드가 아닌 산출물을 만들라는 요청이 오면 먼저 되물어라.

## 최소 정의 (상세는 룰 파일에)

- 상태 `x = [v_y, gamma, e_psi, e_y]^T` — 횡속도, 요레이트, 헤딩오차, 횡방향오차
- 입력 `u = delta` — 전륜 조향각
- 외생신호 `v_x`(종속도), `kappa`(경로곡률) — **결정변수 아님.** 프리뷰 파라미터로만 진입
- 명목 모델 `xdot = A(v_x) x + B(v_x) u + E(v_x) kappa`
- 행렬 성분·부호 규약·유도는 → `.claude/rules/vehicle-model.md`

## 실험 구조 (코드 구조를 결정하는 축)

명목 모델은 **선형 동역학 자전거로 고정**한다. 바뀌는 것은 보정 방식뿐이다.

- **Part 1 — GPR vs EKF (UQ 논증)**
  ① MPC only(보정 없음) ② MPC + EKF ③ MPC + GP
- **Part 2 — GP 실시간화 진행**
  ① offline GP ② online GP ③ online sparse GP ④ online sliding-window GP

따라서 `models / control / estimation / gp / sim / eval`은 서로 독립 교체 가능해야 한다.
**새 케이스를 추가할 때 기존 모듈을 수정하지 말고 주입(dependency injection)으로 해결하라.**
if/else로 케이스 분기를 늘리는 패턴은 금지한다.

## 폴더 구조

```
CLAUDE.md
handoff.md                 # 세션 인계 + 미결 결정 추적 (미결 결정의 단일 소스)
prompts.md                 # Phase별 작업 프롬프트. 규약을 여기에 복제하지 마라
requirements.txt
.gitignore                 # results/, data/ 제외
.claude/rules/             # 경로별 조건부 룰 (평평하게 유지, 하위폴더 금지)
configs/                   # 그룹 디렉토리 방식 (Hydra 호환 레이아웃)
  experiment/              # 조합 파일. part1_*.yaml, part2_*.yaml
  vehicle/                 # sedan.yaml, limo.yaml
  path/  mpc/  gp/  ekf/  sim/
src/
  config.py
  models/                  # linear_bicycle, nonlinear_bicycle, tire, integrators
  path/                    # reference (kappa(s) 프로파일, 프리뷰)
  control/                 # mpc_base, mpc_nominal, mpc_gp
  estimation/              # ekf
  gp/                      # dataset, kernels, train_offline, online, sparse,
                           #   sliding_window, casadi_export
  sim/                     # runner(다중레이트), logger
  eval/                    # metrics, calibration
  viz/                     # plots
scripts/                   # run_openloop.py, run_sim.py(범용 실행기),
                           #   run_part1.py / run_part2.py (run_sim 배치 래퍼)
tests/
data/                      # GP 학습·검증 데이터셋 (git 제외)
results/                   # run_id별 산출물 (git 제외)
```

## 절대 바꾸지 않는 것 — 바꾸려면 먼저 물어볼 것

1. 상태·입력·부호 규약. `vehicle-model.md`가 유일한 기준이다.
2. `v_x`, `kappa`를 상태로 승격하지 않는다. LTV 구조를 유지해야 나중에 QP 전환
   가능성을 논할 수 있다.
3. **오차 기구학(`e_psi`, `e_y` 식)은 건드리지 않는다.** 잔차의 출처를 타이어
   비선형성 하나로 통제하기 위한 실험 설계다.
4. GP는 **동적 상태 잔차(`v_y`, `gamma`)만** 학습한다. `e_psi`, `e_y` 채널은 학습 대상이
   아니라 검증용 진단 채널이다.
5. GP는 **평균 전용(mean-only, non-cautious)**. 분산을 제어에 전파하는 stochastic
   MPC는 이번 스코프 밖이다. 단, 사후분산은 반드시 계산·저장한다(논문의 핵심 주장).
6. 물리 파라미터·가중치 **하드코딩 금지**. 전부 `configs/`에서 읽는다.

## 작업 방식

- **95% 확신 룰**: 작업을 시작하기 전에 95% 확신이 들 때까지 추가 질문을 하라.
  확신이 안 서면 코드를 작성하지 마라.
- **Phase 게이트**: 검증 테스트가 실패하면 다음 단계로 넘어가지 말고 원인부터 보고하라.
  특히 정상상태 요레이트 해석해 대조(1% 이내)를 통과하지 못하면 MPC 작업을 시작하지 마라.
- 여러 Phase를 한 번에 몰아서 구현하지 마라. 검증되지 않은 코드가 쌓인다.
- 파일·설정·의존성을 바꾸는 큰 작업 전에는 plan mode로 계획을 먼저 보여라.
- IMPORTANT: 세션이 끝나거나 `/clear` 하기 전에 `handoff.md`를 갱신하라
  (오늘 한 일 3줄 / 다음에 먼저 할 일 / 절대 하지 말 것 / 미결 결정 / 참고 경로).
- 룰 파일에 `(제안)`으로 표시된 항목은 **아직 확정이 아니다.** 그 항목에 의존하는
  코드를 쓰기 전에 사용자에게 확인받고, 확정되면 `handoff.md`의 「미결 결정」
  표에서 옮겨라.

## 답변 규약

- 한국어로 답한다. 수식·기호는 코드와 논문 표기를 일치시킨다.
- 모르면 "모르겠습니다", 추측이면 "(추측)", 출처가 불분명하면 "확실하지 않음"으로
  명시한다. 단정하지 말고 근거를 함께 제시하라.
- 문헌을 인용하면 **검증된 DOI만** 붙인다. 검증 못 했으면 "DOI 확인 안 됨"이라 적어라.
- 없는 함수·API·논문·수치를 지어내지 마라. 모르는 라이브러리는 먼저 문서를 확인하라.
- 물리적으로 이상한 결과가 나오면 숫자를 포장하지 말고 그대로 보고하라.

## 상황별 참조 인덱스

경로 기반 자동 로드는 `.claude/rules/*.md`의 `paths`가 처리한다. 아래는 **코드를
만지기 전에 설계만 논의할 때** 직접 열어볼 안내다.

> **중요 — 룰 자동 로드는 파일을 `Read`할 때만 걸린다.** 새 파일을 만들거나
> 설계만 논의할 때는 룰이 컨텍스트에 없다. 따라서 어떤 모듈을 건드리기 전에
> **아래 표에서 해당 룰 파일을 먼저 `Read`하라.** `src/`의 스텁 파일들은 이
> 트리거를 확보하려고 미리 만들어 둔 것이므로 지우지 마라.
> 세션 중 `/context`로 실제 로드된 룰을 확인할 수 있다.

| 상황 | 파일 |
|---|---|
| 차량 모델·부호 규약·적분기 | `.claude/rules/vehicle-model.md` |
| MPC 정식화·솔버·warm start | `.claude/rules/mpc-solver.md` |
| GP 잔차 정의·CasADi 결합·희소화 | `.claude/rules/gp-residual.md` |
| EKF 비교군 설계·공정성 | `.claude/rules/ekf-baseline.md` |
| 실험 프로토콜·지표·로깅·플롯 | `.claude/rules/sim-experiment.md` |
| 검증 테스트 기준 | `.claude/rules/testing.md` |
| 코딩 규약·단위·config | `.claude/rules/python-conventions.md` |

## 금지

- `results/`, `data/` 산출물을 git에 커밋하지 않는다.
- seed·config 스냅샷·git hash 없이 `results/`를 만들지 않는다. 재현 불가능한 실험은
  실험이 아니다.
- 벤치마크 수치를 추정으로 채우지 않는다. 측정하지 않았으면 측정하지 않았다고 적어라.
- 비교군(EKF)을 일부러 약하게 만들지 않는다. `ekf-baseline.md` 참조.
