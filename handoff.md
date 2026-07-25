# handoff

세션이 끝나거나 `/clear` 하기 전에 이 파일을 갱신한다. 아래 5개 섹션 구조를 유지하라.

---

## 오늘 한 일 (3줄)

- 2026-07-25 — **Phase 0·1·2·3 완료.** 전체 테스트 **26개 통과** (게이트 1·2·2b·3·5·6·8).
- Phase 3: 명목 MPC. `MpcConfig`, `Preview`+`StepModel`+`MpcBase`(multiple shooting
  +IPOPT, warm start, 수렴실패 fallback), `NominalStepModel`. 게이트 6 통과.
- 설계 확정: preview=Preview 데이터클래스, 동역학=StepModel 주입(이산 전이,
  GP 파라미터는 P 확장블록). dt_ctrl은 MpcConfig 아님 — SimConfig가 단일 소스.

## 측정값 (기준선 — 이후 잔차 해석에 사용)

**noise floor** (모델 완전일치, RK4(dt=0.02) vs DOP853, 1-step, sedan vx=15):

| 채널 | RMS | max |
|---|---|---|
| v_y [m/s]    | 9.57e-9  | 1.28e-8 |
| gamma [rad/s]| 9.09e-10 | 1.22e-9 |
| e_psi [rad]  | 1.84e-10 | 2.47e-10 |
| e_y [m]      | 5.67e-10 | 7.90e-10 |

- 진단 임계값 = 위 채널별 값의 10배 (gp-residual.md). Phase 4에서 잔차와 대조.
- 시상수: sedan tau≈0.126s (고유값 -7.93±3.04i, 감쇠진동),
  limo tau≈0.010s (고유값 -119,-100, 과감쇠). limo Iz는 미식별 근사값.
- `gamma_ss(sedan,15,0.02)=0.094675 rad/s` — 적분 대조 통과.

## 다음에 먼저 할 일

1. `prompts.md`의 **Phase 4 프롬프트** 입력 (다중레이트 폐루프 + 잔차 로깅).
   작업 전 `.claude/rules/sim-experiment.md`, `gp-residual.md` 를 먼저 Read.
2. `src/sim/runner.py`(플랜트 주입 가능, 100/50Hz), `logger.py`, `eval/metrics.py`,
   `viz/plots.py`, `scripts/run_sim.py`, `configs/experiment/baseline_matched.yaml`
   (path/mpc 참조 활성화).
3. **검증 핵심**: 모델완전일치(플랜트도 선형) 잔차가 Phase 1 noise floor와 같은
   자릿수여야 한다. 채널별 배율로 비교(직접 비교 금지). e_psi/e_y ≤ 10×floor.
4. 잔차는 MPC 예측과 **같은 함수**로 계산 (복사 금지), dt_ctrl 경계에서만.

## Phase 3 산출물 메모

- **solve time 평균 ~23ms > dt_ctrl 20ms** (cold Python/IPOPT, warm후 iters 11~15).
  실시간성은 Part 2 과제 — Phase 3 게이트(정합성)와 무관. 측정값으로만 보고.
- StepModel 주입 완료 → Phase 7 GP는 MpcBase/mpc_nominal 수정 없이 붙는다.
  GP StepModel: extra_param_dim>0, step_sym=rk4+B_d@mu_GP, P 확장블록에 Z/alpha.
- warm start = 이전 해 shift. 수렴실패 fallback = 이전 해 shift(converged=False 기록).

## Phase 2 산출물 메모

- 세그먼트 스키마 = {length, kappa_end}. 시작곡률=이전끝(0에서 출발) → C0 연속 자동.
- dlc 최대 횡변위 8.5m (실차선보다 큼) — Phase 5에서 잔차 SNR 보고 재조정 예정.
- `Reference(path_cfg, vx_range)` — vx_range는 차량 config에서 주입 (경로 속성 아님).

## 절대 하지 말 것

- 게이트 1 실패 상태에서 MPC 구현 착수.
- GP 보정을 연속 우변(`xdot`)에 더하기 — `dt_ctrl`배 틀린다. 이산 결합만.
- 4x4 전체 `A`에 Hurwitz 검사 걸기 — 원점 고유값 2개는 정상이다.
- 테스트 허용오차를 늘려서 게이트 통과시키기.
- `src/`의 스텁 파일 삭제 (룰 자동 로드 트리거용).
- `results/`, `data/` 커밋.

## 미결 결정 (룰 파일에 "(제안)"으로 표시된 항목)

확정되면 해당 룰 파일에서 "(제안)" 표기를 지우고 여기서도 지운다.

| # | 항목 | 위치 | 상태 |
|---|---|---|---|
| 1 | EKF 비교군을 **외란/바이어스 증강 상태 EKF**로 할 것인가 | `ekf-baseline.md` 「공정성 요구」 1 | 미결 |
| 2 | EKF `P` 대각 성분을 매 스텝 로깅 (Part 1 주 그림 전제) | `ekf-baseline.md` 「핵심 산출물」 | 미결 |
| 3 | inducing point 개수 `M` 고정 — 실제 solve time 이득은 프로파일링 필요 | `mpc-solver.md` 「NLP 구조 재생성 금지」 | 미결 (프로파일링 대기) |

확정된 결정 (되돌리려면 사용자 확인 필요):

- **GP 결합 = 이산 형태.** `x_{k+1} = rk4(f_nom) + B_d @ mu_GP`. GP 평가는 solve당 `N`회.
- **온라인 GP 하이퍼파라미터는 고정.** `Z`, `alpha`만 갱신. 재학습은 config 옵트인.
- **진단 채널 임계값 = `noise_floor`의 10배.**

## 참고 경로

- **작업 프롬프트: `prompts.md`** (Phase 0~5. Phase 6 이후는 Phase 5 숫자를 보고 작성)
- 룰: `.claude/rules/` (7개 — 모듈 건드리기 전에 해당 파일 먼저 Read)
- 게이트 정의: `.claude/rules/testing.md`
- 부호 규약 단일 기준: `.claude/rules/vehicle-model.md`
- 실험 조합: `configs/experiment/part1_*.yaml`, `part2_*.yaml` (미작성)
