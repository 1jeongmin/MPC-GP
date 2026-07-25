# handoff

세션이 끝나거나 `/clear` 하기 전에 이 파일을 갱신한다. 아래 5개 섹션 구조를 유지하라.

---

## 오늘 한 일 (3줄)

- 2026-07-25 — **Phase 0 완료.** config 로딩 체계(`src/config.py`: VehicleConfig,
  SimConfig, ExperimentConfig + 조합 로더 + 재현성 스냅샷)와 선형 자전거 모델
  (`src/models/linear_bicycle.py`: numpy/CasADi 이중 경로 + 해석 검증 함수) 구현.
- `configs/vehicle/{sedan,limo}.yaml`, `configs/sim/default.yaml`,
  `configs/experiment/baseline_matched.yaml` 작성. casadi/scipy/pytest/pyyaml 설치.
- `tests/test_model.py` 게이트 2b·3·4 **10개 전부 통과**.

## 다음에 먼저 할 일

1. `prompts.md`의 **Phase 1 프롬프트** 입력 (적분기 + 개루프 물리 검증).
   작업 전 `.claude/rules/vehicle-model.md`, `testing.md` 를 먼저 Read.
2. `src/models/integrators.py` (RK4 + 고정밀), `tests/test_integrator.py`
   게이트 1·2·5 구현.
3. **게이트 1(정상상태 요레이트 1% 이내)** 통과가 최우선. 실패 시 MPC로 넘어가지 않는다.
   기준값(적분 대조용): `gamma_ss(sedan, vx=15, delta=0.02) = 0.094675 rad/s`
   (해석해. Phase 1에서 시간적분 결과와 대조).
4. 게이트 5 noise floor는 **채널별로** 숫자 출력하고 이 파일에 기록.

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
