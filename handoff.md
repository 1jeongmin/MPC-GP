# handoff

세션이 끝나거나 `/clear` 하기 전에 이 파일을 갱신한다. 아래 5개 섹션 구조를 유지하라.

---

## 오늘 한 일 (3줄)

- 2026-07-24 — `CLAUDE.md` + `.claude/rules/*` 7개 룰 파일 검토. 차량 모델 수식
  (A/B/E, `K_us` 항등식, `det(A_lat)` 판별식)은 손유도로 대조해 **전부 일치 확인**.
- GP 결합 형태의 단위 불일치를 **이산 결합(a안)으로 확정**하고 `gp-residual.md`,
  `mpc-solver.md` 양쪽을 정합시킴.
- `git init`, `.gitignore`, `requirements.txt`, 폴더 구조 + 모듈 스텁 생성.
  Phase별 프롬프트를 룰 변경에 맞춰 정정해 `prompts.md`로 저장. 코드 구현은 **아직 0줄**.

## 다음에 먼저 할 일

0. `prompts.md`의 **Phase 0 프롬프트**를 그대로 입력한다.
   (Phase -1 저장소 골격은 완료됨 — 다시 실행하면 이 파일이 덮어써진다.)
1. `configs/vehicle/sedan.yaml` 작성 (물리 파라미터 + 근거 주석).
2. `src/models/linear_bicycle.py` — numpy/CasADi 이중 경로 구현.
   **작업 전 `.claude/rules/vehicle-model.md`를 먼저 Read 할 것.**
3. `testing.md` 게이트 1(정상상태 요레이트 1% 이내) 통과시키기.
   **여기를 통과하지 못하면 MPC로 넘어가지 않는다.**
4. 게이트 2b·3·4, 그리고 게이트 5(noise floor 숫자 산출)까지.

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
