# handoff 로그 — Phase 0~7 (2026-07-25 이전)

`handoff.md` 를 3개로 분리한 것 중 하나다(2026-07-30, 파일이 커져서 분리).
**현재 상태·미결 결정은 여기 없다** — `handoff.md` 를 봐라. 이 파일은 Phase 0~7
동안 쌓인 산출물 메모·확정 결정·기준선 수치의 **이력**이다.

---

## Phase 5 특성화 결과 (GP 설계 입력)

**sedan mismatch, a_y_max=4.0, single_curve, 잔차 SNR = RMS/noise_floor:**

| 채널 | 잔차 RMS | SNR | 비고 |
|---|---|---|---|
| v_y   | 1.30e-2 | ×1.36M | GP 학습 대상. 강한 신호 |
| gamma | 1.21e-3 | ×1.33M | GP 학습 대상 |
| e_psi | 1.26e-5 | ×68k | = 0.5·dt·r_gamma (종속) |
| e_y   | 1.33e-4 | ×235k | = 0.5·dt·r_vy (종속) |

**a_y_max 스윕 (SNR 단조 증가, 슬립각으로 타이어 작동영역 확인):**

| a_y | SNR(v_y) | SNR(gamma) | max\|α_f\| |
|---|---|---|---|
| 1.0 | ×70k | ×53k | 0.74° |
| 4.0 | ×1.36M | ×1.33M | 3.68° |
| 6.0 | ×3.86M | ×4.87M | 6.85° |
| 7.0 | ×6.56M | ×13.0M | 9.99° |

(alpha_sl≈15.4°. a_y=4~6이 비선형-비포화 영역, GP 실험 후보.)

- **limo**: single_curve에서 1 m/s 크롤 → 20s에 s=20m(커브 진입 못함), 슬립 0°,
  타이어 선형 → GP 신호 없음. limo 전용 경로 + 파라미터 식별 필요(미룸).
- 산점도: `results/mismatch_sedan_*/…_scatter.png` — (v_y,gamma,delta) 커버리지.

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

## Phase 7 산출물 메모

- **결합 = 현재 작동점 상수 주입** (상태의존 지평결합 금지, mean-only 불안정).
  GPStepModel.set_operating_point이 mu_hat=mu_GP(z_now) 1회 평가(CasADi), 지평 상수.
  MpcBase.solve에 작동점 훅 추가(hasattr). M은 solve time 무관, 적합품질만.
- **GP>KF 재정립**: 지평예측 아님. 상태→외란 직접학습이라 즉시 정확(KF는 필터지연).
- 정칙화 sigma_n_floor=1e-2 필수 (없으면 보간 과적합→궤적밖 진동).
- 입력 3D[vy,gamma,delta], 채널2 독립, ARD RBF, Type-II ML(log-det). 직접 numpy GP.
- 분산은 제어 아님, post-hoc 로깅(mean-only). 사후 std 단조증가 확인됨(UQ 논증).
- **주의(Phase 8/분포이동)**: GP는 MPC-only 궤적으로 학습. 상수결합이라 강건하지만,
  더 공격적 시나리오(a_y=6/dlc)에선 외삽 영역↑ → 사후분산↑(UQ 시연 포인트).

## Phase 6 산출물 메모

- **측정 = 전상태 [v_y,gamma,e_psi,e_y]로 확정** (앞선 "부분관측" 뒤집음). 근거 아래.
- **왜 부분관측이 실패했나 (탐구 결론)**: d_vy 는 e_y 로부터 이중적분 뒤에야 관측
  → v_y 미측정이면 심한 지연·과대추정(d_hat 2.0 vs 참 0.65~1.3) → MPC+KF −18%.
  판별실험: 오라클(참 순간외란, 상수외삽) MPC = **+7.3%**(개념 정상). H3(참상태
  잔차 LPF) +9.2%. 부분관측 x_hat 피드백은 발산(v_y lag). → 전상태 측정 + Q_d↑
  = **+5.8%**(오라클 근접). max|e_y|는 −16%(전이 스파이크, Phase 8 튜닝).
- **논문 논리 유지**: KF 가 MPC 를 도움(+5.8%). GP 는 외란을 지평에서 d(x)로 예측해
  상수외삽 상한(+7.3%)을 넘을 것 → GP > KF > MPC-only.
- **StepModel 재확인**: KF(연속 d_hat, RK4 안) / GP(이산 mean, RK4 밖) 둘 다 수용.
- **전방호환**: ekf.py는 평균함수+야코비안 주입. 실차 비선형측정 시 process/meas
  함수만 교체(ca.jacobian 자동). KF≡EKF 테스트로 현재 선형 확인.
- **Part 8 주그림 데이터**: runner가 (ekf_x_hat, ekf_P_diag, ekf_d_hat, ekf_innovation)
  궤적 로깅. GP 사후 std vs KF P d-블록 비교 준비됨.
- **공정 튜닝은 Phase 8**: 현재 Q/R은 MPC+KF 가 이기는 정상값이나 GP 와 같은 예산의
  최종 튜닝은 Phase 8.

## 결정 기록 (Phase 6 확정 / Phase 7 미결)

**확정된 Phase 6 결정:**
- EKF 증강상태 = [v_y,gamma,e_psi,e_y,d_vy,d_gamma], 외란은 v_y·gamma에 (Phase 5 근거).
- 측정 = **전상태(v_y,gamma,e_psi,e_y)+잡음** (Phase 6 진단으로 부분관측에서 수정 —
  부분관측은 d_vy 관측지연으로 MPC+KF를 악화시킴). 측정 모델 선형.
- 비교군 = 증강 KF 단독. Part 1 = MPC only / MPC+KF / MPC+GP 3-way.
- **엄밀히 EKF 아니라 LTV KF** (명목 선형). 정직히 명명. 실차 비선형측정 대비
  전방호환(평균함수+야코비안 주입, ca.jacobian 자동). KF→EKF 변환은 필터코어
  불변 + 네 함수 교체만 → 쉬움(확인됨).
- 제어통합: 세 케이스 모두 완전상태 피드백(통제), 차이는 모델보정만. KF는 d_hat
  주입(연속 외란, RK4 안), GP는 mean 주입(이산, RK4 밖).

**확정된 Phase 7 결정 (2026-07-25, prompts-phase-late.md「Phase 7」에 기록):**
- 입력 z = **[v_y, gamma, delta] (3D)**. v_x 제외 — 슬립각 민감도
  d(alpha_f)/d(vx)~5e-4 rad/(m/s)라 v_x 독립기여 0.04°(무시). 차원 최소.
- 채널 = v_y, gamma 독립 GP 2개. 커널 ARD RBF, Type-II ML(log-det 포함).
- 프레임워크 = **직접 numpy/scipy exact GP** (torch 미사용).
- 결합 = 이산 x_{k+1}=rk4(f_nom)+B_d@mu_GP(z), RK4 밖. mu_GP는 지평 예측상태에서
  평가 → GP가 d(x)를 지평에서 예측(KF 상수 d_hat/오라클 상한 +7.3% 넘음).
- 학습=single_curve a_y=4, 평가=a_y=6 & dlc(분포밖 사후분산 시연). M은 config.
- 게이트: 게이트7 + 사후 std 단조증가 + numpy↔CasADi 일치 + MPC+GP<MPC+KF<only.

**Phase 8 미결은 `handoff.md` 「미결 결정」 참조.** (M 최종값은 Phase 7 상수결합으로
solve time 과 무관해져 미결에서 제외 — 적합 품질 기준으로만 고르면 된다.)

## Phase 5 산출물 메모

- 잔차 계산은 항상 **명목** 대비. 비선형은 plant_rhs 주입만 (runner 무수정).
- Fiala: z=tan(clip(α,±α_sl)) 단일 큐빅 → 경계밖 자동 포화·C1연속, if_else 불필요.
- **기구학 채널 진단**: 매칭은 10×floor 규칙, 불일치는 0.5·dt 비례관계 이탈로 판정
  (gp-residual.md 정정 반영). 절대크기로 버그 판정 금지.

## Phase 4 산출물 메모

- **solve time 정정**: 폐루프 warm start 로 mean 9.2ms (<20ms), dt초과 0%.
  Phase 3의 23ms 우려는 cold+짧은런 편향이었음. 명목 MPC는 실시간 가능.
- 잔차 계산 = `build_step_function(NominalStepModel)` → MPC 예측과 동일 이산 전이
  재사용(복사 아님). 잔차는 항상 **명목** 대비 (GP 케이스도 동일).
- **limo 주의**: single_curve 속도프로파일(a_y=4.0→vx 14)이 sedan용. limo는
  vx≤1.0 clip 크롤(25s에 25m). 절대잔차는 noise-floor 수준이나 sedan floor 대비
  배율(gamma 548×)은 limo의 다른 동역학 스케일 탓, 버그 아님. limo 전용 경로는
  파라미터 식별 후로 미룸.
- StepModel 주입 완료 → Phase 7 GP는 MpcBase/mpc_nominal 수정 없이 붙는다.
  GP StepModel: extra_param_dim>0, step_sym=rk4+B_d@mu_GP, P 확장블록에 Z/alpha.

## Phase 2 산출물 메모

- 세그먼트 스키마 = {length, kappa_end}. 시작곡률=이전끝(0에서 출발) → C0 연속 자동.
- dlc 최대 횡변위 8.5m (실차선보다 큼) — Phase 5에서 잔차 SNR 보고 재조정 예정.
- `Reference(path_cfg, vx_range)` — vx_range는 차량 config에서 주입 (경로 속성 아님).

## 해소된 미결 (이력 — 되살리지 마라)

| # | 항목 | 해소 |
|---|---|---|
| 구1 | EKF 비교군을 외란증강 상태 EKF로 할 것인가 | **확정(Phase 6)** — 증강 LTV KF. 위 「확정된 Phase 6 결정」 |
| 구2 | `P` 대각을 매 스텝 로깅 | **구현됨(Phase 6)** — `runner.py` 가 `ekf_P_diag` 로깅 |
| 구3 | `M` 고정 — solve time 이득 프로파일링 대기 | **무효(Phase 7)** — 상수결합으로 GP 평가가 solve당 1회가 되어 `M` 은 solve time 과 **무관**해졌다. `M` 은 이제 적합 품질만 결정하는 값이라 프로파일링 대상이 아니다. Part 2 sparse(Phase 9)에서는 다른 이유로 다시 다룬다 |

## 확정된 결정 (되돌리려면 사용자 확인 필요)

- **GP 결합 = 이산 형태 + 지평 상수.** `x_{k+1} = rk4(f_nom) + B_d @ mu_hat`,
  `mu_hat = mu_GP(z_now)`. GP 평가는 **solve당 1회**(작동점). ~~solve당 N회~~ 는
  상태의존 결합 전제의 옛 값 — Phase 7 진단으로 폐기됐다 (`mpc-solver.md` 「GP 결합 시」).
- **온라인 GP 하이퍼파라미터는 고정.** `Z`, `alpha`만 갱신. 재학습은 config 옵트인.
- **진단 채널 임계값 = `noise_floor`의 10배.**
