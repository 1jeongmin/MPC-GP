# 프롬프트 아카이브 — Phase 6~7 (증강 KF · offline GP)

`prompts.md` 에서 분리된 완료 Phase 아카이브다(2026-07-30). 사용 규칙·현재
미결 결정은 `prompts.md`/`handoff.md` 를 봐라.

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

