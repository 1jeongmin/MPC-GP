# MPC-GP — GP 잔차 보정 기반 차량 횡방향 경로추종 MPC

명목 **선형 동역학 자전거 모델**(오차좌표계) 위에 **모델 오차 보정**을 얹은 횡방향
경로추종 MPC를 Python + CasADi로 구현하고 시뮬레이션으로 검증한다.

보정 방식 세 가지를 **완전히 동일한 조건**에서 비교한다.

| 케이스 | 보정 방식 | 불확실성의 정체 |
|---|---|---|
| `mpc_only` | 없음 (기준선) | — |
| `mpc_kf` | 외란증강 칼만필터가 외란 `d`를 **시간축**으로 추정 | `P` 대각 (aleatoric) |
| `mpc_gp` | GP가 상태→잔차 사상을 **상태공간**에서 학습 | 사후분산 (epistemic) |

핵심 논점은 "GP가 추종을 더 잘한다"가 아니라
**"GP의 불확실성은 상태공간 지도이고, KF의 것은 시간축 스칼라다"** 이다.

---

## 1. 지시받은 모델링 순서와 진행 현황

주신 4단계(state equation / 잔차 정의 / path tracking 구현 / 잔차 모델링)에 **전부
완료**로 답했다. 상세 유도·수식·구현 링크는 분리했다 — **[`MODEL.md`](MODEL.md)**.

요지만: 상태 $x=[v_y,\gamma,e_\psi,e_y]$, 잔차는 타이어 비선형성 하나로 통제,
Python(CasADi)으로 구현(MATLAB 대비 GP-NLP 결합이 자연스러워서), KF·GPR 둘 다 완료.

---

## 2. 지시 범위를 넘어 추가로 진행한 것

4번까지 마친 뒤, "GP가 KF보다 낫다"를 **주장이 아니라 측정으로** 만들기 위해 아래를 추가했다.

| 항목 | 내용 |
|---|---|
| **3-way 통합 실험 인프라** | 케이스를 config로만 정의하고, 조립된 config를 diff해 **통제변수가 다르면 실행을 거부** |
| **UQ 캘리브레이션** | 커버리지·NLPD. KF의 연속 외란을 CasADi 야코비안으로 이산 잔차 공간에 환산해 **GP와 같은 대상**으로 비교 |
| **분포이동 시나리오** | in-distribution / 강도 외삽($a_y$ 4→6) / 형상 외삽(dlc) |
| **외부 레이싱 트랙 검증** | 강의 실습 배포 `.mat` 트랙(955.5 m)으로 재현 — 합성 경로 편향 배제 |
| **트랙 GP의 형상외삽 축 추가** | 방향 반전은 사전 검증에서 기각(원본 트랙이 이미 좌우 혼합이라 새 영역이 아님) → 강의 자료의 별도 트랙(right_turn, 76% 우회전 전용)으로 대체, 채택 전 사후분산 실측으로 확인 |
| **현실적 센서 잡음** | 이상적 센서 가정을 걷어내고 측정 잡음 하에서 재실험 |
| **물리 게이트 테스트 73개** | "코드가 도는지"가 아니라 "물리가 맞는지"를 본다 |

---

## 3. `results/` 로 본 진행 과정

각 실행은 `results/<run_id>/`에 **config 전체 스냅샷 + git hash + seed + 라이브러리 버전**과
함께 저장된다. 재현 불가능한 실험은 실험이 아니다.

> `results/`와 `data/`는 `.gitignore` 대상이라 GitHub에는 올라가지 않는다.
> §6의 실행 명령으로 그대로 재생성할 수 있다.

| # | run_id | 단계 | 무엇을 확인했나 |
|---|---|---|---|
| 1–2 | `openloop_20260725_2137*` | 3번 | step steer 응답. sedan $\tau\approx0.126$ s(감쇠진동) vs limo $\tau\approx0.010$ s(과감쇠). **정상상태 요레이트가 해석해와 1% 이내 일치** |
| 3 | `baseline_matched_20260725_2207` | 3번 | 플랜트=명목(완전일치) 폐루프. 잔차가 noise floor와 같은 자릿수인지 → **구현 검증** |
| 4 | `mismatch_sedan_20260725_2220` | 2번 | 플랜트만 Fiala로 교체. **잔차 SNR·커버리지 산점도·$a_y$ 스윕** → GP 설계의 입력 |
| 5 | `part1_20260726_163843` | 4번 | 3-way 첫 통합 실행 (ay4만, 파이프라인 검증) |
| 6 | `part1_20260726_163931` | 추가 | **합성 경로 9런** (3케이스 × ay4/ay6/dlc) |
| 7 | `part1_20260726_170758` | 추가 | **레이싱 트랙 6런** (3케이스 × rt_ay4/rt_ay6) |
| 8 | `part1_20260726_184536` | 추가 | 센서 잡음 첫 실행 → **KF 발산 발견** (설정 오류, §4 참조) |
| 9 | `part1_20260726_190107` | 추가 | $Q_d$ 재튜닝 후 **잡음 6런 재실행** (최종) |
| 10 | `part1_20260727_000727` | 추가 | **right_turn 형상외삽 3런** — 재학습 없이 기존 트랙 GP로 평가 |
| 11 | `part1_20260727_002844` | 추가 | **right_turn + 센서 잡음 3런** — 형상축 우위가 잡음에서도 유지되는지 확인 |

각 `part1_*` 폴더 구성:

```
results/part1_<timestamp>/
├── summary.csv / summary.json      # 전 케이스 비교 요약
├── meta.json                       # 배치 재현성 메타
└── <시나리오>/
    ├── part1_<s>_cases.png         # 케이스 비교 (e_y, δ, 잔차)
    ├── part1_<s>_uncertainty.png   # ★ 주 그림 — GP 상태공간 지도 vs KF 시간축
    └── <케이스>/
        ├── log.npz                 # 제어 스텝별 전체 로그
        ├── meta.json               # config 스냅샷 + git hash + seed
        └── *_report.png            # 단일 런 종합 그림
```

### 현재 상태 (2026-07-30)

두 브랜치로 나눠 뒀던 시기가 있었으나(생측정 피드백 `main` vs 필터링
`feature/state-estimator-feedback`) **둘 다 `main` 에 병합해 흡수했다** —
MPC 비용함수 수정과 공통 상태추정기(잡음 낀 측정을 명목 4상태 KF 로 걸러 세
케이스 모두 동일하게 피드백)를 각각 따로 반영했다. 지금은 브랜치가 하나뿐이고
**필터링이 기본 구성**이다. 그 병합이 아래 §4 수치를 전부 무효로 만들었다 —
경위와 재측정 계획은 `handoff.md` 를 봐라.

---

## 4. 주요 결과

**⚠ 무효 — 옛 비용함수 기준.** 상세는 **[`RESULTS.md`](RESULTS.md)**.
현재 상태·재측정 계획은 `handoff.md`「미결 #9」.

---

## 5. 저장소 구조

```
gpmpc/
├── CLAUDE.md              # 프로젝트 규약 (부호·구조·금지 사항)
├── handoff.md             # 세션 인계 + 미결 결정 단일 소스
├── prompts.md             # Phase별 작업 지시 이력
├── .claude/rules/         # 경로별 조건부 룰 7개
├── configs/               # 모든 수치의 단일 출처 (하드코딩 금지)
│   ├── vehicle/  path/  mpc/  ekf/  gp/  sim/  sensor/  viz/
│   └── experiment/        # 조합 파일 = 케이스 정의
├── src/
│   ├── models/            # linear_bicycle, tire, nonlinear_bicycle, integrators
│   ├── path/              # reference — κ(s) 프로파일·프리뷰
│   ├── control/           # mpc_base, mpc_nominal, mpc_kf, mpc_gp
│   ├── estimation/        # ekf — 외란증강 LTV KF
│   ├── gp/                # dataset, kernels, train_offline, casadi_export
│   ├── sim/               # runner(다중레이트), assemble(케이스 조립), sensor, logger
│   ├── eval/              # metrics, calibration
│   └── viz/               # plots
├── scripts/               # run_openloop, run_sim(범용), run_part1(배치)
├── tests/                 # 물리 게이트 73개
├── data/                  # 경로·GP 데이터셋 (git 제외)
└── results/               # run_id별 산출물 (git 제외)
```

### 설계의 중심 — `StepModel` 주입

세 케이스가 `if`문 없이 갈리는 이유가 여기 있다. **`MpcBase`는 어떤 케이스인지 모른다.**

```python
class StepModel(Protocol):
    extra_param_dim: int
    def step_sym(self, x, u, vx, kappa, p_extra): ...   # 이산 1스텝 전이
    def extra_param_values(self) -> np.ndarray: ...
```

| 구현체 | 이산 전이 | 결합 위치 |
|---|---|---|
| `NominalStepModel` | $\mathrm{rk4}(f_{nom})$ | — |
| `DisturbanceStepModel` (KF) | $\mathrm{rk4}(f_{nom}+B_c d)$ | RK4 **안** ($d$는 rate) |
| `GPStepModel` (GP) | $\mathrm{rk4}(f_{nom})+B_d\hat\mu$ | RK4 **밖** ($r$은 상태차이) |

케이스 선택도 문자열 분기가 아니라 **참조된 config 그룹**으로 한다
([`src/sim/assemble.py`](src/sim/assemble.py)) — `gp` 그룹이 있으면 GP 케이스,
`ekf`가 있으면 KF, 둘 다 없으면 명목. 새 케이스 추가 = 레지스트리 한 줄.

---

## 6. 실행 방법

```bash
pip install -r requirements.txt      # Python 3.12 기준

# 물리 게이트 전체 (약 80초)
pytest tests/ -q

# 개루프 step steer 응답 (지시 3번)
python scripts/run_openloop.py

# 단일 실험
python scripts/run_sim.py part1_gp_rt_ay4

# 3-way 배치 (통제변수 자동 검증 후 실행)
python scripts/run_part1.py rt_ay4 rt_ay6     # 레이싱 트랙
python scripts/run_part1.py rtn_ay4 rtn_ay6   # + 센서 잡음
python scripts/run_part1.py                   # 전체
```

시나리오: `ay4` `ay6` `dlc` (합성) / `rt_ay4` `rt_ay6` `rt_rturn` (트랙) /
`rtn_ay4` `rtn_ay6` `rtn_rturn` (트랙+잡음)

**트랙 데이터**는 `data/path/`에 두 파일을 두어야 한다(둘 다 강의 실습 배포본.
`data/`는 git 제외이므로 별도 복사 필요. config에 sha256을 고정해 파일이 바뀌면 즉시 중단):
- `path_data_racetrack.mat` — 실습 2차 배포본 (rt_ay4/rt_ay6/rtn_*)
- `path_data_right_turn.mat` — 실습 1차 배포본 (rt_rturn)

---

## 7. 실험 조건

**⚠ 아래 표는 §4(옛 결과)를 만들 때의 조건이다. 현재 값은 다르다** — 최신은
`.claude/rules/mpc-solver.md` + `handoff.md` 를 봐라. 요지: `N`=50 + 입력 블로킹
(자유 20 + 꼬리 굵은 블록), IPOPT warm start, 비용함수는 `W_vy`=`R_delta`=0 +
`gamma` 참조 상대화, 피드백은 센서 잡음 + 공통 상태추정기(KF)가 기본.

| 항목 | 값 (옛 결과 기준) |
|---|---|
| 차량 | sedan ($m$=1500 kg, $I_z$=2250 kg·m², $a$=1.2 m, $b$=1.5 m, $C_f$=$C_r$=80000 N/rad, $\mu$=0.9) |
| 다중레이트 | plant 100 Hz (DOP853) / control 50 Hz (RK4), ZOH |
| MPC | $N$=30 (0.6 s), IPOPT, warm start(이전 해 shift) |
| 경로 | 합성 single_curve(140 m)·dlc(126 m) / 외부 레이싱 트랙(955.5 m, $R_{min}$=32.4 m) |
| GP | $M$=100, ARD RBF, Type-II ML, $\sigma_n$ floor 1e-2 |
| 센서 잡음 | std [0.03 m/s, 0.002 rad/s, 0.003 rad, 0.03 m] — 센서 등급별 통상값 기준 **공학적 추정치** |
| seed | 0 (전 케이스 동일) |

**공정성 원칙**(지금도 유효): KF와 GP에 동일한 튜닝 예산(폐루프 평가 20회
이내)을 배정하고, in-distribution 시나리오만 보고 결정한 뒤 외삽 시나리오
결과를 본 후에는 어느 쪽도 수정하지 않는다. 튜닝 방식·예산은 config에 같은
형식으로 기록한다.

---

## 8. 남은 작업

**미결 결정의 단일 소스는 `handoff.md`「미결 결정」표다 — 여기 복제하지 않는다.**
현재 최우선: `Q_d` 재튜닝(미결 #10), 축 B 재측정 범위 확정(미결 #9),
loop10 용 잡음+필터 시나리오 작성(미결 #12).

### 예정 작업

- **Part 2: GP 실시간화** — online / sparse / sliding-window GP.
  `src/gp/{online,sparse,sliding_window}.py`가 현재 스텁. 주 지표는 solve time 분포와
  `dt_ctrl` 초과율. 네 변형이 같은 인터페이스를 구현해 runner가 변형을 몰라야 한다.
- **LIMO 하드웨어** (선택) — 착수 전 $C_f, C_r, I_z, \delta_{max}$ **식별 실험 필수**.
  현재 `configs/vehicle/limo.yaml` 값은 전부 자릿수 추정치라 그대로 쓰면 결과 해석이 불가능하다.
  (시뮬레이션에서 limo는 속도 범위가 좁아 슬립 유도가 안 되는 것도 확인됨)

---

## 9. 알려진 한계

정직하게 적어 둔다.

- **센서 잡음 값은 특정 논문 수치가 아니라 센서 등급별 통상값 기준의 공학적 추정치**다.
  문헌 인용이 필요하면 실제 데이터시트로 대체해야 한다.
- $v_y$를 직접 측정한다고 가정했다. **양산차에는 $v_y$를 직접 재는 센서가 없다.**
  부분관측이 KF를 악화시키는 것을 확인한 뒤, 이 연구의 초점이 상태추정이 아니라
  모델오차 처리이므로 전상태 측정으로 통제한 결과다.
- Fiala 모델의 큐빅 형태는 여러 문헌에서 동일하게 확인되지만 **원 출처 DOI는 확인하지 못했다.**
  다만 선형 극한·포화·원점 대칭 성질은 수치미분으로 자체 검증했다.
- `limo` 차량 파라미터는 전부 미검증 추정치다(YAML의 `TODO` 주석 참조).
- GP는 mean-only(non-cautious)다. 분산을 제어에 전파하는 stochastic MPC는 이번 스코프 밖이다.
- **(해소됨) 잡음 시나리오 조향 rate 포화 ~47%** — 옛 결과(§4)에서는 세 케이스 모두
  필터링 안 된 생측정값을 제어 피드백으로 받아 발생했다. 2026-07-30 공통
  상태추정기(잡음 낀 측정을 명목 4상태 KF 로 걸러 세 케이스 모두 동일하게 피드백)
  도입으로 0%로 해소, 지금은 기본 구성이다. 다만 그 병합이 §4 수치 전체를
  무효화했다 — `handoff.md` 참조.
- **잔차를 타이어 비선형성 하나로 통제한 것은 실차 잔차를 대표하지 않는다.** 확인해 본
  GP-MPC 문헌(Hewing et al. arXiv:1705.10702, GP-based overtaking MPC 2021)은 대부분
  잔차를 "model mismatch and unmodeled dynamics"로 뭉뚱그려 학습시키지, 물리 원인
  하나로 좁히지 않는다. 이 프로젝트의 단일원인 통제는 인과관계를 깨끗하게 검증하기
  위한 의도적 선택이며, combined slip·하중이동·액추에이터 지연·파라미터 불확실성 등
  실차에서 섞여 들어올 다른 잔차원은 다루지 않는다.
