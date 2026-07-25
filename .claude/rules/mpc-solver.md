---
paths:
  - "src/control/**"
  - "configs/mpc/**"
  - "tests/test_mpc*.py"
---

# MPC 정식화·솔버 룰

## 정식화 (multiple shooting + IPOPT)

- 결정변수: `X (4 x (N+1))`, `U (1 x N)`
- 파라미터 `P`: `[x0(4), vx_preview(N), kappa_preview(N), u_prev(1)]`
  - **GP 케이스는 여기에 GP 블록을 이어 붙인다** (아래 「GP 결합 시」 참조).
    명목 케이스와 GP 케이스는 `P` 레이아웃이 다르다. 레이아웃 정의는 한 곳에
    두고(예: `P_layout` 헬퍼) 인덱스를 손으로 세지 마라.
- 동역학 등식 제약: `X[:,k+1] = rk4_step(...)`, 각 스텝의 `vx`, `kappa`는 `P`에서 꺼낸다
  (GP 케이스에서는 여기에 GP 보정항이 더해진다)

비용:
```
sum_k [ Q_ey*e_y^2 + Q_epsi*e_psi^2
        + Q_vy*v_y^2 + Q_gamma*gamma^2          # 작은 정규화 항
        + R*delta^2 + R_d*(delta_k - delta_{k-1})^2 ]
+ 종단항 (e_y, e_psi)
```

제약:
```
|delta| <= delta_max
|delta_k - delta_{k-1}| <= delta_rate_max * dt      # k=0 에는 P의 u_prev 사용
```
상태 제약은 지금 넣지 않는다. 추가하려면 먼저 물어라.

**모든 가중치는 `configs/mpc/*.yaml`에서 읽는다. 하드코딩 금지.**

## 참조 상태·피드포워드를 별도로 계산하지 마라

`x_ref`, `u_ref`를 명시적으로 만들지 않는다. `kappa` 프리뷰가 지평에 채워지면
MPC가 필요한 요레이트를 스스로 만든다. 비용은 `e_y`, `e_psi`를 0으로 보내고
`delta`와 그 변화율을 벌주는 것으로 충분하다.

Mehrez 워크숍의 multiple shooting 골격을 따르되 **이 점이 다르다.**
워크숍 코드를 그대로 옮기면서 `u_ref` 항을 되살리지 마라.

## 표기 충돌 — 반드시 지킬 것

MPC 비용 가중치와 칼만필터 잡음 공분산은 **둘 다 관례적으로 Q, R**이라 불린다.
이 저장소에서는 절대 같은 이름을 쓰지 않는다.

| 대상 | 코드 이름 |
|---|---|
| MPC 상태·입력 가중치 | `W_x`, `W_u`, `W_du` (또는 `Q_ey` 등 명시적 이름) |
| KF 프로세스·측정 잡음 공분산 | `Q_kf`, `R_kf` |

## GP 결합 시

- GP 잔차 평균은 비선형 커널 함수다. 명목 모델이 선형이어도 **결합 모델은
  비선형이 되므로 NMPC**다. QP 솔버(osqp, qpOASES 등)로 전환하려는 시도를 하지 마라.
  전환 가능성은 GP가 없는 명목 케이스에 한해서만 논의 대상이다.
- **결합은 이산(discrete) 형태로 확정되어 있다.** GP 잔차는 상태 차이 단위이므로
  연속 우변에 더하면 `dt_ctrl`배 틀린다. 상세는 `gp-residual.md` 「결합 형태」 참조.

```
X[:,k+1] = rk4_step(f_nom, X[:,k], U[k], (vx_k, kappa_k), dt_ctrl)
           + B_d @ mu_GP(z_k)

z_k = [X[0,k], X[1,k], U[k]]        # [v_y, gamma, delta]
B_d = 4x2, 상단 2x2 단위행렬        # v_y, gamma 행에만 주입
```

- `e_psi`, `e_y` 행에는 **더하지 않는다.** 이 두 행은 순수 기구학이다.
- GP 평가는 RK4 **바깥**이므로 solve당 `N`회다 (`4*N`이 아니다).
  성능 문제가 보고되면 이 숫자를 먼저 확인하라.
- GP 케이스의 파라미터 `P` 확장 블록:
  `[..., Z(M x d), alpha(M x 2), lengthscale(2 x d), sigma_f(2), mu_z(d), sd_z(d), mu_r(2), sd_r(2)]`
  (표준화 통계도 파라미터로 넣어 학습 시점과 동일한 스케일링을 강제한다.)

## NLP 구조 재생성 금지

온라인 GP에서 dictionary가 갱신되어도 **NLP를 매 스텝 다시 빌드하지 마라.**
inducing point 개수 `M`을 고정하고, `Z`, `alpha`, 하이퍼파라미터를 CasADi
**파라미터로 주입**해 `nlpsol` 객체를 재사용한다. (제안 — 구조 고정이 목적이며,
실제 solve time 이득은 프로파일링으로 확인할 것)

## 솔버 운영

- **warm start**: 이전 해를 저장해 다음 스텝 초기추정으로 재사용한다.
- **solve time**: 매 호출마다 기록해 반환한다. Part 2의 핵심 지표다.
  IPOPT iteration 수와 수렴 플래그도 함께 반환한다.
- **수렴 실패 처리**: 조용히 넘어가지 마라. 실패를 로그에 기록하고,
  fallback(이전 해의 shift 적용) 여부를 명시적으로 구현한 뒤 그 사실을 보고한다.
  실패율은 결과 리포트에 반드시 포함한다.
- IPOPT 옵션(`max_iter`, `tol`, `print_level`)은 config에서 읽는다.
  케이스 간 비교 시 옵션을 동일하게 유지하라. 다르면 solve time 비교가 무의미하다.
