# CGC_STAGE=1 학습·진단 계획

진단 정책은 IDM 차간거리 유지 + 정류장 10초 정차, 차선변경 없음.
결과는 `results/idm_diag/`에만 두고 학습 데이터로 쓰지 않는다.

## 수동개입 (코드가 세는 것)

팀 문서 `docs/env_stage_maps.md` §7, `rules/team/experiment_protocol.md`:
수동개입률 = 감시 모드에서 **전체 주행 step 중 shield가 개입했어야 하는 step 비율**.

코드:

- `env/sumo_env.py` `info["shield_active"]`: 그 스텝에 차량 shield · 신호 shield · 보행자 필터 · 차선변경 게이트 중 **하나라도** 걸림.
- `utils/evaluator.py` `intervention_rate` = `n_shield / 전체 step`.
- `scripts/common_eval.py` `shield_step_ratio`도 같은 값. 팀 목표 `team_goal_intervention_met`는 이 평균 ≤ 0.16.
- 학습 중 `SIMULATION["shield"]["mode"]="off"`이면 `_shield_check`를 안 돌려 `shield_active`는 거의 항상 false. **개입률은 평가 때 `--shield shadow`로만 의미가 있다.**

자동 판정(프로토콜): 충돌 5% 미만 + 공식 완주 95% 이상. 수동개입 16%는 팀 목표이며 프로토콜은 자동 판정에 넣지 않는다고 적혀 있다. 단계 맵 §7은 **성공 선언**에 완주 95% + 수동개입 16%를 같이 넣는다. 이 브랜치는 **단계 통과(다음 단계로 넘어감)** 에 §7을 따른다.

## 단계 통과 / 중단

| 구분 | 기준 | 팀 문서와 |
|---|---|---|
| **단계 통과 (다음 `CGC_STAGE`로)** | 개발 판 900100–900129, shield **shadow**, 공식 완주 ≥ 95% **그리고** `intervention_rate` ≤ 16% | `env_stage_maps.md` §7과 같음 |
| 공식 테스트 1000–1099 | 단계 통과 선언 후에만 | 같음 |
| 학습 로그 최근 100판 | 공식 완주가 아님. 중단 신호만 | **팀 문서의 성공 기준이 아님** |

학습 로그 중단 신호 (11/21 B 보고서, 중간 기준 · **팀 문서와 다름**):

- 충돌 > 40% 이면서 시간초과가 줄면 중단 (정차보다 충돌이 이득).
- 충돌은 낮은데 시간초과 ≥ 55%가 유지되면 중단 (다음 한 칸은 `progress_weight`).

이 신호로 가중치를 손볼 수는 있어도, 그 상태로 2단계로 넘기지 않는다.

## 4단계 보호구역 감점 되돌릴 순서

1–3단계는 아래 항을 **0으로 둔다**. 4단계 진입 때 **한 항씩**만 켠다 (B: 한꺼번에 되돌리지 않음).

| 순서 | 항 | 값 | 언제 다음으로 |
|---|---|---|---|
| 1 | `school_zone_speeding_penalty` | 0.1 (코드 상한 0.5) | 개발 판에서 보호구역 안 85% 속도가 20 km/h+3 km/h를 넘으면 |
| 2 | `zone_overspeed_penalty` + `zone_overspeed_cap` | 0.25 / 0.5 | 1만으로 과속이 남으면 |
| 3 | `zone_approach_penalty` | 0.05 | 구역 안에서 지킨 뒤에도 입구 과속이 남으면 |

`setSpeed` 상한은 보호구역 20 km/h를 자르지 않는다 (`ego_vmax` = 도로 30 km/h). 과속 억제는 보상·정책이다.

## Kaggle

`kaggle/exp-ppo-stage1.ipynb`. `CGC_STAGE=1`, `--eval-interval 0`, `--replay-prob 0`, `--total-steps 300000`.
세션마다 `model.resume.pt`를 `--resume`으로 이어 받는다. `--total-steps`는 **최종 누적** 목표다.
