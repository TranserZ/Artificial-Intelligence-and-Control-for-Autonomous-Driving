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

자동 판정(프로토콜): 충돌 5% 미만 + **공식** 완주 95% 이상(놓친 도착 포함). 이 실험의 1단계 목표는 **완벽 완주**(5곳 정차). `env_stage_maps.md` §7 “완주율”이 어느 쪽인지는 **팀 확인 필요**(문서는 안 고침). 표에는 공식·완벽을 둘 다 적는다.

## 단계 통과 / 중단

| 구분 | 기준 | 팀 문서와 |
|---|---|---|
| **1단계 학습 목표** | 개발 판에서 **완벽 완주** (도착 + 정류장 놓침 0, 5곳) | §4-2 “완주 = 도착 + 놓침 0”에 가깝다. §7 문구는 팀 확인 필요 |
| 보고 | 공식 완주와 완벽 완주를 같은 표에 | 프로토콜 3-1과 같음 |
| 학습 로그 최근 100판 | 아래 중단 신호만 | **팀 문서의 성공 기준이 아님** |

학습 로그 중단 신호:

- 충돌 > 40% → 중단.
- 시간초과 ≥ 55%이고 판당 정차 성공이 1곳에 머물면 → 첫 정류장 버티기. 중단.
- `stop_success_mean`과 `lane_changes_per_episode`를 같이 본다.

이 신호로 2단계로 넘기지 않는다.

## 보호구역 과속

**판정 외, 감점 없음.** 4단계에서 보호구역 감점을 되돌리지 않는다.

- `scripts/common_eval.py` 공식 완주율은 `end_reason`이 `completed` 또는 `missed_stop`인 판 비율. `get_end_reason`은 충돌·적색·차로 오류·도착·시간초과만 본다. 과속은 종료 이유가 아니다.
- 자동 판정 `target_passed`는 충돌률 <5%와 공식 완주율 ≥95%만 (`aggregate_seeds`). `school_zone_speeding_ratio`는 선택 컬럼으로 기록만 한다 (`experiment_protocol.md` 3-4).
- `docs/env_stage_maps.md` §7 성공 기준은 완주율 95% + 수동개입률 16%. 과속 문턱은 없다.
- `setSpeed` 상한은 도로 30 km/h(`ego_vmax`)라 보호구역 20 km/h를 명령으로 자르지 않는다. 그래도 판정에 안 넣으므로 보상으로 막지 않는다.

## Kaggle

`kaggle/exp-ppo-stage1.ipynb`. `CGC_STAGE=1`, `--eval-interval 0`, `--replay-prob 0`, `--total-steps 300000`.
세션마다 `model.resume.pt`를 `--resume`으로 이어 받는다. `--total-steps`는 **최종 누적** 목표다.
