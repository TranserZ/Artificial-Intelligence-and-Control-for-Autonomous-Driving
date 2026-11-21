# 데이터 컬럼 정의 (env-v1 · 단계별 맵)

> **한 줄 요약** · 수집 데이터는 한 줄이 한 스텝(0.5초)인 표다. 컬럼은 **167개이고 단계별 맵 1~6단계 모두 같다**. **학습에 쓰는 것은 `obs.*` 67개뿐**이다. 나머지는 행동, 결과, 3인칭 상태(분석용)다. 신호등은 하나당 독립 컬럼이고 색은 숫자로 나눴다.

| 작성일 | 작성자 | 수집 코드 | 파일 |
|---|---|---|---|
| 2026-11-17 | 강동연 | `collect_pomdp_data.py` | `data/<이름>.table.csv.gz` (이 표), `.jsonl.gz` (과제 기본 형식), `.npz` |
| 2026-11-21 수정 | (`env/simplified_road_config`) | 〃 | 신호 정지선 6 → 7개, 컬럼 161 → 167개, 단계와 상관없이 칸 고정 |

## 0. 학습에 쓰는 컬럼

| 구분 | 컬럼 | 학습 입력 | 설명 |
|---|---|---|---|
| 관측 | `obs.*` (67개) | **사용** | 버스가 실제로 보는 부분 관측. 정책 입력은 이것뿐 |
| 행동 | `action_raw.*`, `action.*` | 출력 | 정책이 낸 행동과 실제로 적용된 값 (모방학습이면 정답) |
| 보상·종료 | `reward`, `terminated`, `truncated` | 학습 신호 | 강화학습 보상 |
| 사건 | `event.*` | 사용 안 함 | 충돌·위반·shield 개입·정류장 성공 등 판정 결과 |
| 3인칭 상태 | `state.*` | **사용 안 함** | 시뮬레이터가 아는 상태. 분석·검증용. 관측에 없는 정보(신호 잔여시간, 모든 신호 색 등)가 들어 있다 |
| 식별 | `episode`, `seed`, `t`, `route` | 사용 안 함 | 판 번호, 판 시드, 판 안 스텝 번호, 경로 |

- 모든 값은 숫자다 (`route`만 문자열 `r0`). 없는 값은 `-1`.
- **칸 수 내역 (167개)**: 식별 4 + `obs.*` 67 + 행동 6 + 보상·종료 3 + `event.*` 16 + `state.*` 71.
- **단계별 맵에서도 칸은 같다.** 꺼진 요소는 칸을 지우지 않고 "없음" 값으로 채운다. 예: 1단계(신호 없음)는 `state.sig0~6.*` 42칸이 모두 `-1`, 관측의 신호 칸은 (1, 0, 0, 0, 0) · 보이지 않음.
- 과제 기본 수집 항목(observation, action_raw, action.*, reward, terminated, truncated, state, detected_vehicles, event.*, next_*)은 `.jsonl.gz`에 그대로 있다. 이 표는 그것을 컬럼으로 편 것이다.
- `next_observation`은 같은 판의 다음 줄 `obs.*`다 (표에는 따로 두지 않음). 판의 마지막 줄은 `terminated` 또는 `truncated`가 1이다.

## 1. 관측 `obs.*` (67개, 범위 −1~1)

차로 이름: `L2`/`L1` = 왼쪽 두/한 칸, `C` = 지금 차로, `R1`/`R2` = 오른쪽 한/두 칸.

| 컬럼 | 개수 | 뜻 | 정규화 |
|---|---|---|---|
| `obs.speed` | 1 | 내 속도 | / 13.89 m/s (50 km/h, 정규화 기준. 청계천 버스 최고 30 km/h라 0.6까지만 나옴) |
| `obs.{차로}.lead_gap`, `.lead_dv`, `.follow_gap`, `.follow_dv` | 20 | 그 차로 앞차 거리·상대속도, 뒤차 거리(음수)·상대속도. 50 m 안에 없으면 앞 (1, 1), 뒤 (−1, −1) | 거리 / 50 m, 속도 / 제한속도 |
| `obs.{차로}.lane_end` | 5 | −1 차로 없음, 1 계속 이어짐, 0~1 차로가 끝나는 곳까지 거리 | / 50 m |
| `obs.{차로}.density` | 5 | 앞 50 m 안 차량 밀도 (가려진 차까지 셈) | 0 빈 차로 ~ 1 꽉 참 |
| `obs.signal.dist` | 1 | 다음 신호 정지선까지 거리 | / 150 m |
| `obs.signal.red`, `.yellow`, `.green`, `.stop` | 4 | 다음 신호 색 one-hot (`stop` = 일시정지 후 진행) | 0/1 |
| `obs.signal.visible` | 1 | 신호가 보이는가. **버스가 맨 앞 차일 때만 1**. 0이면 위 5개는 (1, 0, 0, 0, 0) | 0/1 |
| `obs.turn.left`, `.straight`, `.right` | 3 | 다음 교차로에서 할 회전 (내비 안내) | 0/1 |
| `obs.{차로}.arrow_left`, `.arrow_straight`, `.arrow_right` | 15 | 그 차로 노면 화살표 (갈 수 있는 방향) | 0/1 |
| `obs.crossing{0,1}.dist`, `.ped_gap`, `.ped_toward` | 6 | 경로상 다음 횡단보도 2개: 거리, 가장 가까운 보행자 ↔ 내 경로 구간 거리, 다가오는가. 보이는 보행자만 | 거리 / 50 m, 보행자 거리 / 횡단보도 길이 |
| `obs.school_zone.dist`, `.limit` | 2 | 보호구역 입구까지 거리(안이면 0), 제한속도 | / 100 m, / 제한속도 |
| `obs.bus_stop.dist` | 1 | 다음 정류장(끝)까지 거리 | / 150 m |
| `obs.bus_stop.lane_diff` | 1 | 정류장 차로까지 차로 차이 (+ 왼쪽, − 오른쪽으로 가야 함) | / 2 |
| `obs.bus_stop.dwell` | 1 | 정류장 구간에서 연속으로 서 있은 시간 (1이면 정차 성공) | / 10 초 |
| `obs.lc_zone` | 1 | 지금 차선변경 금지 구간인가 | 0/1 |

관측에 넣지 않은 것: 신호 잔여시간, 뒤에 서 있을 때의 신호, 보행자의 다음 경로, "이 차로가 내 경로로 이어지는가".

## 2. 행동

| 컬럼 | 뜻 |
|---|---|
| `action_raw.accel` | 정책이 낸 가감속 (−1 최대 감속 ~ +1 최대 가속) |
| `action_raw.lane_change` | 정책이 낸 차선변경 원본 값 (−1~1) |
| `action.accel_raw` | 실제 적용된 가감속 (shield `on`일 때만 정책 값과 다름. 수집은 `shadow`라 같음) |
| `action.lane_change_raw` | 환경에 전달된 차선변경 원본 값 |
| `action.lane_change` | 실제 명령 −1 오른쪽 / 0 유지 / +1 왼쪽 |
| `action.lane_change_applied` | 그 명령이 실제로 적용됐는가 (교차로 안·없는 차로면 0) |

## 3. 사건 `event.*` (스텝 결과)

| 컬럼 | 뜻 |
|---|---|
| `event.collided`, `.arrived` | 충돌 / 도착 |
| `event.red_violation`, `.yield_violation`, `.ped_collision`, `.wrong_lane` | 적색 통과 / 횡단보도 양보 위반 / 보행자 충돌 / 경로 밖 차로에 갇힘 |
| `event.shield_vehicle`, `.shield_signal`, `.shield_ped`, `.shield_lc_gate` | shield가 개입했을까 (감시 모드): 앞차 간격 / 적색 정지선 / 보행자 / 옆 차로 간격 |
| `event.shield_active` | 위 넷 중 하나라도. 수동개입률 = 이 값의 평균 |
| `event.lc_forbidden` | 차선변경 금지 구간에서 변경을 명령했는가 |
| `event.stop_success`, `.stop_missed` | 이번 스텝 정차 성공 수 / 놓친 정류장 수 |
| `event.lane_before`, `.lane_after` | 행동 전후 차로 (맨 오른쪽 차로 = 0) |

## 4. 3인칭 상태 `state.*` (학습에 쓰지 않음)

| 컬럼 | 뜻 | 단위 |
|---|---|---|
| `state.ego.x`, `.y`, `.angle` | 버스 위치, 방향 | m, 도 |
| `state.ego.speed`, `.accel` | 속도, 가속도 | m/s, m/s² |
| `state.ego.lane`, `.lane_pos`, `.route_index`, `.odometer`, `.in_junction` | 차로(오른쪽 = 0), 엣지 안 위치, 경로상 엣지 순번, 출발 후 주행거리, 교차로 안인가 | — , m, —, m, 0/1 |
| `state.lead.gap`, `.speed` | 같은 차로 앞차 거리·속도 (100 m까지, 없으면 −1) | m, m/s |
| `state.traffic.n_vehicles`, `.n_near50`, `.n_persons` | 전체 일반 차량 수, 버스 반경 50 m 안 차량 수, 보행자 수 | 대, 명 |
| `state.sig{k}.red`, `.yellow`, `.green`, `.stop` | 신호 정지선 k(0~6)의 색 one-hot (버스 진행 방향 신호). 신호 없는 단계는 -1 | 0/1 |
| `state.sig{k}.time_to_switch` | 신호등 k가 다음에 바뀌기까지 | s |
| `state.sig{k}.dist` | 신호등 k 정지선까지 경로상 거리 (지났으면 −1) | m |
| `state.stop{k}.status` | 정류장 k: 0 남음 / 1 정차 성공 / −1 놓침 | — |
| `state.stop{k}.dist` | 정류장 k 끝까지 거리 (끝났으면 −1) | m |
| `state.school_zone.inside`, `.dist` | 보호구역 안인가, 입구까지 거리 (없으면 −1) | 0/1, m |
| `state.pax.onboard`, `.waiting_next_stop` | 탄 승객 수, 다음 정류장 대기 승객 수 | 명 |

### 신호 정지선 번호 (경로 r0, 지나는 순서)

신호기는 6대, 정지선은 7개다. 청계2가(삼일대로)는 큰 교차로라 신호기 1대에 정지선이 2개다 (k = 2, 3).
2026-11-21 도로 모양 버그 수정(엣지 끝점 누락) 뒤 값. 2~6단계에 있고, 1단계는 7칸 모두 -1.

| k | 교차로 | 들어가는 엣지 | 출발점에서 거리 | 신호 시간 출처 (`data/road_data/signal_timing.csv`) |
|---|---|---|---|---|
| 0 | 무교로 (청계광장 동쪽) | `e73541403_2` | 약 153 m | 광교 값 차용 |
| 1 | 광교 (남대문로) | `e73541403_6` | 약 436 m | API 실측 |
| 2 | 청계2가 (삼일대로) 서쪽 정지선 | `e1536659790_4` | 약 859 m | 교통영향평가 |
| 3 | 청계2가 (삼일대로) 동쪽 정지선 | `e908646367_2` | 약 923 m | 〃 (k = 2와 같은 신호기) |
| 4 | 수표교 (수표로) | `e908646367_4` | 약 1067 m | 광교 값 차용 |
| 5 | 청계3가 (충무로) | `e908646367_6` | 약 1320 m | API 실측 |
| 6 | 청계4가 (창경궁로) | `e218924602_6` | 약 1803 m | 교통영향평가 |

### 정류장 번호

| k | 정류장 |
|---|---|
| 0 | bs_01 청계광장 |
| 1 | bs_02 청계1가.광교 |
| 2 | bs_03 청계2가.삼일교(동) |
| 3 | bs_04 청계3가.관수교(동) |
| 4 | bs_05 청계4가.배오개다리 |

## 5. 코드

- 관측 컬럼 이름: `env.obs_names()` (길이 = `env.obs_dim`)
- 3인칭 상태 컬럼: `env.get_state_columns()` (시점 t, 행동 전). 신호 칸 수는 `env/sumo_env.py`의 `STATE_SIGNAL_SLOTS` (= 7)
- 표 한 줄: `collect_pomdp_data.table_row()`

```python
import pandas as pd
df = pd.read_csv("data/<이름>.table.csv.gz")
X = df.filter(like="obs.").to_numpy()      # 학습 입력 (67개)
y = df[["action_raw.accel", "action_raw.lane_change"]].to_numpy()   # 모방학습 정답
```
