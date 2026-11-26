# 6단계 야간 실험

## 한눈에 보기

1. 기존 진단은 판별 저장 이전에 bcppo import 오류로 종료됐다. 핑퐁 원인과 수정 여부는 검증 중이다.
2. 독립 100판 평가 결과 대기 중.
3. 최종 환경 채택 판정 보류.
4. 학생 900103 (376.0–376.5초, bs_03)과 900100 (424.0–424.5초, 주정차 구간)을 화면에서 확인했다. 최종 후보 장면은 독립 평가 뒤 선별한다.
5. 비정상 정지, 15 km/h 미만의 평균 이동 속도, 반경 50 m 내 차량 빈도를 확인한다. 공용 환경과 팀원 파일은 유지했다.

## 0단계 정리

기존 진단 폴더에 판별 결과가 없었다. 부모가 종료된 SUMO 두 개를 정리했고 TensorBoard는 유지했다.

위치 표를 저장된 edge·pos 기준으로 정정했습니다. [정정 위치 표와 사건 좌표](2026-11-26_s6_위치재집계.md) · [samil_east 대조 주행](2026-11-26_s6_samil_대조.md)

## 개발 평가

### stage6_student (30/30판)

| 지표 | 값 |
|---|---:|
| episodes | 30 |
| official_complete | 17/30 (56.7%) |
| perfect_complete | 17/30 (56.7%) |
| collision_count | 7 |
| collision_rate | 7/30 (23.3%) |
| wrong_lane_count | 0 |
| timeout_count | 2 |
| inf_wait_episodes | 19 |
| wait_timeout_wrong_lane_episodes | 19 |
| green_stop_max_sec | 1394.5 |
| green_stop_time_mean | 0.12482 |
| green_stop_events_mean | 50.733 |
| moving_avg_speed_kmh | 18.345 |
| avg_speed_kmh | 9.4917 |
| teleports | 0 |
| sim_errors | 0 |
| red_light | 0.13333 |
| yield_steps | 0 |
| intervention | 0.0082901 |
| reasons | {'completed': 17, 'red_light': 4, 'collision': 7, 'timeout': 2} |
| collision | 0.23333 |
| timeout | 0.066667 |
| stop_success_mean | 3.9333 |
| avg_speed | 2.6391 |
| bs02_lc_fail | 0 |
| zone_max_overspeed | 0.35841 |
| zone_overspeed_step_rate | 0.007097 |
| 평균 이동 속도 15 km/h 미만 | 2/30 |
| 5초 이내 실제 반대 방향 차선 변경 | 2127 |
| 주변 차량이 있었던 스텝 | 39935/43212 (92.4%) |
| 주변 차량이 한 번도 없었던 판 | 0/30 |

| 다음 정류장 | edge | 위치 구간 시작 (m) | 방향 반전 | 대표 시드 | SUMO 시각 (s) |
|---|---|---:|---:|---:|---:|
| bs_03 | e908646367_3 | 30 | 1685 | 900101 | 223.5 |
| bs_03 | e908646367_3 | 40 | 362 | 900102 | 324.5 |
| bs_03 | e908646367_3 | 20 | 40 | 900101 | 220.5 |
| bs_04 | e908646367_6 | 60 | 16 | 900100 | 424.0 |
| bs_04 | e908646367_6 | 50 | 14 | 900104 | 455.5 |
| bs_04 | e908646367_6 | 40 | 4 | 900104 | 453.5 |
| bs_04 | e908646367_6 | 30 | 2 | 900104 | 452.5 |
| bs_05 | e218924602_6 | 60 | 2 | 900113 | 560.0 |
| bs_05 | e218924602_6 | 80 | 2 | 900120 | 621.0 |

### stage6_kl200k (30/30판)

| 지표 | 값 |
|---|---:|
| episodes | 30 |
| official_complete | 20/30 (66.7%) |
| perfect_complete | 20/30 (66.7%) |
| collision_count | 4 |
| collision_rate | 4/30 (13.3%) |
| wrong_lane_count | 0 |
| timeout_count | 3 |
| inf_wait_episodes | 24 |
| wait_timeout_wrong_lane_episodes | 24 |
| green_stop_max_sec | 1412 |
| green_stop_time_mean | 0.16585 |
| green_stop_events_mean | 60.433 |
| moving_avg_speed_kmh | 18.078 |
| avg_speed_kmh | 9.1125 |
| teleports | 0 |
| sim_errors | 0 |
| red_light | 0.1 |
| yield_steps | 0 |
| intervention | 0.017034 |
| reasons | {'completed': 20, 'red_light': 3, 'timeout': 3, 'collision': 4} |
| collision | 0.13333 |
| timeout | 0.1 |
| stop_success_mean | 4.2667 |
| avg_speed | 2.5333 |
| bs02_lc_fail | 0 |
| zone_max_overspeed | 0.62853 |
| zone_overspeed_step_rate | 0.022488 |
| 평균 이동 속도 15 km/h 미만 | 1/30 |
| 5초 이내 실제 반대 방향 차선 변경 | 2632 |
| 주변 차량이 있었던 스텝 | 47210/50104 (94.2%) |
| 주변 차량이 한 번도 없었던 판 | 0/30 |

| 다음 정류장 | edge | 위치 구간 시작 (m) | 방향 반전 | 대표 시드 | SUMO 시각 (s) |
|---|---|---:|---:|---:|---:|
| bs_03 | e908646367_3 | 30 | 2118 | 900100 | 243.5 |
| bs_03 | e908646367_3 | 40 | 425 | 900101 | 227.0 |
| bs_03 | e908646367_3 | 20 | 62 | 900100 | 239.0 |
| bs_04 | e908646367_6 | 50 | 12 | 900116 | 868.0 |
| bs_04 | e908646367_6 | 60 | 9 | 900100 | 421.5 |
| bs_04 | e908646367_6 | 30 | 5 | 900104 | 452.0 |
| bs_04 | e908646367_6 | 40 | 1 | 900107 | 394.5 |

### stage5_kl250k (10/10판)

| 지표 | 값 |
|---|---:|
| episodes | 10 |
| official_complete | 2/10 (20.0%) |
| perfect_complete | 2/10 (20.0%) |
| collision_count | 8 |
| collision_rate | 8/10 (80.0%) |
| wrong_lane_count | 0 |
| timeout_count | 0 |
| inf_wait_episodes | 6 |
| wait_timeout_wrong_lane_episodes | 6 |
| green_stop_max_sec | 610 |
| green_stop_time_mean | 0.30793 |
| green_stop_events_mean | 7.3 |
| moving_avg_speed_kmh | 18.125 |
| avg_speed_kmh | 9.0069 |
| teleports | 0 |
| sim_errors | 0 |
| red_light | 0 |
| yield_steps | 0 |
| intervention | 0.0062206 |
| reasons | {'collision': 8, 'completed': 2} |
| collision | 0.8 |
| timeout | 0 |
| stop_success_mean | 1.8 |
| avg_speed | 2.5079 |
| bs02_lc_fail | 8 |
| zone_max_overspeed | 0.025856 |
| zone_overspeed_step_rate | 0.00369 |
| 평균 이동 속도 15 km/h 미만 | 0/10 |
| 5초 이내 실제 반대 방향 차선 변경 | 0 |
| 주변 차량이 있었던 스텝 | 7416/7887 (94.0%) |
| 주변 차량이 한 번도 없었던 판 | 0/10 |

| 다음 정류장 | edge | 위치 구간 시작 (m) | 방향 반전 | 대표 시드 | SUMO 시각 (s) |
|---|---|---:|---:|---:|---:|

### teacher56 (10/10판)

| 지표 | 값 |
|---|---:|
| episodes | 10 |
| official_complete | 8/10 (80.0%) |
| perfect_complete | 8/10 (80.0%) |
| collision_count | 0 |
| collision_rate | 0/10 (0.0%) |
| wrong_lane_count | 0 |
| timeout_count | 2 |
| inf_wait_episodes | 8 |
| wait_timeout_wrong_lane_episodes | 8 |
| green_stop_max_sec | 35 |
| green_stop_time_mean | 0.056703 |
| green_stop_events_mean | 46.2 |
| moving_avg_speed_kmh | 17.86 |
| avg_speed_kmh | 8.105 |
| teleports | 0 |
| sim_errors | 0 |
| red_light | 0 |
| yield_steps | 0 |
| intervention | 0.64243 |
| reasons | {'completed': 8, 'timeout': 2} |
| collision | 0 |
| timeout | 0.2 |
| stop_success_mean | 4.8 |
| avg_speed | 2.2527 |
| bs02_lc_fail | 0 |
| zone_max_overspeed | 0 |
| zone_overspeed_step_rate | 0 |
| 평균 이동 속도 15 km/h 미만 | 1/10 |
| 5초 이내 실제 반대 방향 차선 변경 | 690 |
| 주변 차량이 있었던 스텝 | 20160/20692 (97.4%) |
| 주변 차량이 한 번도 없었던 판 | 0/10 |

| 다음 정류장 | edge | 위치 구간 시작 (m) | 방향 반전 | 대표 시드 | SUMO 시각 (s) |
|---|---|---:|---:|---:|---:|
| bs_03 | e908646367_3 | 30 | 386 | 900101 | 222.5 |
| bs_03 | e908646367_3 | 40 | 270 | 900102 | 325.0 |
| bs_04 | e908646367_6 | 60 | 9 | 900100 | 427.5 |
| bs_04 | e908646367_6 | 40 | 9 | 900101 | 469.0 |
| bs_03 | e908646367_3 | 20 | 6 | 900101 | 219.5 |
| bs_04 | e908646367_6 | 30 | 6 | 900104 | 440.5 |
| bs_04 | e908646367_6 | 50 | 4 | 900104 | 443.5 |

### teacher_bs03_hold (1/1판)

| 지표 | 값 |
|---|---:|
| episodes | 1 |
| official_complete | 1/1 (100.0%) |
| perfect_complete | 1/1 (100.0%) |
| collision_count | 0 |
| collision_rate | 0/1 (0.0%) |
| wrong_lane_count | 0 |
| timeout_count | 0 |
| inf_wait_episodes | 0 |
| wait_timeout_wrong_lane_episodes | 0 |
| green_stop_max_sec | 20.5 |
| green_stop_time_mean | 0.063738 |
| green_stop_events_mean | 12 |
| moving_avg_speed_kmh | 13.985 |
| avg_speed_kmh | 9.5145 |
| teleports | 0 |
| sim_errors | 0 |
| red_light | 0 |
| yield_steps | 0 |
| intervention | 0.63406 |
| reasons | {'completed': 1} |
| collision | 0 |
| timeout | 0 |
| stop_success_mean | 5 |
| avg_speed | 2.6446 |
| bs02_lc_fail | 0 |
| zone_max_overspeed | 0 |
| zone_overspeed_step_rate | 0 |
| 평균 이동 속도 15 km/h 미만 | 1/1 |
| 5초 이내 실제 반대 방향 차선 변경 | 0 |
| 주변 차량이 있었던 스텝 | 1599/1616 (98.9%) |
| 주변 차량이 한 번도 없었던 판 | 0/1 |

| 다음 정류장 | edge | 위치 구간 시작 (m) | 방향 반전 | 대표 시드 | SUMO 시각 (s) |
|---|---|---:|---:|---:|---:|

### klppo1_s6_s0_step100k (30/30판)

| 지표 | 값 |
|---|---:|
| episodes | 30 |
| official_complete | 18/30 (60.0%) |
| perfect_complete | 18/30 (60.0%) |
| collision_count | 9 |
| collision_rate | 9/30 (30.0%) |
| wrong_lane_count | 0 |
| timeout_count | 2 |
| inf_wait_episodes | 19 |
| wait_timeout_wrong_lane_episodes | 19 |
| green_stop_max_sec | 1438 |
| green_stop_time_mean | 0.14341 |
| green_stop_events_mean | 42.2 |
| moving_avg_speed_kmh | 18.15 |
| avg_speed_kmh | 9.5017 |
| teleports | 0 |
| sim_errors | 0 |
| red_light | 0.033333 |
| yield_steps | 0 |
| intervention | 0.020833 |
| reasons | {'completed': 18, 'collision': 9, 'red_light': 1, 'timeout': 2} |
| collision | 0.3 |
| timeout | 0.066667 |
| stop_success_mean | 3.8333 |
| avg_speed | 2.6419 |
| bs02_lc_fail | 0 |
| zone_max_overspeed | 0.51991 |
| zone_overspeed_step_rate | 0.014843 |
| 평균 이동 속도 15 km/h 미만 | 2/30 |
| 5초 이내 실제 반대 방향 차선 변경 | 1666 |
| 주변 차량이 있었던 스텝 | 40275/43181 (93.3%) |
| 주변 차량이 한 번도 없었던 판 | 0/30 |

| 다음 정류장 | edge | 위치 구간 시작 (m) | 방향 반전 | 대표 시드 | SUMO 시각 (s) |
|---|---|---:|---:|---:|---:|
| bs_03 | e908646367_3 | 30 | 1389 | 900101 | 232.0 |
| bs_03 | e908646367_3 | 40 | 219 | 900102 | 324.5 |
| bs_03 | e908646367_3 | 20 | 35 | 900101 | 230.5 |
| bs_04 | e908646367_6 | 60 | 20 | 900100 | 424.5 |
| bs_04 | e908646367_6 | 50 | 2 | 900116 | 864.0 |
| bs_04 | e908646367_6 | 40 | 1 | 900104 | 486.5 |

### klppo1_s6_s0_step125k (30/30판)

| 지표 | 값 |
|---|---:|
| episodes | 30 |
| official_complete | 19/30 (63.3%) |
| perfect_complete | 19/30 (63.3%) |
| collision_count | 5 |
| collision_rate | 5/30 (16.7%) |
| wrong_lane_count | 0 |
| timeout_count | 2 |
| inf_wait_episodes | 23 |
| wait_timeout_wrong_lane_episodes | 23 |
| green_stop_max_sec | 1340 |
| green_stop_time_mean | 0.14261 |
| green_stop_events_mean | 48.9 |
| moving_avg_speed_kmh | 18.594 |
| avg_speed_kmh | 9.5375 |
| teleports | 0 |
| sim_errors | 0 |
| red_light | 0.13333 |
| yield_steps | 1 |
| intervention | 0.020136 |
| reasons | {'completed': 19, 'collision': 5, 'red_light': 4, 'timeout': 2} |
| collision | 0.16667 |
| timeout | 0.066667 |
| stop_success_mean | 4.0333 |
| avg_speed | 2.6518 |
| bs02_lc_fail | 0 |
| zone_max_overspeed | 0.53034 |
| zone_overspeed_step_rate | 0.013443 |
| 평균 이동 속도 15 km/h 미만 | 0/30 |
| 5초 이내 실제 반대 방향 차선 변경 | 2076 |
| 주변 차량이 있었던 스텝 | 41663/44874 (92.8%) |
| 주변 차량이 한 번도 없었던 판 | 0/30 |

| 다음 정류장 | edge | 위치 구간 시작 (m) | 방향 반전 | 대표 시드 | SUMO 시각 (s) |
|---|---|---:|---:|---:|---:|
| bs_03 | e908646367_3 | 30 | 1534 | 900103 | 385.5 |
| bs_03 | e908646367_3 | 40 | 473 | 900101 | 233.5 |
| bs_03 | e908646367_3 | 20 | 37 | 900103 | 383.0 |
| bs_04 | e908646367_6 | 60 | 30 | 900102 | 514.0 |
| bs_04 | e908646367_6 | 30 | 2 | 900104 | 454.5 |

### klppo1_s6_s0_step150k (30/30판)

| 지표 | 값 |
|---|---:|
| episodes | 30 |
| official_complete | 16/30 (53.3%) |
| perfect_complete | 16/30 (53.3%) |
| collision_count | 12 |
| collision_rate | 12/30 (40.0%) |
| wrong_lane_count | 0 |
| timeout_count | 1 |
| inf_wait_episodes | 23 |
| wait_timeout_wrong_lane_episodes | 23 |
| green_stop_max_sec | 1391.5 |
| green_stop_time_mean | 0.11922 |
| green_stop_events_mean | 52.133 |
| moving_avg_speed_kmh | 18.285 |
| avg_speed_kmh | 9.298 |
| teleports | 0 |
| sim_errors | 0 |
| red_light | 0.033333 |
| yield_steps | 0 |
| intervention | 0.023053 |
| reasons | {'completed': 16, 'collision': 12, 'red_light': 1, 'timeout': 1} |
| collision | 0.4 |
| timeout | 0.033333 |
| stop_success_mean | 3.7333 |
| avg_speed | 2.5852 |
| bs02_lc_fail | 0 |
| zone_max_overspeed | 0.50137 |
| zone_overspeed_step_rate | 0.023481 |
| 평균 이동 속도 15 km/h 미만 | 2/30 |
| 5초 이내 실제 반대 방향 차선 변경 | 2329 |
| 주변 차량이 있었던 스텝 | 39940/42669 (93.6%) |
| 주변 차량이 한 번도 없었던 판 | 0/30 |

| 다음 정류장 | edge | 위치 구간 시작 (m) | 방향 반전 | 대표 시드 | SUMO 시각 (s) |
|---|---|---:|---:|---:|---:|
| bs_03 | e908646367_3 | 30 | 1580 | 900101 | 232.5 |
| bs_03 | e908646367_3 | 40 | 678 | 900102 | 324.5 |
| bs_04 | e908646367_6 | 60 | 28 | 900100 | 424.0 |
| bs_03 | e908646367_3 | 20 | 27 | 900101 | 231.0 |
| bs_04 | e908646367_6 | 50 | 7 | 900102 | 511.0 |
| bs_04 | e908646367_6 | 30 | 6 | 900101 | 452.5 |
| bs_04 | e908646367_6 | 40 | 3 | 900102 | 509.0 |

### klppo1_s6_s0_step175k (30/30판)

| 지표 | 값 |
|---|---:|
| episodes | 30 |
| official_complete | 20/30 (66.7%) |
| perfect_complete | 20/30 (66.7%) |
| collision_count | 8 |
| collision_rate | 8/30 (26.7%) |
| wrong_lane_count | 0 |
| timeout_count | 1 |
| inf_wait_episodes | 23 |
| wait_timeout_wrong_lane_episodes | 23 |
| green_stop_max_sec | 32.5 |
| green_stop_time_mean | 0.095481 |
| green_stop_events_mean | 50.967 |
| moving_avg_speed_kmh | 17.869 |
| avg_speed_kmh | 9.7141 |
| teleports | 0 |
| sim_errors | 0 |
| red_light | 0.033333 |
| yield_steps | 0 |
| intervention | 0.0094831 |
| reasons | {'completed': 20, 'collision': 8, 'timeout': 1, 'red_light': 1} |
| collision | 0.26667 |
| timeout | 0.033333 |
| stop_success_mean | 4.2 |
| avg_speed | 2.7016 |
| bs02_lc_fail | 1 |
| zone_max_overspeed | 0.66202 |
| zone_overspeed_step_rate | 0.015471 |
| 평균 이동 속도 15 km/h 미만 | 1/30 |
| 5초 이내 실제 반대 방향 차선 변경 | 2123 |
| 주변 차량이 있었던 스텝 | 41886/44951 (93.2%) |
| 주변 차량이 한 번도 없었던 판 | 0/30 |

| 다음 정류장 | edge | 위치 구간 시작 (m) | 방향 반전 | 대표 시드 | SUMO 시각 (s) |
|---|---|---:|---:|---:|---:|
| bs_03 | e908646367_3 | 30 | 1396 | 900103 | 411.0 |
| bs_03 | e908646367_3 | 40 | 674 | 900101 | 227.0 |
| bs_03 | e908646367_3 | 20 | 34 | 900103 | 409.0 |
| bs_04 | e908646367_6 | 50 | 8 | 900116 | 875.5 |
| bs_04 | e908646367_6 | 60 | 6 | 900100 | 424.5 |
| bs_05 | e218924602_6 | 60 | 3 | 900110 | 833.5 |
| bs_05 | e218924602_6 | 70 | 1 | 900110 | 835.0 |
| bs_04 | e908646367_6 | 40 | 1 | 900117 | 472.0 |

### klppo1_s6_s0_step200k (30/30판)

| 지표 | 값 |
|---|---:|
| episodes | 30 |
| official_complete | 20/30 (66.7%) |
| perfect_complete | 20/30 (66.7%) |
| collision_count | 6 |
| collision_rate | 6/30 (20.0%) |
| wrong_lane_count | 0 |
| timeout_count | 1 |
| inf_wait_episodes | 20 |
| wait_timeout_wrong_lane_episodes | 20 |
| green_stop_max_sec | 1262.5 |
| green_stop_time_mean | 0.11242 |
| green_stop_events_mean | 49.233 |
| moving_avg_speed_kmh | 18.57 |
| avg_speed_kmh | 9.9256 |
| teleports | 0 |
| sim_errors | 0 |
| red_light | 0.1 |
| yield_steps | 0 |
| intervention | 0.024723 |
| reasons | {'completed': 20, 'collision': 6, 'red_light': 3, 'timeout': 1} |
| collision | 0.2 |
| timeout | 0.033333 |
| stop_success_mean | 4.1667 |
| avg_speed | 2.7597 |
| bs02_lc_fail | 0 |
| zone_max_overspeed | 0.64515 |
| zone_overspeed_step_rate | 0.01883 |
| 평균 이동 속도 15 km/h 미만 | 1/30 |
| 5초 이내 실제 반대 방향 차선 변경 | 1969 |
| 주변 차량이 있었던 스텝 | 39487/42742 (92.4%) |
| 주변 차량이 한 번도 없었던 판 | 0/30 |

| 다음 정류장 | edge | 위치 구간 시작 (m) | 방향 반전 | 대표 시드 | SUMO 시각 (s) |
|---|---|---:|---:|---:|---:|
| bs_03 | e908646367_3 | 30 | 1226 | 900101 | 224.5 |
| bs_03 | e908646367_3 | 40 | 689 | 900101 | 225.0 |
| bs_03 | e908646367_3 | 20 | 34 | 900103 | 370.5 |
| bs_04 | e908646367_6 | 60 | 14 | 900100 | 425.0 |
| bs_04 | e908646367_6 | 30 | 4 | 900104 | 439.5 |
| bs_04 | e908646367_6 | 40 | 2 | 900104 | 441.0 |

### klppo1_s6_s0_step225k (30/30판)

| 지표 | 값 |
|---|---:|
| episodes | 30 |
| official_complete | 18/30 (60.0%) |
| perfect_complete | 18/30 (60.0%) |
| collision_count | 10 |
| collision_rate | 10/30 (33.3%) |
| wrong_lane_count | 0 |
| timeout_count | 1 |
| inf_wait_episodes | 19 |
| wait_timeout_wrong_lane_episodes | 19 |
| green_stop_max_sec | 40 |
| green_stop_time_mean | 0.090912 |
| green_stop_events_mean | 47.367 |
| moving_avg_speed_kmh | 18.476 |
| avg_speed_kmh | 9.9205 |
| teleports | 0 |
| sim_errors | 0 |
| red_light | 0.033333 |
| yield_steps | 1 |
| intervention | 0.021921 |
| reasons | {'completed': 18, 'collision': 10, 'red_light': 1, 'timeout': 1} |
| collision | 0.33333 |
| timeout | 0.033333 |
| stop_success_mean | 4.0333 |
| avg_speed | 2.7585 |
| bs02_lc_fail | 0 |
| zone_max_overspeed | 0.75937 |
| zone_overspeed_step_rate | 0.013828 |
| 평균 이동 속도 15 km/h 미만 | 1/30 |
| 5초 이내 실제 반대 방향 차선 변경 | 1916 |
| 주변 차량이 있었던 스텝 | 38655/41541 (93.1%) |
| 주변 차량이 한 번도 없었던 판 | 0/30 |

| 다음 정류장 | edge | 위치 구간 시작 (m) | 방향 반전 | 대표 시드 | SUMO 시각 (s) |
|---|---|---:|---:|---:|---:|
| bs_03 | e908646367_3 | 30 | 1557 | 900102 | 322.5 |
| bs_03 | e908646367_3 | 40 | 318 | 900109 | 514.5 |
| bs_03 | e908646367_3 | 20 | 21 | 900109 | 498.0 |
| bs_04 | e908646367_6 | 60 | 18 | 900100 | 423.0 |
| bs_04 | e908646367_6 | 50 | 2 | 900128 | 424.0 |

### klppo1_s6_s0_step250k (30/30판)

| 지표 | 값 |
|---|---:|
| episodes | 30 |
| official_complete | 19/30 (63.3%) |
| perfect_complete | 19/30 (63.3%) |
| collision_count | 8 |
| collision_rate | 8/30 (26.7%) |
| wrong_lane_count | 0 |
| timeout_count | 3 |
| inf_wait_episodes | 25 |
| wait_timeout_wrong_lane_episodes | 25 |
| green_stop_max_sec | 1392 |
| green_stop_time_mean | 0.13532 |
| green_stop_events_mean | 52.267 |
| moving_avg_speed_kmh | 18.3 |
| avg_speed_kmh | 8.9907 |
| teleports | 0 |
| sim_errors | 0 |
| red_light | 0 |
| yield_steps | 0 |
| intervention | 0.014925 |
| reasons | {'completed': 19, 'collision': 8, 'timeout': 3} |
| collision | 0.26667 |
| timeout | 0.1 |
| stop_success_mean | 4.1333 |
| avg_speed | 2.4995 |
| bs02_lc_fail | 0 |
| zone_max_overspeed | 0.88686 |
| zone_overspeed_step_rate | 0.024209 |
| 평균 이동 속도 15 km/h 미만 | 2/30 |
| 5초 이내 실제 반대 방향 차선 변경 | 2268 |
| 주변 차량이 있었던 스텝 | 46565/49475 (94.1%) |
| 주변 차량이 한 번도 없었던 판 | 0/30 |

| 다음 정류장 | edge | 위치 구간 시작 (m) | 방향 반전 | 대표 시드 | SUMO 시각 (s) |
|---|---|---:|---:|---:|---:|
| bs_03 | e908646367_3 | 30 | 1673 | 900105 | 337.5 |
| bs_03 | e908646367_3 | 40 | 532 | 900101 | 226.5 |
| bs_03 | e908646367_3 | 20 | 37 | 900105 | 337.0 |
| bs_04 | e908646367_6 | 60 | 14 | 900100 | 425.5 |
| bs_04 | e908646367_6 | 50 | 8 | 900128 | 422.0 |
| bs_04 | e908646367_6 | 40 | 2 | 900117 | 448.5 |
| bs_04 | e908646367_6 | 30 | 1 | 900117 | 448.0 |
| bs_05 | e218924602_6 | 80 | 1 | 900120 | 619.0 |

### klppo1_s6_s0_step25k (30/30판)

| 지표 | 값 |
|---|---:|
| episodes | 30 |
| official_complete | 18/30 (60.0%) |
| perfect_complete | 18/30 (60.0%) |
| collision_count | 7 |
| collision_rate | 7/30 (23.3%) |
| wrong_lane_count | 0 |
| timeout_count | 3 |
| inf_wait_episodes | 25 |
| wait_timeout_wrong_lane_episodes | 25 |
| green_stop_max_sec | 1411.5 |
| green_stop_time_mean | 0.14069 |
| green_stop_events_mean | 53.033 |
| moving_avg_speed_kmh | 18.55 |
| avg_speed_kmh | 9.0556 |
| teleports | 0 |
| sim_errors | 0 |
| red_light | 0.066667 |
| yield_steps | 1 |
| intervention | 0.0093513 |
| reasons | {'completed': 18, 'collision': 7, 'red_light': 2, 'timeout': 3} |
| collision | 0.23333 |
| timeout | 0.1 |
| stop_success_mean | 4.2 |
| avg_speed | 2.5175 |
| bs02_lc_fail | 0 |
| zone_max_overspeed | 0.59451 |
| zone_overspeed_step_rate | 0.011752 |
| 평균 이동 속도 15 km/h 미만 | 1/30 |
| 5초 이내 실제 반대 방향 차선 변경 | 2261 |
| 주변 차량이 있었던 스텝 | 45699/49004 (93.3%) |
| 주변 차량이 한 번도 없었던 판 | 0/30 |

| 다음 정류장 | edge | 위치 구간 시작 (m) | 방향 반전 | 대표 시드 | SUMO 시각 (s) |
|---|---|---:|---:|---:|---:|
| bs_03 | e908646367_3 | 30 | 1901 | 900108 | 324.5 |
| bs_03 | e908646367_3 | 40 | 301 | 900101 | 276.0 |
| bs_03 | e908646367_3 | 20 | 31 | 900110 | 412.5 |
| bs_04 | e908646367_6 | 60 | 24 | 900100 | 423.0 |
| bs_04 | e908646367_6 | 30 | 2 | 900101 | 451.5 |
| bs_04 | e908646367_6 | 50 | 2 | 900128 | 416.5 |

### klppo1_s6_s0_step275k (30/30판)

| 지표 | 값 |
|---|---:|
| episodes | 30 |
| official_complete | 18/30 (60.0%) |
| perfect_complete | 18/30 (60.0%) |
| collision_count | 10 |
| collision_rate | 10/30 (33.3%) |
| wrong_lane_count | 0 |
| timeout_count | 1 |
| inf_wait_episodes | 22 |
| wait_timeout_wrong_lane_episodes | 22 |
| green_stop_max_sec | 29 |
| green_stop_time_mean | 0.089559 |
| green_stop_events_mean | 48 |
| moving_avg_speed_kmh | 18.355 |
| avg_speed_kmh | 9.5327 |
| teleports | 0 |
| sim_errors | 0 |
| red_light | 0.033333 |
| yield_steps | 1 |
| intervention | 0.018958 |
| reasons | {'completed': 18, 'collision': 10, 'red_light': 1, 'timeout': 1} |
| collision | 0.33333 |
| timeout | 0.033333 |
| stop_success_mean | 4.1 |
| avg_speed | 2.6504 |
| bs02_lc_fail | 0 |
| zone_max_overspeed | 0.70566 |
| zone_overspeed_step_rate | 0.017366 |
| 평균 이동 속도 15 km/h 미만 | 1/30 |
| 5초 이내 실제 반대 방향 차선 변경 | 1939 |
| 주변 차량이 있었던 스텝 | 41343/44127 (93.7%) |
| 주변 차량이 한 번도 없었던 판 | 0/30 |

| 다음 정류장 | edge | 위치 구간 시작 (m) | 방향 반전 | 대표 시드 | SUMO 시각 (s) |
|---|---|---:|---:|---:|---:|
| bs_03 | e908646367_3 | 30 | 1549 | 900102 | 321.5 |
| bs_03 | e908646367_3 | 40 | 322 | 900101 | 275.0 |
| bs_03 | e908646367_3 | 20 | 29 | 900110 | 427.0 |
| bs_04 | e908646367_6 | 60 | 15 | 900100 | 425.0 |
| bs_04 | e908646367_6 | 50 | 14 | 900116 | 873.0 |
| bs_04 | e908646367_6 | 30 | 4 | 900101 | 449.5 |
| bs_05 | e218924602_6 | 60 | 4 | 900126 | 963.0 |
| bs_04 | e908646367_6 | 40 | 2 | 900101 | 450.5 |

### klppo1_s6_s0_step300k (30/30판)

| 지표 | 값 |
|---|---:|
| episodes | 30 |
| official_complete | 22/30 (73.3%) |
| perfect_complete | 22/30 (73.3%) |
| collision_count | 6 |
| collision_rate | 6/30 (20.0%) |
| wrong_lane_count | 0 |
| timeout_count | 0 |
| inf_wait_episodes | 23 |
| wait_timeout_wrong_lane_episodes | 23 |
| green_stop_max_sec | 43.5 |
| green_stop_time_mean | 0.094789 |
| green_stop_events_mean | 49.6 |
| moving_avg_speed_kmh | 18.766 |
| avg_speed_kmh | 10.032 |
| teleports | 0 |
| sim_errors | 0 |
| red_light | 0.066667 |
| yield_steps | 1 |
| intervention | 0.018259 |
| reasons | {'completed': 22, 'collision': 6, 'red_light': 2} |
| collision | 0.2 |
| timeout | 0 |
| stop_success_mean | 4.2 |
| avg_speed | 2.7896 |
| bs02_lc_fail | 1 |
| zone_max_overspeed | 0.76556 |
| zone_overspeed_step_rate | 0.027762 |
| 평균 이동 속도 15 km/h 미만 | 3/30 |
| 5초 이내 실제 반대 방향 차선 변경 | 2056 |
| 주변 차량이 있었던 스텝 | 40167/42859 (93.7%) |
| 주변 차량이 한 번도 없었던 판 | 0/30 |

| 다음 정류장 | edge | 위치 구간 시작 (m) | 방향 반전 | 대표 시드 | SUMO 시각 (s) |
|---|---|---:|---:|---:|---:|
| bs_03 | e908646367_3 | 30 | 1264 | 900106 | 352.5 |
| bs_03 | e908646367_3 | 40 | 721 | 900101 | 227.5 |
| bs_04 | e908646367_6 | 60 | 32 | 900100 | 426.5 |
| bs_03 | e908646367_3 | 20 | 19 | 900108 | 323.5 |
| bs_04 | e908646367_6 | 50 | 11 | 900101 | 454.5 |
| bs_04 | e908646367_6 | 40 | 7 | 900101 | 452.0 |
| bs_04 | e908646367_6 | 30 | 2 | 900101 | 451.5 |

### klppo1_s6_s0_step50k (30/30판)

| 지표 | 값 |
|---|---:|
| episodes | 30 |
| official_complete | 15/30 (50.0%) |
| perfect_complete | 15/30 (50.0%) |
| collision_count | 10 |
| collision_rate | 10/30 (33.3%) |
| wrong_lane_count | 0 |
| timeout_count | 1 |
| inf_wait_episodes | 20 |
| wait_timeout_wrong_lane_episodes | 20 |
| green_stop_max_sec | 1354.5 |
| green_stop_time_mean | 0.12897 |
| green_stop_events_mean | 53.967 |
| moving_avg_speed_kmh | 18.647 |
| avg_speed_kmh | 9.6857 |
| teleports | 0 |
| sim_errors | 0 |
| red_light | 0.13333 |
| yield_steps | 0 |
| intervention | 0.0099141 |
| reasons | {'collision': 10, 'completed': 15, 'red_light': 4, 'timeout': 1} |
| collision | 0.33333 |
| timeout | 0.033333 |
| stop_success_mean | 3.7 |
| avg_speed | 2.6937 |
| bs02_lc_fail | 0 |
| zone_max_overspeed | 0.53194 |
| zone_overspeed_step_rate | 0.018203 |
| 평균 이동 속도 15 km/h 미만 | 2/30 |
| 5초 이내 실제 반대 방향 차선 변경 | 2352 |
| 주변 차량이 있었던 스텝 | 37106/39972 (92.8%) |
| 주변 차량이 한 번도 없었던 판 | 0/30 |

| 다음 정류장 | edge | 위치 구간 시작 (m) | 방향 반전 | 대표 시드 | SUMO 시각 (s) |
|---|---|---:|---:|---:|---:|
| bs_03 | e908646367_3 | 30 | 1269 | 900101 | 225.0 |
| bs_03 | e908646367_3 | 40 | 1014 | 900101 | 225.5 |
| bs_03 | e908646367_3 | 20 | 34 | 900110 | 406.5 |
| bs_04 | e908646367_6 | 60 | 24 | 900100 | 426.0 |
| bs_04 | e908646367_6 | 50 | 8 | 900127 | 842.0 |
| bs_05 | e218924602_6 | 60 | 2 | 900110 | 823.5 |
| bs_05 | e218924602_6 | 80 | 1 | 900110 | 831.5 |

### klppo1_s6_s0_step75k (30/30판)

| 지표 | 값 |
|---|---:|
| episodes | 30 |
| official_complete | 19/30 (63.3%) |
| perfect_complete | 19/30 (63.3%) |
| collision_count | 7 |
| collision_rate | 7/30 (23.3%) |
| wrong_lane_count | 0 |
| timeout_count | 2 |
| inf_wait_episodes | 21 |
| wait_timeout_wrong_lane_episodes | 21 |
| green_stop_max_sec | 1393.5 |
| green_stop_time_mean | 0.13573 |
| green_stop_events_mean | 55.733 |
| moving_avg_speed_kmh | 18.284 |
| avg_speed_kmh | 9.2792 |
| teleports | 0 |
| sim_errors | 0 |
| red_light | 0.066667 |
| yield_steps | 1 |
| intervention | 0.012069 |
| reasons | {'completed': 19, 'collision': 7, 'red_light': 2, 'timeout': 2} |
| collision | 0.23333 |
| timeout | 0.066667 |
| stop_success_mean | 4.2667 |
| avg_speed | 2.5797 |
| bs02_lc_fail | 0 |
| zone_max_overspeed | 0.54047 |
| zone_overspeed_step_rate | 0.018893 |
| 평균 이동 속도 15 km/h 미만 | 1/30 |
| 5초 이내 실제 반대 방향 차선 변경 | 2432 |
| 주변 차량이 있었던 스텝 | 44535/47669 (93.4%) |
| 주변 차량이 한 번도 없었던 판 | 0/30 |

| 다음 정류장 | edge | 위치 구간 시작 (m) | 방향 반전 | 대표 시드 | SUMO 시각 (s) |
|---|---|---:|---:|---:|---:|
| bs_03 | e908646367_3 | 30 | 1457 | 900105 | 351.5 |
| bs_03 | e908646367_3 | 40 | 940 | 900101 | 275.5 |
| bs_03 | e908646367_3 | 20 | 18 | 900112 | 271.0 |
| bs_04 | e908646367_6 | 60 | 9 | 900100 | 426.5 |
| bs_04 | e908646367_6 | 50 | 4 | 900128 | 421.5 |
| bs_04 | e908646367_6 | 70 | 2 | 900105 | 478.5 |
| bs_04 | e908646367_6 | 30 | 1 | 900125 | 345.5 |
| bs_04 | e908646367_6 | 40 | 1 | 900125 | 346.0 |

### klppo1_s6_s1_step100k (30/30판)

| 지표 | 값 |
|---|---:|
| episodes | 30 |
| official_complete | 16/30 (53.3%) |
| perfect_complete | 16/30 (53.3%) |
| collision_count | 7 |
| collision_rate | 7/30 (23.3%) |
| wrong_lane_count | 0 |
| timeout_count | 3 |
| inf_wait_episodes | 24 |
| wait_timeout_wrong_lane_episodes | 24 |
| green_stop_max_sec | 1394.5 |
| green_stop_time_mean | 0.11525 |
| green_stop_events_mean | 56.067 |
| moving_avg_speed_kmh | 18.157 |
| avg_speed_kmh | 8.7718 |
| teleports | 0 |
| sim_errors | 0 |
| red_light | 0.13333 |
| yield_steps | 0 |
| intervention | 0.012567 |
| reasons | {'completed': 16, 'red_light': 4, 'timeout': 3, 'collision': 7} |
| collision | 0.23333 |
| timeout | 0.1 |
| stop_success_mean | 4.0333 |
| avg_speed | 2.4388 |
| bs02_lc_fail | 0 |
| zone_max_overspeed | 0.54984 |
| zone_overspeed_step_rate | 0.011421 |
| 평균 이동 속도 15 km/h 미만 | 1/30 |
| 5초 이내 실제 반대 방향 차선 변경 | 2380 |
| 주변 차량이 있었던 스텝 | 46104/49282 (93.6%) |
| 주변 차량이 한 번도 없었던 판 | 0/30 |

| 다음 정류장 | edge | 위치 구간 시작 (m) | 방향 반전 | 대표 시드 | SUMO 시각 (s) |
|---|---|---:|---:|---:|---:|
| bs_03 | e908646367_3 | 30 | 1465 | 900100 | 245.5 |
| bs_03 | e908646367_3 | 40 | 831 | 900102 | 324.0 |
| bs_03 | e908646367_3 | 20 | 44 | 900100 | 244.5 |
| bs_04 | e908646367_6 | 60 | 22 | 900100 | 423.0 |
| bs_04 | e908646367_6 | 50 | 8 | 900116 | 880.0 |
| bs_04 | e908646367_6 | 40 | 5 | 900104 | 439.5 |
| bs_04 | e908646367_6 | 30 | 3 | 900104 | 438.5 |
| bs_05 | e218924602_6 | 60 | 2 | 900101 | 696.5 |

### klppo1_s6_s1_step125k (30/30판)

| 지표 | 값 |
|---|---:|
| episodes | 30 |
| official_complete | 16/30 (53.3%) |
| perfect_complete | 16/30 (53.3%) |
| collision_count | 12 |
| collision_rate | 12/30 (40.0%) |
| wrong_lane_count | 0 |
| timeout_count | 0 |
| inf_wait_episodes | 20 |
| wait_timeout_wrong_lane_episodes | 20 |
| green_stop_max_sec | 40.5 |
| green_stop_time_mean | 0.10729 |
| green_stop_events_mean | 47.967 |
| moving_avg_speed_kmh | 18.772 |
| avg_speed_kmh | 10.054 |
| teleports | 0 |
| sim_errors | 0 |
| red_light | 0.066667 |
| yield_steps | 0 |
| intervention | 0.023631 |
| reasons | {'completed': 16, 'collision': 12, 'red_light': 2} |
| collision | 0.4 |
| timeout | 0 |
| stop_success_mean | 3.8333 |
| avg_speed | 2.7961 |
| bs02_lc_fail | 0 |
| zone_max_overspeed | 0.34314 |
| zone_overspeed_step_rate | 0.011037 |
| 평균 이동 속도 15 km/h 미만 | 2/30 |
| 5초 이내 실제 반대 방향 차선 변경 | 1897 |
| 주변 차량이 있었던 스텝 | 34837/37989 (91.7%) |
| 주변 차량이 한 번도 없었던 판 | 0/30 |

| 다음 정류장 | edge | 위치 구간 시작 (m) | 방향 반전 | 대표 시드 | SUMO 시각 (s) |
|---|---|---:|---:|---:|---:|
| bs_03 | e908646367_3 | 30 | 1080 | 900103 | 391.0 |
| bs_03 | e908646367_3 | 40 | 768 | 900102 | 324.5 |
| bs_03 | e908646367_3 | 20 | 33 | 900103 | 388.5 |
| bs_04 | e908646367_6 | 60 | 11 | 900100 | 424.0 |
| bs_04 | e908646367_6 | 30 | 4 | 900101 | 301.5 |
| bs_04 | e908646367_6 | 40 | 1 | 900101 | 303.0 |

### klppo1_s6_s1_step150k (30/30판)

| 지표 | 값 |
|---|---:|
| episodes | 30 |
| official_complete | 15/30 (50.0%) |
| perfect_complete | 15/30 (50.0%) |
| collision_count | 11 |
| collision_rate | 11/30 (36.7%) |
| wrong_lane_count | 0 |
| timeout_count | 2 |
| inf_wait_episodes | 22 |
| wait_timeout_wrong_lane_episodes | 22 |
| green_stop_max_sec | 1369.5 |
| green_stop_time_mean | 0.14105 |
| green_stop_events_mean | 39.733 |
| moving_avg_speed_kmh | 18.178 |
| avg_speed_kmh | 9.1924 |
| teleports | 0 |
| sim_errors | 0 |
| red_light | 0.066667 |
| yield_steps | 0 |
| intervention | 0.012305 |
| reasons | {'completed': 15, 'collision': 11, 'red_light': 2, 'timeout': 2} |
| collision | 0.36667 |
| timeout | 0.066667 |
| stop_success_mean | 3.8 |
| avg_speed | 2.5558 |
| bs02_lc_fail | 0 |
| zone_max_overspeed | 0.41702 |
| zone_overspeed_step_rate | 0.0068345 |
| 평균 이동 속도 15 km/h 미만 | 1/30 |
| 5초 이내 실제 반대 방향 차선 변경 | 1419 |
| 주변 차량이 있었던 스텝 | 41060/43994 (93.3%) |
| 주변 차량이 한 번도 없었던 판 | 0/30 |

| 다음 정류장 | edge | 위치 구간 시작 (m) | 방향 반전 | 대표 시드 | SUMO 시각 (s) |
|---|---|---:|---:|---:|---:|
| bs_03 | e908646367_3 | 30 | 894 | 900101 | 235.5 |
| bs_03 | e908646367_3 | 40 | 467 | 900102 | 324.5 |
| bs_04 | e908646367_6 | 60 | 22 | 900100 | 426.0 |
| bs_03 | e908646367_3 | 20 | 19 | 900101 | 232.0 |
| bs_04 | e908646367_6 | 40 | 8 | 900101 | 476.5 |
| bs_04 | e908646367_6 | 30 | 6 | 900104 | 446.5 |
| bs_04 | e908646367_6 | 50 | 2 | 900101 | 480.5 |
| bs_05 | e218924602_6 | 80 | 1 | 900120 | 619.0 |

### klppo1_s6_s1_step175k (30/30판)

| 지표 | 값 |
|---|---:|
| episodes | 30 |
| official_complete | 21/30 (70.0%) |
| perfect_complete | 21/30 (70.0%) |
| collision_count | 6 |
| collision_rate | 6/30 (20.0%) |
| wrong_lane_count | 0 |
| timeout_count | 2 |
| inf_wait_episodes | 22 |
| wait_timeout_wrong_lane_episodes | 22 |
| green_stop_max_sec | 1410 |
| green_stop_time_mean | 0.15052 |
| green_stop_events_mean | 59.767 |
| moving_avg_speed_kmh | 18.466 |
| avg_speed_kmh | 9.2568 |
| teleports | 0 |
| sim_errors | 0 |
| red_light | 0.033333 |
| yield_steps | 0 |
| intervention | 0.014775 |
| reasons | {'completed': 21, 'collision': 6, 'red_light': 1, 'timeout': 2} |
| collision | 0.2 |
| timeout | 0.066667 |
| stop_success_mean | 4.1667 |
| avg_speed | 2.5737 |
| bs02_lc_fail | 0 |
| zone_max_overspeed | 0.32832 |
| zone_overspeed_step_rate | 0.0083612 |
| 평균 이동 속도 15 km/h 미만 | 1/30 |
| 5초 이내 실제 반대 방향 차선 변경 | 2543 |
| 주변 차량이 있었던 스텝 | 45104/48691 (92.6%) |
| 주변 차량이 한 번도 없었던 판 | 0/30 |

| 다음 정류장 | edge | 위치 구간 시작 (m) | 방향 반전 | 대표 시드 | SUMO 시각 (s) |
|---|---|---:|---:|---:|---:|
| bs_03 | e908646367_3 | 40 | 1300 | 900101 | 227.0 |
| bs_03 | e908646367_3 | 30 | 1186 | 900103 | 406.5 |
| bs_03 | e908646367_3 | 20 | 33 | 900103 | 404.5 |
| bs_04 | e908646367_6 | 60 | 13 | 900100 | 426.5 |
| bs_04 | e908646367_6 | 50 | 4 | 900116 | 865.5 |
| bs_04 | e908646367_6 | 40 | 3 | 900117 | 451.5 |
| bs_05 | e218924602_6 | 60 | 2 | 900110 | 849.0 |
| bs_04 | e908646367_6 | 30 | 2 | 900117 | 450.5 |

### klppo1_s6_s1_step225k (30/30판)

| 지표 | 값 |
|---|---:|
| episodes | 30 |
| official_complete | 16/30 (53.3%) |
| perfect_complete | 16/30 (53.3%) |
| collision_count | 9 |
| collision_rate | 9/30 (30.0%) |
| wrong_lane_count | 0 |
| timeout_count | 1 |
| inf_wait_episodes | 20 |
| wait_timeout_wrong_lane_episodes | 20 |
| green_stop_max_sec | 1368.5 |
| green_stop_time_mean | 0.1196 |
| green_stop_events_mean | 47.1 |
| moving_avg_speed_kmh | 18.204 |
| avg_speed_kmh | 9.7995 |
| teleports | 0 |
| sim_errors | 0 |
| red_light | 0.13333 |
| yield_steps | 0 |
| intervention | 0.017844 |
| reasons | {'completed': 16, 'red_light': 4, 'collision': 9, 'timeout': 1} |
| collision | 0.3 |
| timeout | 0.033333 |
| stop_success_mean | 3.8667 |
| avg_speed | 2.725 |
| bs02_lc_fail | 0 |
| zone_max_overspeed | 0.28513 |
| zone_overspeed_step_rate | 0.0053908 |
| 평균 이동 속도 15 km/h 미만 | 3/30 |
| 5초 이내 실제 반대 방향 차선 변경 | 1871 |
| 주변 차량이 있었던 스텝 | 38729/41500 (93.3%) |
| 주변 차량이 한 번도 없었던 판 | 0/30 |

| 다음 정류장 | edge | 위치 구간 시작 (m) | 방향 반전 | 대표 시드 | SUMO 시각 (s) |
|---|---|---:|---:|---:|---:|
| bs_03 | e908646367_3 | 30 | 1118 | 900106 | 355.0 |
| bs_03 | e908646367_3 | 40 | 705 | 900102 | 324.5 |
| bs_03 | e908646367_3 | 20 | 31 | 900106 | 354.5 |
| bs_04 | e908646367_6 | 60 | 15 | 900100 | 424.5 |
| bs_04 | e908646367_6 | 50 | 1 | 900111 | 536.0 |
| bs_05 | e218924602_6 | 30 | 1 | 900112 | 573.0 |

### klppo1_s6_s1_step250k (30/30판)

| 지표 | 값 |
|---|---:|
| episodes | 30 |
| official_complete | 19/30 (63.3%) |
| perfect_complete | 19/30 (63.3%) |
| collision_count | 9 |
| collision_rate | 9/30 (30.0%) |
| wrong_lane_count | 0 |
| timeout_count | 1 |
| inf_wait_episodes | 22 |
| wait_timeout_wrong_lane_episodes | 22 |
| green_stop_max_sec | 1394.5 |
| green_stop_time_mean | 0.11583 |
| green_stop_events_mean | 58 |
| moving_avg_speed_kmh | 17.852 |
| avg_speed_kmh | 9.3996 |
| teleports | 0 |
| sim_errors | 0 |
| red_light | 0.033333 |
| yield_steps | 1 |
| intervention | 0.016124 |
| reasons | {'completed': 19, 'collision': 9, 'red_light': 1, 'timeout': 1} |
| collision | 0.3 |
| timeout | 0.033333 |
| stop_success_mean | 4.1 |
| avg_speed | 2.6135 |
| bs02_lc_fail | 0 |
| zone_max_overspeed | 0.64544 |
| zone_overspeed_step_rate | 0.012452 |
| 평균 이동 속도 15 km/h 미만 | 2/30 |
| 5초 이내 실제 반대 방향 차선 변경 | 2467 |
| 주변 차량이 있었던 스텝 | 43138/46172 (93.4%) |
| 주변 차량이 한 번도 없었던 판 | 0/30 |

| 다음 정류장 | edge | 위치 구간 시작 (m) | 방향 반전 | 대표 시드 | SUMO 시각 (s) |
|---|---|---:|---:|---:|---:|
| bs_03 | e908646367_3 | 30 | 1560 | 900112 | 274.5 |
| bs_03 | e908646367_3 | 40 | 852 | 900101 | 276.0 |
| bs_03 | e908646367_3 | 20 | 29 | 900112 | 272.0 |
| bs_04 | e908646367_6 | 60 | 14 | 900100 | 424.0 |
| bs_04 | e908646367_6 | 50 | 9 | 900104 | 448.0 |
| bs_04 | e908646367_6 | 30 | 3 | 900104 | 444.5 |

### klppo1_s6_s1_step25k (30/30판)

| 지표 | 값 |
|---|---:|
| episodes | 30 |
| official_complete | 18/30 (60.0%) |
| perfect_complete | 18/30 (60.0%) |
| collision_count | 8 |
| collision_rate | 8/30 (26.7%) |
| wrong_lane_count | 0 |
| timeout_count | 1 |
| inf_wait_episodes | 17 |
| wait_timeout_wrong_lane_episodes | 17 |
| green_stop_max_sec | 1339.5 |
| green_stop_time_mean | 0.12532 |
| green_stop_events_mean | 50.467 |
| moving_avg_speed_kmh | 18.365 |
| avg_speed_kmh | 9.72 |
| teleports | 0 |
| sim_errors | 0 |
| red_light | 0.1 |
| yield_steps | 1 |
| intervention | 0.0097739 |
| reasons | {'completed': 18, 'collision': 8, 'red_light': 3, 'timeout': 1} |
| collision | 0.26667 |
| timeout | 0.033333 |
| stop_success_mean | 3.9 |
| avg_speed | 2.7027 |
| bs02_lc_fail | 0 |
| zone_max_overspeed | 0.22115 |
| zone_overspeed_step_rate | 0.005291 |
| 평균 이동 속도 15 km/h 미만 | 1/30 |
| 5초 이내 실제 반대 방향 차선 변경 | 2105 |
| 주변 차량이 있었던 스텝 | 38312/41466 (92.4%) |
| 주변 차량이 한 번도 없었던 판 | 0/30 |

| 다음 정류장 | edge | 위치 구간 시작 (m) | 방향 반전 | 대표 시드 | SUMO 시각 (s) |
|---|---|---:|---:|---:|---:|
| bs_03 | e908646367_3 | 30 | 1577 | 900106 | 354.0 |
| bs_03 | e908646367_3 | 40 | 452 | 900101 | 227.0 |
| bs_03 | e908646367_3 | 20 | 45 | 900106 | 351.5 |
| bs_04 | e908646367_6 | 60 | 24 | 900100 | 423.5 |
| bs_04 | e908646367_6 | 30 | 3 | 900117 | 449.0 |
| bs_04 | e908646367_6 | 40 | 3 | 900117 | 450.5 |
| bs_04 | e908646367_6 | 50 | 1 | 900104 | 482.0 |

### klppo1_s6_s1_step275k (30/30판)

| 지표 | 값 |
|---|---:|
| episodes | 30 |
| official_complete | 16/30 (53.3%) |
| perfect_complete | 16/30 (53.3%) |
| collision_count | 11 |
| collision_rate | 11/30 (36.7%) |
| wrong_lane_count | 0 |
| timeout_count | 2 |
| inf_wait_episodes | 21 |
| wait_timeout_wrong_lane_episodes | 21 |
| green_stop_max_sec | 25.5 |
| green_stop_time_mean | 0.099247 |
| green_stop_events_mean | 50.833 |
| moving_avg_speed_kmh | 17.937 |
| avg_speed_kmh | 9.3478 |
| teleports | 0 |
| sim_errors | 0 |
| red_light | 0.033333 |
| yield_steps | 0 |
| intervention | 0.028201 |
| reasons | {'completed': 16, 'collision': 11, 'timeout': 2, 'red_light': 1} |
| collision | 0.36667 |
| timeout | 0.066667 |
| stop_success_mean | 3.9333 |
| avg_speed | 2.599 |
| bs02_lc_fail | 0 |
| zone_max_overspeed | 0.51034 |
| zone_overspeed_step_rate | 0.0052622 |
| 평균 이동 속도 15 km/h 미만 | 2/30 |
| 5초 이내 실제 반대 방향 차선 변경 | 2023 |
| 주변 차량이 있었던 스텝 | 40905/44281 (92.4%) |
| 주변 차량이 한 번도 없었던 판 | 0/30 |

| 다음 정류장 | edge | 위치 구간 시작 (m) | 방향 반전 | 대표 시드 | SUMO 시각 (s) |
|---|---|---:|---:|---:|---:|
| bs_03 | e908646367_3 | 30 | 1386 | 900101 | 410.5 |
| bs_03 | e908646367_3 | 40 | 568 | 900102 | 325.0 |
| bs_03 | e908646367_3 | 20 | 46 | 900101 | 409.0 |
| bs_04 | e908646367_6 | 60 | 17 | 900100 | 425.0 |
| bs_04 | e908646367_6 | 30 | 3 | 900104 | 437.5 |
| bs_04 | e908646367_6 | 50 | 2 | 900116 | 872.5 |
| bs_04 | e908646367_6 | 40 | 1 | 900104 | 439.0 |

### klppo1_s6_s1_step300k (30/30판)

| 지표 | 값 |
|---|---:|
| episodes | 30 |
| official_complete | 18/30 (60.0%) |
| perfect_complete | 18/30 (60.0%) |
| collision_count | 7 |
| collision_rate | 7/30 (23.3%) |
| wrong_lane_count | 0 |
| timeout_count | 2 |
| inf_wait_episodes | 22 |
| wait_timeout_wrong_lane_episodes | 22 |
| green_stop_max_sec | 1319 |
| green_stop_time_mean | 0.1123 |
| green_stop_events_mean | 48.433 |
| moving_avg_speed_kmh | 18.299 |
| avg_speed_kmh | 9.6563 |
| teleports | 0 |
| sim_errors | 0 |
| red_light | 0.1 |
| yield_steps | 0 |
| intervention | 0.010905 |
| reasons | {'completed': 18, 'collision': 7, 'red_light': 3, 'timeout': 2} |
| collision | 0.23333 |
| timeout | 0.066667 |
| stop_success_mean | 3.9667 |
| avg_speed | 2.6852 |
| bs02_lc_fail | 1 |
| zone_max_overspeed | 0.3015 |
| zone_overspeed_step_rate | 0.012264 |
| 평균 이동 속도 15 km/h 미만 | 1/30 |
| 5초 이내 실제 반대 방향 차선 변경 | 2107 |
| 주변 차량이 있었던 스텝 | 41060/44321 (92.6%) |
| 주변 차량이 한 번도 없었던 판 | 0/30 |

| 다음 정류장 | edge | 위치 구간 시작 (m) | 방향 반전 | 대표 시드 | SUMO 시각 (s) |
|---|---|---:|---:|---:|---:|
| bs_03 | e908646367_3 | 30 | 1340 | 900104 | 374.5 |
| bs_03 | e908646367_3 | 40 | 683 | 900101 | 227.5 |
| bs_03 | e908646367_3 | 20 | 48 | 900104 | 372.0 |
| bs_04 | e908646367_6 | 60 | 28 | 900100 | 425.0 |
| bs_04 | e908646367_6 | 50 | 6 | 900116 | 872.5 |
| bs_04 | e908646367_6 | 30 | 2 | 900117 | 454.5 |

### klppo1_s6_s1_step50k (30/30판)

| 지표 | 값 |
|---|---:|
| episodes | 30 |
| official_complete | 17/30 (56.7%) |
| perfect_complete | 17/30 (56.7%) |
| collision_count | 10 |
| collision_rate | 10/30 (33.3%) |
| wrong_lane_count | 0 |
| timeout_count | 1 |
| inf_wait_episodes | 23 |
| wait_timeout_wrong_lane_episodes | 23 |
| green_stop_max_sec | 1394 |
| green_stop_time_mean | 0.12642 |
| green_stop_events_mean | 53.333 |
| moving_avg_speed_kmh | 17.745 |
| avg_speed_kmh | 9.3392 |
| teleports | 0 |
| sim_errors | 0 |
| red_light | 0.066667 |
| yield_steps | 1 |
| intervention | 0.012436 |
| reasons | {'completed': 17, 'collision': 10, 'red_light': 2, 'timeout': 1} |
| collision | 0.33333 |
| timeout | 0.033333 |
| stop_success_mean | 4 |
| avg_speed | 2.5966 |
| bs02_lc_fail | 0 |
| zone_max_overspeed | 0.14834 |
| zone_overspeed_step_rate | 0.0048921 |
| 평균 이동 속도 15 km/h 미만 | 4/30 |
| 5초 이내 실제 반대 방향 차선 변경 | 2242 |
| 주변 차량이 있었던 스텝 | 40837/43934 (93.0%) |
| 주변 차량이 한 번도 없었던 판 | 0/30 |

| 다음 정류장 | edge | 위치 구간 시작 (m) | 방향 반전 | 대표 시드 | SUMO 시각 (s) |
|---|---|---:|---:|---:|---:|
| bs_03 | e908646367_3 | 30 | 1279 | 900103 | 388.5 |
| bs_03 | e908646367_3 | 40 | 928 | 900101 | 275.5 |
| bs_03 | e908646367_3 | 20 | 15 | 900112 | 269.5 |
| bs_04 | e908646367_6 | 60 | 12 | 900120 | 419.5 |
| bs_04 | e908646367_6 | 50 | 6 | 900128 | 588.5 |
| bs_05 | e218924602_6 | 60 | 2 | 900126 | 967.0 |

### klppo1_s6_s1_step75k (30/30판)

| 지표 | 값 |
|---|---:|
| episodes | 30 |
| official_complete | 18/30 (60.0%) |
| perfect_complete | 18/30 (60.0%) |
| collision_count | 5 |
| collision_rate | 5/30 (16.7%) |
| wrong_lane_count | 0 |
| timeout_count | 5 |
| inf_wait_episodes | 21 |
| wait_timeout_wrong_lane_episodes | 21 |
| green_stop_max_sec | 1370 |
| green_stop_time_mean | 0.18915 |
| green_stop_events_mean | 58.533 |
| moving_avg_speed_kmh | 18.159 |
| avg_speed_kmh | 8.9783 |
| teleports | 0 |
| sim_errors | 0 |
| red_light | 0.066667 |
| yield_steps | 0 |
| intervention | 0.015424 |
| reasons | {'completed': 18, 'collision': 5, 'timeout': 5, 'red_light': 2} |
| collision | 0.16667 |
| timeout | 0.16667 |
| stop_success_mean | 4 |
| avg_speed | 2.4962 |
| bs02_lc_fail | 0 |
| zone_max_overspeed | 0.48329 |
| zone_overspeed_step_rate | 0.010347 |
| 평균 이동 속도 15 km/h 미만 | 3/30 |
| 5초 이내 실제 반대 방향 차선 변경 | 2529 |
| 주변 차량이 있었던 스텝 | 48125/51440 (93.6%) |
| 주변 차량이 한 번도 없었던 판 | 0/30 |

| 다음 정류장 | edge | 위치 구간 시작 (m) | 방향 반전 | 대표 시드 | SUMO 시각 (s) |
|---|---|---:|---:|---:|---:|
| bs_03 | e908646367_3 | 40 | 1312 | 900101 | 230.0 |
| bs_03 | e908646367_3 | 30 | 1171 | 900103 | 366.0 |
| bs_03 | e908646367_3 | 20 | 20 | 900103 | 364.0 |
| bs_04 | e908646367_6 | 60 | 19 | 900100 | 424.5 |
| bs_04 | e908646367_6 | 50 | 7 | 900124 | 324.0 |

### health_klppo3_s6_s1_50k (10/10판)

| 지표 | 값 |
|---|---:|
| episodes | 10 |
| official_complete | 0/10 (0.0%) |
| perfect_complete | 0/10 (0.0%) |
| collision_count | 0 |
| collision_rate | 0/10 (0.0%) |
| wrong_lane_count | 0 |
| timeout_count | 9 |
| inf_wait_episodes | 9 |
| wait_timeout_wrong_lane_episodes | 9 |
| green_stop_max_sec | 22 |
| green_stop_time_mean | 0.027015 |
| green_stop_events_mean | 12.3 |
| moving_avg_speed_kmh | 18.645 |
| avg_speed_kmh | 2.3257 |
| teleports | 0 |
| sim_errors | 0 |
| red_light | 0.1 |
| yield_steps | 0 |
| intervention | 0.033756 |
| reasons | {'timeout': 9, 'red_light': 1} |
| collision | 0 |
| timeout | 0.9 |
| stop_success_mean | 2.1 |
| avg_speed | 0.64653 |
| bs02_lc_fail | 0 |
| zone_max_overspeed | nan |
| zone_overspeed_step_rate | nan |
| 평균 이동 속도 15 km/h 미만 | 0/10 |
| 5초 이내 실제 반대 방향 차선 변경 | 0 |
| 주변 차량이 있었던 스텝 | 32545/33003 (98.6%) |
| 주변 차량이 한 번도 없었던 판 | 0/10 |

| 다음 정류장 | edge | 위치 구간 시작 (m) | 방향 반전 | 대표 시드 | SUMO 시각 (s) |
|---|---|---:|---:|---:|---:|

### health_klppo3_s6_s0_50k (10/10판)

| 지표 | 값 |
|---|---:|
| episodes | 10 |
| official_complete | 0/10 (0.0%) |
| perfect_complete | 0/10 (0.0%) |
| collision_count | 0 |
| collision_rate | 0/10 (0.0%) |
| wrong_lane_count | 0 |
| timeout_count | 9 |
| inf_wait_episodes | 9 |
| wait_timeout_wrong_lane_episodes | 9 |
| green_stop_max_sec | 14 |
| green_stop_time_mean | 0.030304 |
| green_stop_events_mean | 12.8 |
| moving_avg_speed_kmh | 18.629 |
| avg_speed_kmh | 2.258 |
| teleports | 0 |
| sim_errors | 0 |
| red_light | 0.1 |
| yield_steps | 0 |
| intervention | 0.032517 |
| reasons | {'timeout': 9, 'red_light': 1} |
| collision | 0 |
| timeout | 0.9 |
| stop_success_mean | 2.1 |
| avg_speed | 0.62769 |
| bs02_lc_fail | 0 |
| zone_max_overspeed | nan |
| zone_overspeed_step_rate | nan |
| 평균 이동 속도 15 km/h 미만 | 0/10 |
| 5초 이내 실제 반대 방향 차선 변경 | 0 |
| 주변 차량이 있었던 스텝 | 32500/32999 (98.5%) |
| 주변 차량이 한 번도 없었던 판 | 0/10 |

| 다음 정류장 | edge | 위치 구간 시작 (m) | 방향 반전 | 대표 시드 | SUMO 시각 (s) |
|---|---|---:|---:|---:|---:|

### health_klppo3_s6_s0_100k (10/10판)

| 지표 | 값 |
|---|---:|
| episodes | 10 |
| official_complete | 9/10 (90.0%) |
| perfect_complete | 9/10 (90.0%) |
| collision_count | 0 |
| collision_rate | 0/10 (0.0%) |
| wrong_lane_count | 0 |
| timeout_count | 0 |
| inf_wait_episodes | 6 |
| wait_timeout_wrong_lane_episodes | 6 |
| green_stop_max_sec | 113 |
| green_stop_time_mean | 0.11808 |
| green_stop_events_mean | 28.8 |
| moving_avg_speed_kmh | 18.542 |
| avg_speed_kmh | 9.9326 |
| teleports | 0 |
| sim_errors | 0 |
| red_light | 0.1 |
| yield_steps | 0 |
| intervention | 0.08565 |
| reasons | {'completed': 9, 'red_light': 1} |
| collision | 0 |
| timeout | 0 |
| stop_success_mean | 4.7 |
| avg_speed | 2.7612 |
| bs02_lc_fail | 0 |
| zone_max_overspeed | 0.37866 |
| zone_overspeed_step_rate | 0.012422 |
| 평균 이동 속도 15 km/h 미만 | 0/10 |
| 5초 이내 실제 반대 방향 차선 변경 | 158 |
| 주변 차량이 있었던 스텝 | 13761/15013 (91.7%) |
| 주변 차량이 한 번도 없었던 판 | 0/10 |

| 다음 정류장 | edge | 위치 구간 시작 (m) | 방향 반전 | 대표 시드 | SUMO 시각 (s) |
|---|---|---:|---:|---:|---:|
| bs_03 | e908646367_3 | 40 | 55 | 900102 | 328.5 |
| bs_03 | e908646367_3 | 30 | 52 | 900101 | 411.0 |
| bs_05 | e218924602_6 | 60 | 50 | 900105 | 666.5 |
| bs_03 | e908646367_3 | 20 | 1 | 900101 | 410.5 |

### health_klppo3_s6_s1_100k (10/10판)

| 지표 | 값 |
|---|---:|
| episodes | 10 |
| official_complete | 0/10 (0.0%) |
| perfect_complete | 0/10 (0.0%) |
| collision_count | 0 |
| collision_rate | 0/10 (0.0%) |
| wrong_lane_count | 0 |
| timeout_count | 10 |
| inf_wait_episodes | 10 |
| wait_timeout_wrong_lane_episodes | 10 |
| green_stop_max_sec | 23.5 |
| green_stop_time_mean | 0.0155 |
| green_stop_events_mean | 11.7 |
| moving_avg_speed_kmh | 19.023 |
| avg_speed_kmh | 1.425 |
| teleports | 0 |
| sim_errors | 0 |
| red_light | 0 |
| yield_steps | 0 |
| intervention | 0.022389 |
| reasons | {'timeout': 10} |
| collision | 0 |
| timeout | 1 |
| stop_success_mean | 2.2 |
| avg_speed | 0.39583 |
| bs02_lc_fail | 0 |
| zone_max_overspeed | nan |
| zone_overspeed_step_rate | nan |
| 평균 이동 속도 15 km/h 미만 | 0/10 |
| 5초 이내 실제 반대 방향 차선 변경 | 0 |
| 주변 차량이 있었던 스텝 | 35515/36000 (98.7%) |
| 주변 차량이 한 번도 없었던 판 | 0/10 |

| 다음 정류장 | edge | 위치 구간 시작 (m) | 방향 반전 | 대표 시드 | SUMO 시각 (s) |
|---|---|---:|---:|---:|---:|

### klppo3_s6_s0_step50k (30/30판)

| 지표 | 값 |
|---|---:|
| episodes | 30 |
| official_complete | 0/30 (0.0%) |
| perfect_complete | 0/30 (0.0%) |
| collision_count | 0 |
| collision_rate | 0/30 (0.0%) |
| wrong_lane_count | 0 |
| timeout_count | 27 |
| inf_wait_episodes | 27 |
| wait_timeout_wrong_lane_episodes | 27 |
| green_stop_max_sec | 47 |
| green_stop_time_mean | 0.031981 |
| green_stop_events_mean | 11.9 |
| moving_avg_speed_kmh | 19.392 |
| avg_speed_kmh | 2.3338 |
| teleports | 0 |
| sim_errors | 0 |
| red_light | 0.1 |
| yield_steps | 0 |
| intervention | 0.035005 |
| reasons | {'timeout': 27, 'red_light': 3} |
| collision | 0 |
| timeout | 0.9 |
| stop_success_mean | 2.1667 |
| avg_speed | 0.64878 |
| bs02_lc_fail | 0 |
| zone_max_overspeed | nan |
| zone_overspeed_step_rate | nan |
| 평균 이동 속도 15 km/h 미만 | 0/30 |
| 5초 이내 실제 반대 방향 차선 변경 | 0 |
| 주변 차량이 있었던 스텝 | 97480/98990 (98.5%) |
| 주변 차량이 한 번도 없었던 판 | 0/30 |

| 다음 정류장 | edge | 위치 구간 시작 (m) | 방향 반전 | 대표 시드 | SUMO 시각 (s) |
|---|---|---:|---:|---:|---:|

### klppo3_s6_s0_step100k (30/30판)

| 지표 | 값 |
|---|---:|
| episodes | 30 |
| official_complete | 22/30 (73.3%) |
| perfect_complete | 22/30 (73.3%) |
| collision_count | 4 |
| collision_rate | 4/30 (13.3%) |
| wrong_lane_count | 0 |
| timeout_count | 2 |
| inf_wait_episodes | 20 |
| wait_timeout_wrong_lane_episodes | 20 |
| green_stop_max_sec | 116.5 |
| green_stop_time_mean | 0.11158 |
| green_stop_events_mean | 35.133 |
| moving_avg_speed_kmh | 18.641 |
| avg_speed_kmh | 9.3022 |
| teleports | 0 |
| sim_errors | 0 |
| red_light | 0.066667 |
| yield_steps | 0 |
| intervention | 0.075176 |
| reasons | {'completed': 22, 'red_light': 2, 'timeout': 2, 'collision': 4} |
| collision | 0.13333 |
| timeout | 0.066667 |
| stop_success_mean | 4.5333 |
| avg_speed | 2.5864 |
| bs02_lc_fail | 0 |
| zone_max_overspeed | 0.37866 |
| zone_overspeed_step_rate | 0.010112 |
| 평균 이동 속도 15 km/h 미만 | 1/30 |
| 5초 이내 실제 반대 방향 차선 변경 | 930 |
| 주변 차량이 있었던 스텝 | 46867/50496 (92.8%) |
| 주변 차량이 한 번도 없었던 판 | 0/30 |

| 다음 정류장 | edge | 위치 구간 시작 (m) | 방향 반전 | 대표 시드 | SUMO 시각 (s) |
|---|---|---:|---:|---:|---:|
| bs_03 | e908646367_3 | 40 | 508 | 900102 | 328.5 |
| bs_03 | e908646367_3 | 30 | 366 | 900101 | 411.0 |
| bs_05 | e218924602_6 | 60 | 50 | 900105 | 666.5 |
| bs_03 | e908646367_3 | 20 | 6 | 900101 | 410.5 |

### klppo3_s6_s0_step150k (30/30판)

| 지표 | 값 |
|---|---:|
| episodes | 30 |
| official_complete | 0/30 (0.0%) |
| perfect_complete | 0/30 (0.0%) |
| collision_count | 0 |
| collision_rate | 0/30 (0.0%) |
| wrong_lane_count | 0 |
| timeout_count | 28 |
| inf_wait_episodes | 28 |
| wait_timeout_wrong_lane_episodes | 28 |
| green_stop_max_sec | 32.5 |
| green_stop_time_mean | 0.026397 |
| green_stop_events_mean | 11.433 |
| moving_avg_speed_kmh | 19.654 |
| avg_speed_kmh | 2.1054 |
| teleports | 0 |
| sim_errors | 0 |
| red_light | 0.066667 |
| yield_steps | 0 |
| intervention | 0.02126 |
| reasons | {'timeout': 28, 'red_light': 2} |
| collision | 0 |
| timeout | 0.93333 |
| stop_success_mean | 2.2 |
| avg_speed | 0.58519 |
| bs02_lc_fail | 0 |
| zone_max_overspeed | nan |
| zone_overspeed_step_rate | nan |
| 평균 이동 속도 15 km/h 미만 | 1/30 |
| 5초 이내 실제 반대 방향 차선 변경 | 0 |
| 주변 차량이 있었던 스텝 | 100479/101931 (98.6%) |
| 주변 차량이 한 번도 없었던 판 | 0/30 |

| 다음 정류장 | edge | 위치 구간 시작 (m) | 방향 반전 | 대표 시드 | SUMO 시각 (s) |
|---|---|---:|---:|---:|---:|

### klppo3_s6_s0_step200k (30/30판)

| 지표 | 값 |
|---|---:|
| episodes | 30 |
| official_complete | 1/30 (3.3%) |
| perfect_complete | 1/30 (3.3%) |
| collision_count | 0 |
| collision_rate | 0/30 (0.0%) |
| wrong_lane_count | 0 |
| timeout_count | 29 |
| inf_wait_episodes | 30 |
| wait_timeout_wrong_lane_episodes | 30 |
| green_stop_max_sec | 108 |
| green_stop_time_mean | 0.043672 |
| green_stop_events_mean | 14.4 |
| moving_avg_speed_kmh | 19.293 |
| avg_speed_kmh | 2.2947 |
| teleports | 0 |
| sim_errors | 0 |
| red_light | 0 |
| yield_steps | 0 |
| intervention | 0.021099 |
| reasons | {'timeout': 29, 'completed': 1} |
| collision | 0 |
| timeout | 0.96667 |
| stop_success_mean | 2.9 |
| avg_speed | 0.6375 |
| bs02_lc_fail | 0 |
| zone_max_overspeed | 0.38791 |
| zone_overspeed_step_rate | 0.0175 |
| 평균 이동 속도 15 km/h 미만 | 2/30 |
| 5초 이내 실제 반대 방향 차선 변경 | 6 |
| 주변 차량이 있었던 스텝 | 103372/105691 (97.8%) |
| 주변 차량이 한 번도 없었던 판 | 0/30 |

| 다음 정류장 | edge | 위치 구간 시작 (m) | 방향 반전 | 대표 시드 | SUMO 시각 (s) |
|---|---|---:|---:|---:|---:|
| bs_03 | e908646367_3 | 20 | 4 | 900103 | 397.0 |
| bs_03 | e908646367_3 | 30 | 2 | 900114 | 286.0 |

### klppo3_s6_s0_step250k (30/30판)

| 지표 | 값 |
|---|---:|
| episodes | 30 |
| official_complete | 0/30 (0.0%) |
| perfect_complete | 0/30 (0.0%) |
| collision_count | 1 |
| collision_rate | 1/30 (3.3%) |
| wrong_lane_count | 0 |
| timeout_count | 26 |
| inf_wait_episodes | 27 |
| wait_timeout_wrong_lane_episodes | 27 |
| green_stop_max_sec | 26 |
| green_stop_time_mean | 0.029324 |
| green_stop_events_mean | 8.4 |
| moving_avg_speed_kmh | 20.162 |
| avg_speed_kmh | 3.0406 |
| teleports | 0 |
| sim_errors | 0 |
| red_light | 0.1 |
| yield_steps | 0 |
| intervention | 0.030709 |
| reasons | {'timeout': 26, 'collision': 1, 'red_light': 3} |
| collision | 0.033333 |
| timeout | 0.86667 |
| stop_success_mean | 2.3333 |
| avg_speed | 0.84556 |
| bs02_lc_fail | 0 |
| zone_max_overspeed | 0.064777 |
| zone_overspeed_step_rate | 0.01 |
| 평균 이동 속도 15 km/h 미만 | 0/30 |
| 5초 이내 실제 반대 방향 차선 변경 | 0 |
| 주변 차량이 있었던 스텝 | 94680/95741 (98.9%) |
| 주변 차량이 한 번도 없었던 판 | 0/30 |

| 다음 정류장 | edge | 위치 구간 시작 (m) | 방향 반전 | 대표 시드 | SUMO 시각 (s) |
|---|---|---:|---:|---:|---:|

### klppo3_s6_s1_step50k (30/30판)

| 지표 | 값 |
|---|---:|
| episodes | 30 |
| official_complete | 0/30 (0.0%) |
| perfect_complete | 0/30 (0.0%) |
| collision_count | 0 |
| collision_rate | 0/30 (0.0%) |
| wrong_lane_count | 0 |
| timeout_count | 28 |
| inf_wait_episodes | 28 |
| wait_timeout_wrong_lane_episodes | 28 |
| green_stop_max_sec | 44.5 |
| green_stop_time_mean | 0.024695 |
| green_stop_events_mean | 11.9 |
| moving_avg_speed_kmh | 19.462 |
| avg_speed_kmh | 2.0566 |
| teleports | 0 |
| sim_errors | 0 |
| red_light | 0.066667 |
| yield_steps | 0 |
| intervention | 0.03016 |
| reasons | {'timeout': 28, 'red_light': 2} |
| collision | 0 |
| timeout | 0.93333 |
| stop_success_mean | 2.2 |
| avg_speed | 0.57159 |
| bs02_lc_fail | 0 |
| zone_max_overspeed | nan |
| zone_overspeed_step_rate | nan |
| 평균 이동 속도 15 km/h 미만 | 0/30 |
| 5초 이내 실제 반대 방향 차선 변경 | 0 |
| 주변 차량이 있었던 스텝 | 100572/102058 (98.5%) |
| 주변 차량이 한 번도 없었던 판 | 0/30 |

| 다음 정류장 | edge | 위치 구간 시작 (m) | 방향 반전 | 대표 시드 | SUMO 시각 (s) |
|---|---|---:|---:|---:|---:|

### klppo3_s6_s1_step100k (30/30판)

| 지표 | 값 |
|---|---:|
| episodes | 30 |
| official_complete | 0/30 (0.0%) |
| perfect_complete | 0/30 (0.0%) |
| collision_count | 0 |
| collision_rate | 0/30 (0.0%) |
| wrong_lane_count | 0 |
| timeout_count | 28 |
| inf_wait_episodes | 28 |
| wait_timeout_wrong_lane_episodes | 28 |
| green_stop_max_sec | 28 |
| green_stop_time_mean | 0.025764 |
| green_stop_events_mean | 11.8 |
| moving_avg_speed_kmh | 19.775 |
| avg_speed_kmh | 2.064 |
| teleports | 0 |
| sim_errors | 0 |
| red_light | 0.066667 |
| yield_steps | 0 |
| intervention | 0.022506 |
| reasons | {'timeout': 28, 'red_light': 2} |
| collision | 0 |
| timeout | 0.93333 |
| stop_success_mean | 2.2 |
| avg_speed | 0.57367 |
| bs02_lc_fail | 0 |
| zone_max_overspeed | nan |
| zone_overspeed_step_rate | nan |
| 평균 이동 속도 15 km/h 미만 | 0/30 |
| 5초 이내 실제 반대 방향 차선 변경 | 0 |
| 주변 차량이 있었던 스텝 | 100497/101977 (98.5%) |
| 주변 차량이 한 번도 없었던 판 | 0/30 |

| 다음 정류장 | edge | 위치 구간 시작 (m) | 방향 반전 | 대표 시드 | SUMO 시각 (s) |
|---|---|---:|---:|---:|---:|

### klppo3_s6_s1_step150k (30/30판)

| 지표 | 값 |
|---|---:|
| episodes | 30 |
| official_complete | 0/30 (0.0%) |
| perfect_complete | 0/30 (0.0%) |
| collision_count | 0 |
| collision_rate | 0/30 (0.0%) |
| wrong_lane_count | 0 |
| timeout_count | 30 |
| inf_wait_episodes | 30 |
| wait_timeout_wrong_lane_episodes | 30 |
| green_stop_max_sec | 221 |
| green_stop_time_mean | 0.032083 |
| green_stop_events_mean | 12.233 |
| moving_avg_speed_kmh | 19.877 |
| avg_speed_kmh | 1.5255 |
| teleports | 0 |
| sim_errors | 0 |
| red_light | 0 |
| yield_steps | 0 |
| intervention | 0.02188 |
| reasons | {'timeout': 30} |
| collision | 0 |
| timeout | 1 |
| stop_success_mean | 2.2667 |
| avg_speed | 0.42376 |
| bs02_lc_fail | 0 |
| zone_max_overspeed | nan |
| zone_overspeed_step_rate | nan |
| 평균 이동 속도 15 km/h 미만 | 0/30 |
| 5초 이내 실제 반대 방향 차선 변경 | 0 |
| 주변 차량이 있었던 스텝 | 105816/108000 (98.0%) |
| 주변 차량이 한 번도 없었던 판 | 0/30 |

| 다음 정류장 | edge | 위치 구간 시작 (m) | 방향 반전 | 대표 시드 | SUMO 시각 (s) |
|---|---|---:|---:|---:|---:|

### klppo3_s6_s1_step200k (30/30판)

| 지표 | 값 |
|---|---:|
| episodes | 30 |
| official_complete | 0/30 (0.0%) |
| perfect_complete | 0/30 (0.0%) |
| collision_count | 0 |
| collision_rate | 0/30 (0.0%) |
| wrong_lane_count | 0 |
| timeout_count | 29 |
| inf_wait_episodes | 30 |
| wait_timeout_wrong_lane_episodes | 30 |
| green_stop_max_sec | 29.5 |
| green_stop_time_mean | 0.021162 |
| green_stop_events_mean | 11.567 |
| moving_avg_speed_kmh | 20.292 |
| avg_speed_kmh | 1.7619 |
| teleports | 0 |
| sim_errors | 0 |
| red_light | 0.033333 |
| yield_steps | 0 |
| intervention | 0.020573 |
| reasons | {'timeout': 29, 'red_light': 1} |
| collision | 0 |
| timeout | 0.96667 |
| stop_success_mean | 2.2333 |
| avg_speed | 0.48955 |
| bs02_lc_fail | 0 |
| zone_max_overspeed | nan |
| zone_overspeed_step_rate | nan |
| 평균 이동 속도 15 km/h 미만 | 0/30 |
| 5초 이내 실제 반대 방향 차선 변경 | 0 |
| 주변 차량이 있었던 스텝 | 103489/105043 (98.5%) |
| 주변 차량이 한 번도 없었던 판 | 0/30 |

| 다음 정류장 | edge | 위치 구간 시작 (m) | 방향 반전 | 대표 시드 | SUMO 시각 (s) |
|---|---|---:|---:|---:|---:|

### klppo3_s6_s1_step250k (30/30판)

| 지표 | 값 |
|---|---:|
| episodes | 30 |
| official_complete | 0/30 (0.0%) |
| perfect_complete | 0/30 (0.0%) |
| collision_count | 0 |
| collision_rate | 0/30 (0.0%) |
| wrong_lane_count | 0 |
| timeout_count | 28 |
| inf_wait_episodes | 29 |
| wait_timeout_wrong_lane_episodes | 29 |
| green_stop_max_sec | 36 |
| green_stop_time_mean | 0.027687 |
| green_stop_events_mean | 12.533 |
| moving_avg_speed_kmh | 20.341 |
| avg_speed_kmh | 2.0601 |
| teleports | 0 |
| sim_errors | 0 |
| red_light | 0.066667 |
| yield_steps | 0 |
| intervention | 0.027712 |
| reasons | {'timeout': 28, 'red_light': 2} |
| collision | 0 |
| timeout | 0.93333 |
| stop_success_mean | 2.2 |
| avg_speed | 0.57256 |
| bs02_lc_fail | 0 |
| zone_max_overspeed | nan |
| zone_overspeed_step_rate | nan |
| 평균 이동 속도 15 km/h 미만 | 0/30 |
| 5초 이내 실제 반대 방향 차선 변경 | 0 |
| 주변 차량이 있었던 스텝 | 100377/102051 (98.4%) |
| 주변 차량이 한 번도 없었던 판 | 0/30 |

| 다음 정류장 | edge | 위치 구간 시작 (m) | 방향 반전 | 대표 시드 | SUMO 시각 (s) |
|---|---|---:|---:|---:|---:|

- 학생 수동 검토용 GUI 시드 900103, 372–379초. `C:\Users\USER\Desktop\청계로드-kl-ppo\results\overnight_s6_final\gui\900103`

## 1단계 장면 1 직접 확인

- 학생 900103, SUMO 376.0초와 376.5초. 정류장 입구에서 초록색 버스가 승강장 쪽 차로에서 왼쪽 차로로 바뀌는 모습을 확인했다. 안전지대 표시가 후방에 있고, 정류장 전방에는 흰 차량과 오토바이 등 주변 교통이 보인다.
- 대표 이미지 [376.0초](img/900103_cc83522_376.0s.png), [376.5초](img/900103_4d54e2f_376.5s.png).
- 같은 edge에서 원시 차로 1과 2가 반복된다. `lane_diff`는 각각 0과 -0.5이며, `bs_03` 정의(원시 차로 1, 30–45 m)는 유지됐다. 차선 변경 금지 플래그도 꺼져 있었다. 30.03 m에서 0.5초 정차 진행률이 생겼다가 옆 차로로 나간 뒤 0으로 돌아갔다.
- 이 학생 사례는 부호 뒤집힘, edge 간 차로 번호 재매핑, 다음 정류장 전환이 발생하지 않았다. 선생님 원인과 lane hold 효과는 별도 시험에 의존한다. 관측/정류장 정의 수정은 아직 적용하지 않았다.

```powershell
$env:PYTHONUTF8='1'; $env:CGC_STAGE='6'; $env:OBS_SCAN='1'; $env:STOP_FIX='0'
py -3.13 scripts/gui_drive_seed.py --seed 900103 --model student --delay 80
```

- 학습 코드 기존 단위 검사 25개 통과. 제어된 차선 로짓에서 변경 확률 감소 벌점 0, 증가 벌점 양수, 새 체크포인트 복원 시 행동과 벌점 동일을 별도로 확인했다.
- 첫 판의 초기 집계 오류는 `None`이 섞인 위치 배열 형식 때문이었다. 저장판을 집계 단계에서 복구했다. 이후 C·D·F 정의를 고친 상태로 새 스모크 및 개발 평가를 시작했다.

- 학생 수동 검토용 GUI 시드 900100, 422–429초. `C:\Users\USER\Desktop\청계로드-kl-ppo\results\overnight_s6_final\gui\900100`

## 1단계 장면 2 직접 확인

- 학생 900100, SUMO 424.0초와 424.5초. 초록색 버스가 파란 노면 표시 옆에서 차로를 옮겼다. 전방 승용차·오토바이와 교차로 대기열이 보였다. 이 장면은 bs_03 승강장 바로 옆의 사례와 위치가 다르며, 해당 판은 정류장 5곳을 모두 정차했다.
- 대표 이미지 [424.0초](img/900100_6cae10c_424.0s.png), [424.5초](img/900100_754d23c_424.5s.png).
- 선생님 코드에서는 정류장 규칙이 차로 유지 명령을 냈을 때, 같은 차로 80 m 안에 주정차 차량이 있으면 후속 회피 규칙이 그 명령을 덮을 수 있다. 이 코드 경로가 bs_03에서 실제로 작동했는지는 선생님 트레이스의 명령을 비교한다.

```powershell
$env:PYTHONUTF8='1'; $env:CGC_STAGE='6'; $env:OBS_SCAN='1'; $env:STOP_FIX='0'
py -3.13 scripts/gui_drive_seed.py --seed 900100 --model student --delay 80
```

## 스모크 결과와 변경 목록

- 6단계 학생 시드 900100은 충돌 0/1, 공식·완벽 완주 1/1, 최대 연속 정지 80.5초, 이동 중 평균 19.1 km/h였다. 반경 50 m 내 차량을 1,270스텝 중 1,157스텝에서 확인했다. 정지 80.5초는 경로상 ipjeong 주정차 구간에 있었다.
- 선생님 동일 시드도 충돌 0/1, 공식·완벽 완주 1/1이었다. 60초 이상 정지는 두 번, 최장 73.5초, 이동 중 평균 17.4 km/h였다. 1판 스모크로 일반 성능을 판단하지 않는다.
- `scripts/overnight_s6_final.py`, `scripts/drive_parallel.py`, `scripts/gui_drive_seed.py`, `scripts/pingpong_diagnose_s6.py`에 실행·판별 저장·재생·선생님 진단을 반영했다. `utils/official_metrics.py`에서 정지 시간, 속도 및 합집합 판 수 계산을 보정했다.
- `train.py`, `algorithms/kl_ppo.py`, `utils/networks.py`, `env/sumo_env.py`에 선택형 학습 항과 기록을 추가했다. 기존 보상 40키의 기준 설정 일치를 확인했다. 새 보상의 4096스텝 스모크와 정규 학습 결과는 대기 중이다.
- 작업 전부터 브랜치에 스테이징된 팀 변경과 미커밋 학습 코드가 있었다. 기존 변경을 섞지 않도록 이번 실험은 작업 트리에 두었다.

- 선생님 시드 900103 통제 스모크에서 bs_03 정류장 차로 진입 뒤 유지 명령을 19스텝 냈다. 10초 정차에 성공했고 5개 정류장을 모두 들러 도착했다. 60초 이상 연속 정지는 없었고, 이동 중 평균은 14.0 km/h였다. 같은 시드의 통제 없는 원본과 비교 중이다.

- 선생님 시드 900103의 bs_03 최대 정차는 원본 10.0초, 차로 유지 대조군 10.0초. 유지 명령 적용 19스텝.

## 출발 모델

- stage6_kl200k, 충돌 4/30, 공식 완주 66.7%.

## 실행 중단

RuntimeError: See C:\Users\USER\Desktop\청계로드-kl-ppo\results\overnight_s6_final\logs\training_smoke.log

- 선생님 시드 900103의 bs_03 최대 정차는 원본 10.0초, 차로 유지 대조군 10.0초. 유지 명령 적용 19스텝.

## 출발 모델

- stage6_kl200k, 충돌 4/30, 공식 완주 66.7%.

## 실행 중단

RuntimeError: See C:\Users\USER\Desktop\청계로드-kl-ppo\results\overnight_s6_final\logs\training_smoke.log

- 선생님 시드 900103의 bs_03 최대 정차는 원본 10.0초, 차로 유지 대조군 10.0초. 유지 명령 적용 19스텝.

## 출발 모델

- stage6_kl200k, 충돌 4/30, 공식 완주 66.7%.

- 2048스텝 × 2 스모크에서 두 감점과 비대칭 벌점에 유한한 양수 값이 기록됐다.

- health_klppo3_s6_s1_50k 충돌 0/10 (출발 0/10), 핑퐁 0 (출발 857). 통과.

- health_klppo3_s6_s0_50k 충돌 0/10 (출발 0/10), 핑퐁 0 (출발 857). 통과.

- health_klppo3_s6_s0_100k 충돌 0/10 (출발 0/10), 핑퐁 158 (출발 857). 통과.

- health_klppo3_s6_s1_100k 충돌 0/10 (출발 0/10), 핑퐁 0 (출발 857). 통과.

기존 SUMO 평가가 끝난 뒤 4,096스텝 스모크, 24,576스텝 KL-PPO를 차례로 실행한다. 개발판의 완주율은 탈락 조건에서 제외한다.

## 최종 판정

개발 30판에서 독립 평가 진입 기준을 충족한 모델이 없다.



## 오케스트레이터 계획

- 정리와 1판 스모크 약 5분.
- 6단계 학생·KL 60판, 5단계 10판, 선생님 및 통제 실험과 GUI 확인 최대 2시간.
- 확인된 환경 원인이 있을 때 STOP_FIX 수정 및 재검증. 원인 진단과 합산 최대 4시간.
- 기존 체크포인트별 개발 30판 평가. 소요 시간은 첫 평가 처리량으로 산정.
- 두 시드 병렬 학습. 1차 실측 10.68스텝/초 기준 25만 스텝은 시드당 약 6시간 30분이며, 동시 실행 시 달라진다. 학습 종료 또는 건강검진 중단 후 평가로 전환한다.
- 독립 100판 평가, GUI 확인 및 보고 최대 2시간.
