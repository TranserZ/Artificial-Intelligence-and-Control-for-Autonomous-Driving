# 6단계 야간 실험

## 한눈에 보기

5단계 독립 평가(OBS_SCAN=0) · 충돌 3/100, 장기대기 또는 시간초과 63/100. 보완 D 및 이유 없는 대기 판 수 미측정. [96개 구간 위치·길이](2026-11-26_s5_장기대기.md)

1. 학생과 선생님의 정차·회피 행동 충돌을 검증했다. [8판 대조와 화면](2026-11-26_s6_samil_대조.md)
2. 8k D 10/10, 충돌 0/10, 탈락 / 16k D 8/10, 충돌 0/10, 탈락 / 24k D 8/10, 충돌 0/10, 탈락
3. 429.5초 중 왼쪽 변경은 1회였고 게이트가 허용했다. 항상 차단한다는 가설은 기각하며, 과거 학습에 대한 인과관계는 미확정이다.
4. 5단계 독립 평가 및 GUI 재현 완료
5. 원래 배치 s6_label035 개발 30판 충돌 2, 장기대기 19, 시간초과 0, 공식·완벽 완주 22, 이동 18.9 km/h. 후보 기준 미달로 6단계 독립 시험 미실시.
6. [무효 · OBS_SCAN 불일치] 충돌 63/100, 장기대기·시간초과 38/100, 공식·완벽 완주 29%·29%, 이동 18.6 km/h.
7. 5단계 대비책. 6단계 개발 기준 통과 모델 없음. 5단계 적합성은 위 실측값 참조.
8. 아래 GUI 재현 목록 참조.

## 게이트 가설

| 저장된 항목 | 900100 · 255.5–684.5초 · 859스텝 |
|---|---|
| 차선 명령 | 유지 858회, 왼쪽 1회. 왼쪽 1회는 게이트 허용 및 실제 변경 |
| 평균 신경망 확률 | 왼쪽 5.12%, 유지 94.88%. 유지 명령에서 가상의 왼쪽 게이트는 호출되지 않음 |
| 왼쪽 앞·뒤 거리, 상세 차단 사유, 회피 감점 조건 | 당시 트레이스에 미저장. 과거 klppo3·recovery의 구간별 학습 시도·감점 횟수도 복원 불가. 단계별 평균으로 대체하지 않았다 |

[859스텝 표](../../results/samil_counterfactual/gate_900100.csv) · [장기대기 위치·선생님 동일 10m 구간 통과 시간](../../results/samil_counterfactual/teacher_passage.csv) · [보정 위치 표](2026-11-26_s6_위치재집계.md)

## 실험 기록

| 가설 | 바꾼 것 | 결과 | 판정 |
|---|---|---|---|
| 정차와 회피 행동의 충돌 | samil_east만 제거 | 학생 48→0회, 정차 실패→성공. 8k 구간 최장 429.5→0초 | 해당 시드 지지 |
| 주정차 뒤 안전한 왼쪽 차로와 승강장 유지 라벨을 배우면 대기가 감소한다 | 학습 전용 차선 지도 손실 ×0.1, PLACE_FIX=0 | 10판 D 7 대 6, 충돌 1 대 0 | 탈락 |
| 주정차 뒤 안전한 왼쪽 차로와 승강장 유지 라벨을 배우면 대기가 감소한다 | 학습 전용 차선 지도 손실 ×0.35, PLACE_FIX=0 | 10판 D 3 대 6, 충돌 0 대 0 | 30판 진입 |
| 주정차 뒤 안전한 왼쪽 차로와 승강장 유지 라벨을 배우면 대기가 감소한다 | 학습 전용 차선 지도 손실 ×1.0, PLACE_FIX=0 | 10판 D 8 대 6, 충돌 2 대 0 | 탈락 |
| 승강장과 회피 구간 분리 | bs_03 상류 8m | 출발 모델 장기대기·시간초과 24/30, 선생님 29/30. 03:25 보고서 교체 오류로 학생과 새 정책 비교 중단 | 배치 개선 미입증 |

## 단계별 공식 평가

| 모델 | 판 | A 충돌 | B 공식 | B 완벽 | C 비율·횟수/판·최장 | D 장기대기∪시간초과·시간초과·최장 | E 경로 | F 총시간·이동 km/h | G 순간이동·오류 | H 적색·양보·shadow·변경/판 |
|---|---:|---:|---|---|---|---|---:|---|---|---|
| klppo3_s6_s0_step100k | 30 | 4 | 22/30 (73.3%) | 22/30 (73.3%) | 11.2% · 35.1 · 116.5s | 20 · 2 · 1606.5s | 0 | 9.3 · 18.6 | 0 · 0 | 6.7% · 0 · 7.5% · 36.9 |
| klppo3_s6_recovery_step8k | 10 | 0 | 4/10 (40.0%) | 4/10 (40.0%) | 12.3% · 20.0 · 136.0s | 10 · 5 · 1509.0s | 0 | 5.5 · 18.8 | 0 · 0 | 10.0% · 0 · 4.5% · 5.6 |
| klppo3_s6_recovery_step16k | 10 | 0 | 5/10 (50.0%) | 5/10 (50.0%) | 12.4% · 24.4 · 122.0s | 8 · 2 · 1642.5s | 0 | 7.7 · 18.0 | 0 · 0 | 30.0% · 0 · 7.5% · 14.5 |
| klppo3_s6_recovery_step24k | 10 | 0 | 5/10 (50.0%) | 5/10 (50.0%) | 27.2% · 22.1 · 1460.0s | 8 · 2 · 1493.0s | 0 | 7.6 · 18.8 | 0 · 0 | 30.0% · 0 · 7.1% · 4.2 |
| s6_label01 | 10 | 1 | 5/10 (50.0%) | 5/10 (50.0%) | 16.9% · 21.5 · 137.0s | 7 · 1 · 789.5s | 0 | 9.1 · 18.6 | 0 · 0 | 30.0% · 0 · 6.1% · 4.3 |
| s6_label035 | 30 | 2 | 22/30 (73.3%) | 22/30 (73.3%) | 11.5% · 21.5 · 119.5s | 19 · 0 · 169.0s | 0 | 10.1 · 18.9 | 0 · 0 | 20.0% · 0 · 6.3% · 5.5 |
| s6_label10 | 10 | 2 | 7/10 (70.0%) | 7/10 (70.0%) | 23.6% · 25.0 · 122.5s | 8 · 0 · 614.5s | 0 | 8.8 · 18.0 | 0 · 0 | 10.0% · 0 · 5.6% · 6.1 |
| fix_anchor | 30 | 2 | 14/30 (46.7%) | 14/30 (46.7%) | 29.2% · 42.0 · 1425.0s | 24 · 12 · 1606.5s | 0 | 6.8 · 19.0 | 0 · 0 | 6.7% · 0 · 4.1% · 22.3 |
| teacher56_fix | 30 | 0 | 23/30 (76.7%) | 23/30 (76.7%) | 7.4% · 64.2 · 130.5s | 29 · 7 · 1297.0s | 0 | 7.7 · 18.1 | 0 · 0 | 0.0% · 0 · 65.3% · 108.7 |
| 무효 · final_s5_step250k | 100 | 63 | 29/100 (29.0%) | 29/100 (29.0%) | 24.1% · 7.9 · 1627.0s | 38 · 7 · 1624.0s | 0 | 9.8 · 18.6 | 0 · 0 | 1.0% · 2 · 0.4% · 0.9 |

## 화면과 재현

- A·B의 각 판 정차·회피 화면과 이륜차 좌표는 [대조 보고서](2026-11-26_s6_samil_대조.md)에 있다.
- klppo3_s6_recovery_step16k, 시드 900100, SUMO 907.0초, 종료 timeout. [대표 프레임](img/900100_site_b7724dd_906.5s.png). `$env:PLACE_FIX='0'; py -3.13 scripts/gui_drive_seed.py --stage 6 --seed 900100 --checkpoint 'C:\Users\USER\Desktop\청계로드-kl-ppo\results\klppo3_s6_recovery\step16k.pt' --delay 80`
- klppo3_s6_recovery_step24k, 시드 900100, SUMO 264.0초, 종료 timeout. [대표 프레임](img/900100_site_774a3f9_263.5s.png). `$env:PLACE_FIX='0'; py -3.13 scripts/gui_drive_seed.py --stage 6 --seed 900100 --checkpoint 'C:\Users\USER\Desktop\청계로드-kl-ppo\results\klppo3_s6_recovery\step24k.pt' --delay 80`
- s6_label01, 시드 900100, SUMO 300.5초, 종료 red_light. [대표 프레임](img/900100_site_065b034_300.0s.png). `$env:PLACE_FIX='0'; py -3.13 scripts/gui_drive_seed.py --stage 6 --seed 900100 --checkpoint 'C:\Users\USER\Desktop\청계로드-kl-ppo\results\s6_label01\step25k.pt' --delay 80`
- s6_label035, 시드 900100, SUMO 294.5초, 종료 completed. [대표 프레임](img/900100_site_9ff922e_294.0s.png). `$env:PLACE_FIX='0'; py -3.13 scripts/gui_drive_seed.py --stage 6 --seed 900100 --checkpoint 'C:\Users\USER\Desktop\청계로드-kl-ppo\results\s6_label035\step25k.pt' --delay 80`
- s6_label10, 시드 900100, SUMO 280.5초, 종료 completed. [대표 프레임](img/900100_site_9a3016f_280.0s.png). `$env:PLACE_FIX='0'; py -3.13 scripts/gui_drive_seed.py --stage 6 --seed 900100 --checkpoint 'C:\Users\USER\Desktop\청계로드-kl-ppo\results\s6_label10\step25k.pt' --delay 80`
- fix_anchor, 시드 900100, SUMO 907.0초, 종료 timeout. [대표 프레임](img/900100_site_122f4dc_906.5s.png). `$env:PLACE_FIX='1'; py -3.13 scripts/gui_drive_seed.py --stage 6 --seed 900100 --checkpoint 'C:\Users\USER\Desktop\청계로드-kl-ppo\results\klppo3_s6_s0\step100k.pt' --delay 80`
- final_s5_step250k, 시드 900377, SUMO 213.0초, 종료 completed. [대표 프레임](img/900377_site_dee394c_212.5s.png). `$env:PLACE_FIX='0'; py -3.13 scripts/gui_drive_seed.py --stage 5 --seed 900377 --checkpoint 'C:\Users\USER\Desktop\청계로드-kl-ppo\results\klppo1_s5_s0\step250k.pt' --delay 80`
- final_s5_step250k, 시드 900300, SUMO 170.0초, 종료 collision. [대표 프레임](img/900300_end_7aece94_168.5s.png). `$env:PLACE_FIX='0'; py -3.13 scripts/gui_drive_seed.py --stage 5 --seed 900300 --checkpoint 'C:\Users\USER\Desktop\청계로드-kl-ppo\results\klppo1_s5_s0\step250k.pt' --delay 80`
- final_s5_step250k, 시드 900301, SUMO 548.5초, 종료 completed. [대표 프레임](img/900301_end_750d18e_547.0s.png). `$env:PLACE_FIX='0'; py -3.13 scripts/gui_drive_seed.py --stage 5 --seed 900301 --checkpoint 'C:\Users\USER\Desktop\청계로드-kl-ppo\results\klppo1_s5_s0\step250k.pt' --delay 80`
- 발표 영상 확인 후보 900377, 주행 686초. 단계 5, PLACE_FIX=0. `py -3.13 scripts/gui_drive_seed.py --stage 5 --seed 900377 --checkpoint 'C:\Users\USER\Desktop\청계로드-kl-ppo\results\klppo1_s5_s0\step250k.pt' --delay 40`. 5단계에는 6단계 불법주정차 요소가 없다.

## 팀 공유용 배치 조정

| 옵션 | 항목 | 원위치 | 조정 위치 | 근거 |
|---|---|---|---|---|
| PLACE_FIX=1, 6단계 | bs_03 승강장 | e908646367_3, 차로 1, 30–45 m | 같은 edge·차로, 22–37 m | 8 m 상류 이동으로 주정차 구간 시작까지 간격을 7 m에서 15 m로 확대. 이륜차 위치·대수·종류는 보존 |

원래 배치 및 PLACE_FIX에서 독립 시험에는 결정적 신경망과 shadow 기록을 사용한다. 독립 100판은 개발 순위를 정할 때 읽지 않는다.

## 변경 코드

좌표 집계·대조 실험은 scripts/rebuild_event_locations.py, scripts/samil_counterfactual.py, scripts/report_samil_counterfactual.py, scripts/analyze_gate_archive.py, utils/official_metrics.py에 있다. 학습과 배치는 algorithms/kl_ppo.py, env/sumo_env.py, train.py, tools/make_cgc_scenario.py에서 옵션으로 제어한다. scripts/morning_ladder.py가 단계·마감·보고서를 관리한다. 기존 스테이징 변경이 있는 작업 트리에서 실행했고 새 커밋은 없다.

[이전 전체 개발 평가·실험 경과](2026-11-26_s6_overnight_history.md)
