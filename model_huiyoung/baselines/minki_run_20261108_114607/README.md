# minki_run_20261108_114607 (베이스라인 PPO)

- **원본**: `results/run_20261108_114607/best_model.pt` (학습자: 민기, 왕복 route 기준 학습)
- **알고리즘**: PPO (단일 Gaussian 정책, 가감속+차선변경 2차원 연속 행동)
- **학습 당시 환경**: `road_config_snapshot.py` / `mdp_config_snapshot.py` 참고.
  왕복(r0_loop) route, 정류장 11곳, ego vType=passenger(5m) 기준으로 학습됨.
- **이 브랜치(feature/huiyoung-oneway)에서의 위치**: 편도(동행) + 환경 리얼화
  (차로 수 보정, 어린이보호구역 3차로, 불법주정차 밀집, ego→bus 차체) 이후
  환경에서 이 체크포인트를 그대로 평가해 "환경이 바뀌면 기존 정책이 얼마나
  무너지는가"를 측정하는 기준선(baseline)으로 쓴다.
- **주의**: ego 차체(길이·가감속)가 학습 시점과 달라졌으므로(passenger 5m,
  accel/decel 5.4/5.4 → bus 12m, 1.2/4.0) 성능 저하가 나타나는 것이 정상이며,
  이것이 바로 HRL 하위 정책이 필요한 근거 데이터가 된다.
- **평가**: `scripts/evaluate_all.py`의 `minki_ppo` 항목, 또는
  `py -3.13 test.py model_huiyoung/baselines/minki_run_20261108_114607/best_model.pt --nogui`
