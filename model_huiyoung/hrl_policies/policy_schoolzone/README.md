# policy_schoolzone

- **내용**: `model_huiyoung/baselines/rule_based_expert` 그대로
  (`scripts/rule_based_expert.py`의 `RuleBasedExpert`).
- **용도**: HRL 상위 컨트롤러가 세운상가 어린이보호구역(e218924602_1~3,
  3차로, 제한 8.33 m/s)에 진입했을 때 호출하는 하위 정책. 제한속도 이상이면
  감속, 그 안에서는 완만히 가속. 가운데 차로 유지는 `env/sumo_env.py`의
  `SCHOOL_ZONE_CENTER_LANE_EDGES` 로직이 정책과 별개로 TraCI 레벨에서
  보조한다.
- **학습 방법**: 없음(규칙). 속도 제한 준수가 규칙만으로 충분히 단순해서
  별도 학습 없이 바로 쓸 수 있다고 판단.
