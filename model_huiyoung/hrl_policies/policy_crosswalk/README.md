# policy_crosswalk

- **내용**: `model_huiyoung/baselines/rule_based_expert` 그대로
  (`scripts/rule_based_expert.py`의 `RuleBasedExpert`).
- **용도**: HRL 상위 컨트롤러가 "경로상 다음 횡단보도가 가까워질 때" 호출하는
  하위 정책. 보행자가 있거나 다가오는 중이면 정지, 아니면 통과.
- **학습 방법**: 없음(규칙). 기존 평가(민기 PPO)에서 양보위반/보행자충돌이
  관찰되는 구간이므로, 우선은 "확실히 멈추는" 규칙으로 안전을 담보하고,
  이후 BC(규칙 시범 데이터)로 더 매끄러운 감속 곡선을 학습시키는 것이
  다음 단계다.
