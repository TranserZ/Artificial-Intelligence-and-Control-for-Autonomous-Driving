# rule_based_expert (모델 파일 없음, 로직만)

- **구현**: `scripts/rule_based_expert.py`의 `RuleBasedExpert` 클래스.
- **학습 여부**: 학습되지 않음 — 가중치 파일이 없다. traci로 환경 내부
  상태(선행차 간격, 신호, 횡단보도 보행자, 어린이보호구역 제한속도)를
  직접 조회해 결정하는 if-else 규칙 기반 컨트롤러.
- **규칙**: 신호 적/황색 접근 시 정지, 보행자 대기 필요 시 정지,
  어린이보호구역 제한속도(8.33 m/s) 이상이면 감속, 그 외엔 앞차 간격
  (>20m 가속 / <10m 감속 / 사이 유지). 차선변경은 항상 0(금지, 단순화).
- **용도**:
  1) HRL 하위 정책(crosswalk/schoolzone)의 베이스 — 규칙이 이미 그 상황의
     정답에 가깝기 때문에 별도 학습 없이 그대로 쓴다.
  2) BC 전문가 시범 데이터 생성 (`data_huiyoung/bc_expert_data/`).
  3) `evaluate_policy`/`evaluate_all.py`와 같은 `predict(obs, deterministic)`
     인터페이스를 맞춰 다른 정책과 동일 조건으로 비교 평가 가능.
