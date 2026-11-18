# env-v2 — 공통 학습 환경 확정

> `env-v1`의 규칙 없는 운전 환경을 유지하면서, 독립 최종 평가 시드와 공통 학습 산출물 형식을 고정한다.

## env-v1에서 추가된 것

- 학습 판 제외 범위를 `1000~2999`, `900000~999999`로 확대
  - 팀 비교: 1000~1099
  - 최종 후보 평가: 2000~2999
  - 검증·튜닝: 900000~999999
- 학습 시드마다 `step300k.pt`, `step500k.pt`, `final.pt` 저장
  - `final.pt`는 학습이 정상 종료된 마지막 step 모델
  - 중단·오류 종료에서는 `final.pt`를 만들지 않고 `model.pt`만 보존
- 각 모델 옆에 `*.resume.pt`를 저장해 정책·optimizer·누적 step·난수 상태를 함께 복원
  - `--resume final.resume.pt --total-steps 1000000`처럼 최종 누적 목표를 지정
- `--total-steps`, `--out-dir` 학습 옵션 추가
- TensorBoard와 학습 CSV에 보상·손실·엔트로피·PPO 진단값 및 최근 100판 결과 기록
- 학습 종료 시 `curves/`에 보상·손실·탐색·최근 학습 결과 PNG 자동 저장
- `config.json`에 커밋, 환경 태그, 학습 설정, 보상, 관측, shield, 제외 시드, ego SUMO 안전 설정 기록

## 유지되는 환경 기준

- 관측 67차원
- shield 학습 `off`, 검증 `shadow`
- ego `speedMode=0`, `laneChangeMode=0`
- teleport `-1`, collision action `remove`, junction collision 검사 켬
- 정차 성공 +5, 정류장 놓침 −5, 차선변경 금지 구간 위반 −1
- 최대 2400 step, step 길이 0.5초

## 산출물

```text
{out-dir}/
├── step300k.pt
├── step300k.resume.pt
├── step500k.pt
├── step500k.resume.pt
├── final.pt
├── final.resume.pt
├── model.pt
├── best_model.pt
├── config.json
├── training_log.csv
├── eval_log.csv
├── tb/
└── curves/
```

`best_model.pt`는 검증용 참고 모델이고, 공통 final은 마지막 step의 `final.pt`다.

## 검증

- Python 문법·CLI·제외 시드 경계 검사 통과
- 2,048 step 빠른 스모크에서 67차원 학습, 체크포인트, `final.pt`, TensorBoard, CSV, config, PNG 생성 확인
- 20,000 step 학습 스모크 통과
  - 요청 20,000 step, PPO 업데이트 경계에서 실제 20,480 step
  - `step10k.pt`, `step20k.pt`, `final.pt`, TensorBoard, CSV, config, PNG 생성
  - 차선변경 확률 0.56~0.65, 최근 판 실행 차선변경 평균 약 0.95회로 0에 붕괴하지 않음
  - 2만 step은 학습 성능 판정이 아니라 파이프라인 확인용이며, 완주율은 0%였다

공통 평가 스크립트와 평가 그래프는 별도 `tools/eval` 단계에서 구현한 뒤 `env-v2` 태그를 붙인다.
