---
language: ko
library_name: pytorch
tags:
  - imitation-learning
  - behavior-cloning
  - autonomous-driving
  - sumo
---

# 청계로드 5단계 학생 모델 (`s5_student.pt`)

청계천 자율주행 버스(청계A01) 노선 2.1 km를 SUMO로 재현한 환경에서, 규칙 없이 혼자 운전하는 모방학습 모델입니다.
팀 최종 모델이며 5단계 도로(정류장, 신호, 끼어들기, 어린이보호구역, 오토바이)에서 학습하고 평가했습니다.

- 코드: [TranserZ/Artificial-Intelligence-and-Control-for-Autonomous-Driving](https://github.com/TranserZ/Artificial-Intelligence-and-Control-for-Autonomous-Driving), 브랜치 `model/dongyeon-bc56` (환경 코드는 `BehaviorCloning` 브랜치와 같음)
- 저장소 안 위치: `models/dongyeon_bc/s5_student.pt`
- 파일 크기 341,549 bytes, sha256 `973ba8889406e53392af5741d3b2c53f80151499c93d9daae816a5218b2c78d5`

## 실행 조건 (하나라도 다르면 성능이 달라집니다)

| 항목 | 값 | 비고 |
|---|---|---|
| 도로 단계 | `CGC_STAGE=5` | **꼭 지정.** 기본값이 6(불법주정차 포함)이라 빼먹으면 6단계 도로에서 돌아갑니다 |
| 관측 | 67개 값 (`obs.*`) | `docs/data_columns.md`의 67개 그대로 |
| `OBS_SCAN` | 쓰지 않음 (설정하지 않거나 0) | 6단계 실험용 관측 보강입니다. 켜면 이 모델은 완주 32.2%로 떨어집니다 |
| shield | `shadow` (감시 모드) | 운전은 모델 혼자 하고, shield는 개입했을지만 기록 |
| 행동 | 결정적 | 가감속은 평균값, 차선은 argmax. `--stochastic` 쓰지 않음 |
| SUMO | 1.27.1 | Kaggle: `eclipse-sumo==1.27.1`, `traci==1.28.0`, `gymnasium==1.4.0` |
| 한 스텝 | 0.5초, 한 판 최대 3,600스텝 (1,800초) | 저장소 설정 그대로 |

## 모델 구조

`utils/networks.py`의 `HybridActorCritic`, `train.py`의 `HPARAMS`로 만든 구조와 같습니다.

| 부분 | 크기 | 출력 |
|---|---|---|
| 입력 | 67 | 관측 (−1~1로 정규화) |
| 몸통 `trunk` | 67 → 256 → 256, ReLU | |
| `mu_head` | 256 → 1, tanh | 가감속 (−1 최대 감속 ~ +1 최대 가속) |
| `lane_head` | 256 → 3 | 차선 명령. 0번 = 왼쪽(+1), 1번 = 유지(0), 2번 = 오른쪽(−1) |
| `v_head` | 256 → 1 | 가치 (실행에는 쓰지 않음) |

파일은 `state_dict` 그대로입니다 (`train.py`가 저장하는 PPO 모델과 같은 형식). 직전 행동 입력이나 순환 구조는 없습니다.

## 실행 방법

저장소 최상위 폴더에서 실행합니다.

데이터 수집 (한 판):

```bash
CGC_STAGE=5 python collect_pomdp_data.py --model models/dongyeon_bc/s5_student.pt --episodes 1 --name s5_check
```

- `--policy`는 생략합니다. 파일을 보고 `ppo` 형식으로 자동 판별합니다 (`bc`로 지정하면 읽지 못합니다).
- 수집 스크립트는 shield를 감시 모드로 고정하므로 따로 지정할 것이 없습니다.
- 결과 표(`data/s5_check.table.csv.gz`)는 한 줄이 한 스텝이고 컬럼은 167개여야 합니다 (`obs.*` 67개 포함).

평가:

```bash
CGC_STAGE=5 python models/bc_ppo/bcppo.py eval --ckpt models/dongyeon_bc/s5_student.pt --shield shadow --episodes 15 --seed-base 1000
```

## 성능 (shield shadow, 300판)

| 결과 | 판 수 | 비율 |
|---|---|---|
| 완주 (도착 + 놓친 정류장 0) | 257 | 85.7% |
| 충돌 | 13 | 4.3% |
| 시간초과 | 22 | 7.3% |
| 적색 통과 | 7 | 2.3% |
| 도착했지만 정류장 놓침 | 1 | 0.3% |

| 판 묶음 | 판 수 | 완주 | 충돌 |
|---|---|---|---|
| 테스트 시드 1000~1099 | 100 | 88.0% | 2.0% |
| 추가 시드 2000~2109 | 110 | 81.8% | 5.5% |
| 개발·학습 범위 90판 (900100~900129, 570000~570059) | 90 | 87.8% | 5.6% |

마지막 묶음은 Kaggle에서 돌렸을 때 완주 87.8%·충돌 6.7%·적색 3.3%·시간초과 2.2%, 수동개입률 0.3%였습니다.
자세한 내용은 저장소 `docs/reports/2026-11-26_강동연_청계천로_5단계_300판_재평가.md`에 있습니다.

## 학습 방법

1~4단계 학생을 차례로 이어받는 단계별 모방학습입니다.

1. 4단계 선생님(`models/bc_ppo/s4_teacher.pt`: PPO + shield 켬 + 정류장·신호·보호구역 규칙)이 5단계 도로에서 100판을 운전했습니다 (100판 모두 완주).
2. 4단계 학생(`models/bc_ppo/s4_student.pt`)에서 시작해 그 시범을 흉내 냈습니다 (BC 30 epoch, 감속 가중치 10, 차선 가중치 3).
3. DAgger를 한 번 더 해 봤지만 나아지지 않아(완주 87.8% 동일, 충돌 11.1%) 2번 결과를 최종으로 정했습니다.

학습 코드: `models/bc_ppo/bcppo.py`, Kaggle 커널 `models/dongyeon_bc/kaggle/s5v2.py`.

## 한계

- 실패 43판 중 22판이 시간초과입니다. 광교 교차로와 청계3가 교차로 안에서 멈춘 뒤 출발하지 않습니다.
- 충돌 13판 중 6판은 광교 다음 구간에서 정류장 쪽으로 차선을 바꾸다 난 측면 충돌입니다.
- 6단계(불법주정차) 도로에서는 쓸 수 없습니다. 세워진 차를 피하지 못해 대부분 시간초과가 납니다.
