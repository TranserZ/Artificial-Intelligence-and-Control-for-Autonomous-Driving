# CGC_STAGE 단계별 PPO 커리큘럼 학습 기록 (Stage 1)

## 1. 실험 환경
- **브랜치:** `model/sunwoo-ppo-stage` (`origin/env/v3-merge` 기반)
- **커리큘럼 단계:** `CGC_STAGE=1` (`bus_stop` 전용)
- **목표:** 충돌률 5% 이하 조건에서 버스정류장 정차 및 완주 기본기 확보

## 2. 보상 가중치 조정 내역 (`env/mdp_config.py`)
| 파라미터 | 기본값 | 수정값 | 변경 의도 |
| :--- | :---: | :---: | :--- |
| `speed_weight` | `1.0` | `0.0` | 단순 속도 유지보다 전진 거리 중심으로 유도 |
| `progress_weight` | `5.0` | `10.0` | 경로 진행 보상 상향 |
| `green_stop_penalty` | `0.0` | `0.08` | 불필요한 정차/저속 정체 페널티 |
| `green_stop_extra` | `0.0` | `0.15` | 저속 정체 추가 페널티 강화 |

## 3. Stage 1 학습 결과
- **200k 학습 (`cgc_stage1_real_200k`):** 최고 평가(`@153600`) **공식완주율 60%, 충돌률 0%, 수동개입률 0.0%** (충돌률 5% 이하 통과)
- **300k 추가 학습 (`cgc_stage1_real_300k`):** 롤아웃 `completion_rate=0.76` 달성했으나 결정론적 평가에서는 속도 보수화(`0.5m/s`) 발생 → **Stage 2 초기 가중치로 `cgc_stage1_real_200k/best_model.pt` 채택**
