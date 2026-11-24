# CGC_STAGE 아키텍처 개선 실험 기록 (Stage 3 Report: Entity Self-Attention)

## 1. 실험 목적 및 아키텍처 변경 내역 (utils/networks.py)
- **브랜치:** model/sunwoo-ppo-stage
- **커리큘럼 단계:** CGC_STAGE=1 (bus_stop 전용 — 청계천로 편도 2차로, 주변 차량 400대/시간)
- **변경 동기:** 기존 67차원 평면 관측 벡터 기반 2층 MLP(`build_mlp`)는 객체 간 관계 추론 없이 수치 가중합으로만 동작하여 사실상 규칙 기반(`if-else`) 근사에 그친다는 피드백 반영.
- **신규 아키텍처 (`DrivingEntityAttentionTrunk` + `HybridActorCritic`):**
  1. **Domain-Grouped Entity Tokenizer:** 67차원 관측을 9개의 물리적 의미 토큰(자차·규제 1, 5개 차선별 주변차·도로 5, 신호 1, 횡단보도 1, 정류장 1)으로 구조화.
  2. **Multi-Head Self-Attention (2-Layer Transformer Encoder, 4 Heads):** 자차와 주변 차선·신호·정류장 간 공간적 상호작용을 어텐션 가중치로 직접 추론.
  3. **Hybrid Action Head:** 종방향 가감속은 연속 `Normal`, 차선변경은 3-way `Categorical`로 분리하여 결정론적 평가 시 양자화 dead-zone 해소.

## 2. 실험 반복(Iteration) 및 지표 상세 해석
| 실험 Run | 주요 설정 | 평가/로그 시점 | 완주율 | 충돌률 | 정류장 놓침률 | 평균 속도 | 차선 변경 | 비고 |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| cgc_stage1_attn_200k | Entity-Attn + Hybrid (Stage 2 보상 잔존) | Rollout @51200 | 50% | 100% | 50% | 2.19 m/s | 8.21회 | `explained_variance=0.9926` 도달, 가치함수 표현력 확인 |
| cgc_stage1_attn_200k | 동일 | Eval @102400 (20ep) | 0% | 0% | 10% | 0.00 m/s | 0.0회 | `최소차간=12.9m` 유지하며 결정론적 평가 시 출발 포기 |
| cgc_stage1_attn_v2_200k | gap=8m, speed=0.10, keep_bias=1.2 | Rollout @155648 | **81%** | 91% | **11%** | 1.94 m/s | **6.14회** | 롤아웃 기준 완주율 81%·정류장 놓침 11%까지 상승 |
| cgc_stage1_attn_v2_200k | 동일 | Eval @153600 (20ep) | 0% | 0% | 0% | 0.00 m/s | 0.0회 | 탐험 노이즈 제거(`deterministic=True`) 시 평균 가속 `μ < 0` 수렴으로 정지 고착 |
| cgc_stage1_attn_ft_100k | v2 가중치 파인튜닝 (`col=5.0, stop_pen=0.20`) | Eval @51200 (20ep) | 0% | 0% | 0% | 0.00 m/s | 0.0회 | 이미 브레이크(`μ < 0`)로 고착된 정책은 미세조정만으로 탈출 불가 |

### 학습 결과의 핵심 의미
1. **객체 관계형 표현력 개선 입증:** Entity Self-Attention 도입 후 롤아웃 기준 완주율이 기존 MLP(74~76%)를 넘어 **81%**까지 상승하고, 차선변경 빈도도 **6.14회**로 안정화되었으며 Critic의 `explained_variance`가 **0.99** 수준에 도달함.
2. **결정론적 평가(Eval) 정지 고착 재현:** 그러나 탐험 분산(`std ≈ 0.55`)을 제거하고 평균 행동(`μ`)만 사용하는 평가 모드에서는 충돌·페널티 회피 성향이 결합해 `μ`가 음수(제동) 영역에 고착되면서 `0.0 m/s` 제자리 정지(시간초과 100%)가 반복됨. (상세 원인 및 대안은 `docs/experiments/single_ppo_limitations.md` 참조)

## 3. 재현 명령어
- Entity-Attention Stage 1 학습:
  `$env:CGC_STAGE=1; python train.py --timesteps 200000 --seed 0 --run-name cgc_stage1_attn_v2_200k`