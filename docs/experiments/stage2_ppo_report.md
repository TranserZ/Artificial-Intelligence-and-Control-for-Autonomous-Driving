# CGC_STAGE 단계별 PPO 커리큘럼 학습 기록 (Stage 2)

## 1. 실험 목적 및 환경
- **브랜치:** model/sunwoo-ppo-stage (origin/env/v3-merge 기반)
- **커리큘럼 단계:** CGC_STAGE=2 (bus_stop + signal — 청계천로 편도 2차로, 6개 교차로 신호·횡단보도·정류장 활성화, 주변 차량 400대/시간)
- **초기 가중치(--init):** `results/cgc_stage1_real_200k/best_model.pt` (관측 차원 67 → 67, 신호 입력 가중치 0 초기화 전이)
- **핵심 목표:** Stage 1의 정류장 정차·주행 기본기를 유지한 채 적색 신호 정지 및 전방 정차 차량 제동 학습

## 2. 보상 가중치 조정 내역 (env/mdp_config.py)
Stage 1 가중치 전이 직후에는 ① 전방 신호·차량 감속 가중치 부재로 인한 고속 추돌 및 신호 위반 (충돌률 65~70%, 신호위반 69%), ② 과도한 감속 페널티 부여 시 결정론적(Eval) 정책이 출발 자체를 포기하는 보수화(평균속도 0.0~0.7 m/s, 시간초과 90%)가 번갈아 발생함. 이를 해결하기 위해 제동 유도 페널티와 전진 속도의 균형을 단계적으로 조정함.

| 파라미터 | Stage 1 기준 | Stage 2 최종 수정값 | 변경 의도 및 물리적 의미 |
| :--- | :---: | :---: | :--- |
| collision_penalty | 5.0 | 12.0 | 충돌 1회 벌점을 완주 보상(+12.0)과 동일 수준으로 상향하여 추돌 회피 최우선화 |
| red_light_penalty | 5.0 | 10.0 | 적색 신호 돌파 시 강한 종료 페널티 부여 |
| red_light_terminate | True | True | 위반 즉시 에피소드를 종료하여 신호 무시 연쇄 학습 방지 |
| approach_overspeed_penalty | 0.0 | 0.18 | 적색 신호 접근 구간 감속 커브 초과 시 매 스텝 페널티로 조기 제동 유도 |
| close_gap_threshold | 10.0 | 12.0 | 전방 차량 안전거리 감지 임계값 현실화 (20m 적용 시 도심 정체 구간 출발 불가 해결) |
| close_gap_penalty | 0.10 | 0.15 | 안전거리 이내 근접 시 감속 유도 |
| speed_weight | 0.0 | 0.06 | 결정론적 평가 시 0.0 m/s 정지 고착을 방지하고 적정 순항 전진 유인 제공 |

## 3. Stage 2 실험 반복(Iteration) 및 지표 상세 해석
| 실험 Run | 주요 설정 | 평가/로그 시점 | 완주율 | 충돌률 | 신호위반 | 정류장 놓침률 | 평균 속도 | 비고 |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| cgc_stage2_tuned_200k | col=5.0, red=5.0, approach=0.08 | Rollout @104448 | 16% | 31% | 69% | 27% | 1.93 m/s | 신호 감속 유인 부족으로 적색 신호 돌파 다발 |
| cgc_stage2_v4_200k | red_terminate=False, lr=2e-4 | Eval @102400 (20ep) | 0% | 70% | 30% | 15% | 4.60 m/s | 순항 속도 및 정류장 정차 회복, 결정론적 평가 시 제동 지연 추돌 |
| cgc_stage2_brake_200k | approach=0.35, gap=20m, speed=0.0 | Eval @51200 (20ep) | 0% | 0% | 0% | 0% | 0.00 m/s | Rollout 완주율 27% 달성했으나 Eval에서 과도한 보수화로 출발 포기 |
| cgc_stage2_overnight_500k | speed=0.06, gap=12m, approach=0.18 | Rollout @462848 | **41%** | 51% | 49% | **15%** | **3.95 m/s** | 훈련 롤아웃 기준 완주율 41%·정류장 놓침 15%까지 상승 |
| cgc_stage2_overnight_500k | 동일 (Eval, shield=shadow) | Eval @358400 (20ep) | 0% | 70% | 30% | **5%** | 2.60 m/s | `best_model.pt` 저장 시점 (정류장·양보·구역과속·차로오류 0~5%) |
| cgc_stage2_overnight_500k | 동일 (Test, shield=on) | Test 20ep | 0% | **10%** | **0%** | 20% | 0.63 m/s | 안전 쉴드 개입 시 신호위반 0%·충돌 10%로 억제되나 교착 시간초과(90%) 발생 |

### 핵심 진단 요약
1. **훈련(Rollout) vs 결정론적 평가(Eval) 간 행동 분포 괴리:**
   - 50만 스텝 학습 결과 탐험 노이즈(`std=0.40`)가 포함된 롤아웃에서는 완주율 **41%**, 정류장 놓침률 **15%**, 평균속도 **3.95 m/s**까지 도달함.
   - 반면 노이즈를 제거한 평가(`deterministic=True`)에서는 차선 변경 빈도가 `0.76~3.62회`에서 `0.1회`로 급감하고, 가속 출력 평균이 한쪽으로 치우쳐 정체 차선 뒤 추돌(70%) 또는 안전 쉴드 개입 시 교착(시간초과 90%) 현상이 고착됨.
2. **시각화(SUMO GUI) 검증 결과:**
   - `test.py` 직접 주행 확인 결과, 정류장 정차 및 차선 유지는 정상 작동하나 적색 신호 및 정지 차량 전방에서의 풀브레이크(`accel < 0`) 전환 타이밍이 반 박자 늦거나 결정론적 모드에서 차선 회피가 작동하지 않는 것이 주원인으로 확인됨.

## 4. 향후 개선 방향 (Next Steps)
1. **결정론적 평가(Eval) 교착 및 추돌 해소:** `env/` 내 전방 차량 TTC(Time-to-Collision) 기반 제동 보상 강화 및 정체 차선 회피(차선 변경) 유도 관측·보상 점검.
2. **Stage 2 목표 지표(충돌률 <= 5%, 공식 완주율 확보) 달성 후 Stage 3 전환:** 순수 모델 및 안전 쉴드 결합 기준 교착 없이 완주율 확보 후 다음 커리큘럼 단계로 확장.

## 5. 재현 명령어
- Stage 2 밤샘 학습(500k):
  `$env:CGC_STAGE=2; python train.py --init results\cgc_stage1_real_200k\best_model.pt --timesteps 500000 --seed 0 --lr 0.00015 --replay-prob 0.15 --save-steps 150000,300000,450000 --run-name cgc_stage2_overnight_500k`
- 학습 모델 GUI / Shield 평가:
  `$env:CGC_STAGE=2; python test.py results\cgc_stage2_overnight_500k\best_model.pt --nogui --episodes 20 --shield on`
