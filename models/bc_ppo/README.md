# bc_ppo — 모방학습(BC) + PPO

바꾸는 축: **학습 방식** (보상·관측·네트워크는 `train.py` 기본값 그대로).

1. 선생님(Optimal Bus): PPO + shield **on**으로 학습 (`expert`)
2. 시범 데이터: 선생님이 shield on으로 운전, 정답 = 실제 적용된 행동 (`collect`)
3. BC: 같은 HybridActorCritic으로 흉내 + 가치 함수를 시범 할인 수익으로 미리 맞춤 (`bc`)
4. 학생 PPO: shield **off**, PPO 손실 + λ·BC 손실, λ는 1 → 0 (`finetune`).
   처음 warmup step은 가치 출력층만 학습 (BC 행동 보존)

최종 모델은 실행할 때 규칙 없이 혼자 운전한다. 평가는 shield 감시(shadow) 모드.
도로 단계: `CGC_STAGE=1~6`. 사용법은 `bcppo.py` 맨 위 설명.
