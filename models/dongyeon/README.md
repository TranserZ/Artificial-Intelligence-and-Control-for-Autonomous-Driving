# GRU-PPO (강동연)

> **한 줄 요약** · 베이스라인 PPO와 환경·보상·하이퍼파라미터는 같고, **네트워크에 기억(GRU)만 추가**했다. 신호가 맨 앞 차일 때만 보이는 부분 관측에서, 지나간 관측을 기억해 판단하게 하려는 실험이다.

| 항목 | 내용 |
|---|---|
| 바꾸는 축 | 알고리즘/네트워크 (MLP → GRU 기억) — 1인 1축 |
| 기반 | `env/v2-prep` (`5e78ec7`). `env-v2` 태그가 붙으면 그 태그로 다시 맞춘다 |
| 학습 | `python models/dongyeon/train_gru_ppo.py --seed {0,1,2} --total-steps 500000 --out-dir models/dongyeon/checkpoints/ts{시드} --max-hours 11` |
| 평가 | `scripts/evaluate_all.py ... --policy-factory models.dongyeon.eval_adapter:make_policy` |
| 검사 | `python models/dongyeon/selftest.py` (SUMO 없이 가짜 환경으로 정확성 검사) |

## 왜 GRU인가

- **부분 관측**: env-v1부터 신호는 버스가 맨 앞일 때만 보이고, 뒤에 서 있으면 가려진다. 신호 잔여시간도 없다.
  - 한 스텝 관측만 보는 MLP는 "방금 녹색이 됐다", "앞차 뒤에 서기 전 신호가 무슨 색이었다", "여기서 얼마나 서 있었다"를 알 수 없다.
- **베이스라인 실패 모습**: env-v1 베이스라인 PPO는 테스트 100판 완주 0%였다. 대부분 신호 앞에서 맨 앞 차로 선 뒤, 녹색이 와도 출발하지 않고 시간 초과로 끝났다 (`docs/results/baseline_ppo.md`, 브랜치 `model/baseline-ppo`).
- **교수님 피드백**: "MLP로는 부족할 것"

## 네트워크

```
[관측 67 + 직전 행동 2] ─┬─ actor:  Linear 256 + ReLU → GRU 256 → 가감속 μ (tanh), 차선 로짓 3
                         └─ critic: Linear 256 + ReLU → GRU 256 → V
가감속 σ: 상태와 무관한 학습 파라미터 (초기 0.61)
```

| 결정 | 근거 |
|---|---|
| GRU | POPGym (Morad et al., ICLR 2023): 기억 모델 13개 중 GRU가 가장 좋은 범용 기억 모델 |
| actor·critic이 GRU를 따로 가짐 | Ni et al., "Recurrent Model-Free RL Can Be a Strong Baseline for Many POMDPs" (ICML 2022) 4·5.2절: 인코더를 공유하면 critic gradient가 actor를 압도해 학습을 방해한다 |
| 입력에 직전 행동 | Ni et al. 4절: 과거 행동·보상을 넣으면 숨은 상태 추정이 쉬워진다. 직전 행동은 버스가 직접 낸 명령이라 새로운 외부 정보가 아니다. **보상은 넣지 않는다** (운전 입력이 아님) |
| 첫 층 256 → GRU 256 | 베이스라인 MLP 256×256의 둘째 층을 GRU로 바꾼 것. 크기를 같게 둬서 "기억" 효과만 비교 |
| 직교 초기화, 출력층 gain 0.01 | 베이스라인과 같은 관례 (Huang et al., "The 37 Implementation Details of PPO") |

## 학습 (truncated BPTT)

- rollout 2048 스텝을 **64 스텝(32초) 조각 32개**로 자른다. 각 조각 시작의 은닉 상태를 rollout 때 저장해 두고, 학습할 때 그 상태에서 조각을 다시 계산한다.
  - SB3-contrib `RecurrentPPO`, CleanRL `ppo_atari_lstm`, MarcoMeter `recurrent-ppo-truncated-bptt`와 같은 방식이다.
- **판이 바뀌는 스텝에서는 은닉 상태와 직전 행동을 0으로** 만든다. 학습과 실행이 같은 규칙을 쓴다.
- 미니배치 = 조각 2개(128 스텝). 업데이트당 16 미니배치 × 10 에폭.
- **실행할 때 기억은 판 전체로 이어진다.** gradient만 64 스텝에서 끊긴다.

## 베이스라인과 같은 것 / 다른 것

| 같은 것 (`train.py`에서 그대로 import) | 다른 것 |
|---|---|
| 보상 가중치 `REWARD` 전체 | 네트워크 (위) |
| shield 학습 중 off, 검증 감시 모드 | 시간 초과 처리 버그 수정 (아래) |
| 실패 판 다시 열기 0.3 | best 선택 기준 (아래) |
| 학습 판 시드 제외 범위, 검증 판 900000~900019 (20판) | 검증 주기 25 → 50 업데이트 (약 10만 step) |
| lr 1e-4, n_steps 2048, epochs 10, γ 0.99, λ 0.95, clip 0.2, value 0.5, 차선 엔트로피 0.01, gradient clip 0.5, 초기 σ 0.61, 차선 유지 편향 0.5 | 미니배치: 무작위 64 스텝 → 64 스텝 조각 2개 (순서를 지켜야 해서) |
| 30만 / 50만 / final 체크포인트, `*.resume.pt` | `--max-hours`: Kaggle 시간 제한 전에 업데이트 경계에서 멈추고 이어 학습 파일을 남김 |

### 공용 PPO와 다르게 처리한 것

1. **시간 초과(truncated) 처리**: 공용 `algorithms/ppo.py`는 버퍼에 `done=terminated`만 넣는다. 그래서 시간 초과 스텝에서 문제가 생긴다.
   - γV(s')를 보상에 더한 뒤 다음 판 첫 상태의 V까지 한 번 더 더한다.
   - GAE가 다음 판으로 새어 나간다.
   - 여기서는 `done = terminated or truncated`로 판을 끊고, 시간 초과는 γV(s')만 더한다.
   - 베이스라인 판의 76%가 시간 초과였으므로 영향이 작지 않다. 팀에 따로 알린다.
2. **best 선택**: 공용 `train.py`는 완주 0%일 때 수동개입률이 가장 낮은 모델을 고른다. 그러면 서 있는 모델이 best가 된다. 여기서는 공식 판정에 맞춰 순서대로 고른다.
   - 자동 판정 통과(공식 완주 ≥95%, 충돌 <5%) 중 개입률이 가장 낮은 것
   - 통과한 게 없으면 공식 완주율 → 충돌률 → 주행 거리 순
   - 공식 결과는 어차피 `final.pt`라 비교에는 영향이 없다.

## 평가할 때 기억 지우기

공통 평가 코드는 `predict(obs)`만 부르고 판 시작을 알려 주지 않는다. 그래서 `eval_adapter.make_policy`가 `env.reset`을 감싸서, 판이 시작될 때마다 기억을 0으로 만든다. 공용 파일은 고치지 않는다.

`selftest.py`가 확인하는 것:
- rollout 때 한 스텝씩 계산한 값과 학습 때 조각으로 다시 계산한 값이 같은지 (log π, V 오차 1e-4 미만)
- 판 경계에서 리셋되는지, GAE가 다음 판으로 새지 않는지
- 저장·불러오기와 이어 학습이 맞는지
- 같은 판을 두 번 돌리면 같은 행동이 나오는지

## 참고 문헌·코드

- Ni, Eysenbach, Salakhutdinov. *Recurrent Model-Free RL Can Be a Strong Baseline for Many POMDPs.* ICML 2022. https://arxiv.org/abs/2110.05038
- Morad et al. *POPGym: Benchmarking Partially Observable Reinforcement Learning.* ICLR 2023. https://arxiv.org/abs/2303.01859
- Pleines et al. *Generalization, Mayhems and Limits in Recurrent Proximal Policy Optimization.* 2022. https://arxiv.org/abs/2205.11104 / https://github.com/MarcoMeter/recurrent-ppo-truncated-bptt
- SB3-contrib RecurrentPPO: https://sb3-contrib.readthedocs.io/en/master/modules/ppo_recurrent.html
- Huang et al. *The 37 Implementation Details of Proximal Policy Optimization.* ICLR Blog Track 2022.
