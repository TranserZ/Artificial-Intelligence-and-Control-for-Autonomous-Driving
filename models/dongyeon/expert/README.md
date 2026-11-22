# 모방학습: 규칙 선생 → GRU 학생 (강동연)

팀 의견(11/21): 규칙으로 잘 되는 운전자를 모방학습하자. 이 폴더는 1단계(선생 성능 측정)와
2단계(시범 데이터 수집)이다. 학습(BC → DAgger → PPO 이어 학습)은 다음 단계.

| 파일 | 내용 |
|---|---|
| `rule_expert.py` | 규칙 선생 `RuleExpert` (traci 특권 정보). 공통 평가 policy factory `make_policy` |
| `rule_expert_v3.json` | 공통 평가 `--ckpt` 자리에 넣는 선생 설명 파일 (가중치 없음) |
| `collect_demos.py` | 선생이 운전하며 (관측, 선생 행동) 기록 → `demos.npz`, `episodes.csv`, `meta.json`. `--student <bc.pt> --beta β`면 DAgger 수집 (학생이 운전, 라벨은 선생) |
| `train_bc.py` | GRU 학생 BC: 가감속 MSE + 차선 CE(클래스 가중치) + critic 수익 회귀 → `bc.pt`(GRU-PPO 형식) + `config.json` |

## 선생 성능 측정 (공통 평가 그대로)

```bash
CGC_STAGE=1 python scripts/evaluate_all.py --ckpt models/dongyeon/expert/rule_expert_v3.json \
  --model-name rule_expert --train-seed 0 --ckpt-step v3 --seeds 1000-1099 --shield shadow \
  --out docs/reports/rule_expert/stage1 --policy-factory models.dongyeon.expert.rule_expert:make_policy
```

선생은 학습 대상이 아니라 "학생이 닿을 수 있는 목표치"다. 단계별 선생 성적이 학생 판정 기준
(완주 95%·개입 16%)을 넘는지 먼저 본다. 선생이 못 넘는 단계는 학생도 모방만으로는 못 넘는다.

## 시범 데이터

```bash
CGC_STAGE=1 python models/dongyeon/expert/collect_demos.py --episodes 200 --out <폴더>
```

판 번호 20000 + 단계×10000 + i (테스트 1000~2999, 검증 900000~ 와 안 겹침, 코드가 검사).

## 자료조사 → 설계 결정

| 문제 (근거) | 우리 상황 | 대응 |
|---|---|---|
| **분포 이동·오차 누적**: 선생 상태에서만 배운 학생은 조금만 벗어나도 처음 보는 상태가 되어 오차가 쌓인다. DAgger는 학생이 운전한 상태에 선생 라벨을 붙여 모은다 (Ross, Gordon, Bagnell, AISTATS 2011) | 선생이 시뮬레이터 안 규칙이라 어떤 상태든 즉시 라벨을 붙일 수 있다 → DAgger가 공짜에 가깝다 | BC 1회 → DAgger 2~3회. 수집기에 `--noise`(실행 가감속에만 잡음, 라벨은 깨끗)로 회복 데이터도 만든다 |
| **특권 선생 → 관측만 보는 학생**: 특권 정보로 운전하는 선생을 관측만 보는 학생으로 옮기는 2단계 방식 (Learning by Cheating, Chen et al., CoRL 2019) | 선생은 traci로 앞차·신호·정류장을 정확히 본다. 학생은 67차원 관측 | 같은 구조. 학생 관측에 없는 정보는 GRU 기억으로 보완 |
| **모방 격차(imitation gap)**: 선생만 보는 정보는 학생 정책에서 "평균내져" 사라진다. 모방→RL 순차 전환이 해법이지만 한계가 있다 (Weihs et al., NeurIPS 2021, ADVISOR) | 관측에 없는 것: 신호 남은 시간 정확값, 가려진 차 등 | BC 뒤 PPO 이어 학습으로 보상이 격차를 메우게 한다. 단계별 판정은 학생 기준으로 |
| **copycat 문제**: 관측 기록(history)을 받는 BC 학생이 직전 선생 행동을 베껴 "다음 행동 = 직전 행동"으로 배운다 (Wen et al., NeurIPS 2020) | GRU 입력에 직전 행동 2개가 들어간다 (prev_action=True). 선생 가감속은 시간상 강하게 이어진다 | `prev_act`를 따로 저장. BC에서 직전 행동 입력을 0으로 두거나 학생 자신의 행동(DAgger)으로 바꿔 비교 |
| **인과 혼동**: 정보가 많을수록 BC가 엉뚱한 신호에 기댄다 (예: 앞 보행자 대신 브레이크등) (de Haan, Jayaraman, Levine, NeurIPS 2019) | 자기 속도 0 ↔ 정지 행동처럼 결과가 원인처럼 보이는 특징 | DAgger 개입 데이터가 해법 중 하나(같은 논문 비교군). 평가는 반드시 폐루프(공통 평가)로 |
| **BC 일반화 한계**: 데이터 편향·동적 객체·인과 모델 부재로 BC가 한계에 부딪힌다 (Codevilla et al., ICCV 2019) | 오토바이·끼어들기·주정차 같은 동적 객체가 4~6단계에 있다 | 단계별 데이터(1→6), 실패 판 분석 |
| **모방만으로는 위험 상황을 못 배운다**: 순수 모방은 충분히 강건하지 않아 교란(perturbation)과 손실 추가가 필요했다 (ChauffeurNet, Bansal et al., RSS 2019) | 선생이 거의 실패하지 않아 위험 상황 데이터가 적다 | `--noise`, DAgger, 그리고 PPO 이어 학습 |
| **BC로 시작 → RL 미세조정**: BC 초기화가 탐험을 이끌고, 모방 항을 점점 줄이면 선생을 넘을 수 있다 (DAPG, Rajeswaran et al., RSS 2018) | 순수 PPO는 6단계에서 완주 0% — 탐험이 문제 | BC 가중치를 `--init`으로 넣어 PPO. 필요하면 BC 손실을 줄여 가며 같이 쓴다 |
| **미세조정 중 망각**: RL 미세조정이 초반에 안 가는 상태 공간의 능력을 잊는다. BC 손실·kickstarting 같은 지식 유지 기법이 막는다 (Wołczyk et al., ICML 2024) | 정류장 5곳 중 뒤쪽 정류장·주정차 구간은 초반에 잘 못 간다 | PPO에 BC 보조 손실(점점 감소)을 둔다 |
| **critic 초기화**: actor만 BC로 시작하면 critic이 엉망이라 초반 PPO 업데이트가 정책을 망칠 수 있다. 선생/BC 정책 rollout 수익으로 critic을 먼저 학습하는 방법이 제안됐다 (2026 arXiv 2602.23804, 미검증 preprint) | 우리 `--init`은 critic도 BC 가중치(학습 안 됨) 그대로 | PPO 첫 몇 업데이트는 actor를 고정하고 critic만 학습(warm-up) |

### 데이터에서 바로 보인 것 (1단계 3판 시험 수집)

- 차선 변경 라벨이 **판당 1번(0.15%)**뿐이다. 그대로 BC하면 학생은 "차선 유지"만 배워 정류장 차로로
  못 간다 → 차선 head 손실에 클래스 가중치 또는 차선 변경 전후 구간 과대표집.
- 선생도 shield 감시 기준 개입이 판에 따라 0~27 step 생긴다 (판정 기준은 개입률 16% 이하).

## 선생 공식 성적 (Kaggle, 판 1000~1099, 감시 모드, `scripts/evaluate_all.py`)

| 단계 | v3 완주(정류장 5곳) | v4 완주 | v4 충돌 | v4 시간초과 | v4 개입률 | v4 판정 (95%·16%) |
|---|---|---|---|---|---|---|
| 1 | 100% | **100%** | 0 | 0 | 5.4% | 통과 |
| 2 | 100% | **100%** | 0 | 0 | 8.6% | 통과 |
| 3 | 94% | **99%** | 1% | 0 | 4.7% | 통과 |
| 4 | 94% | **97%** | 3% | 0 | 5.4% | 통과 |
| 5 | 94% | **100%** | 0 | 0 | 5.5% | 통과 |
| 6 | 77% | 73% | 3% | 24% | 2.8% | 미달 |

- v3 → v4: 3~5단계 충돌(옆길 합류 차, 교차로 안 0.8 km)을 양보 규칙으로 해결.
- 6단계 시간초과는 주교 불법주정차(e218924602_4, 34 m, 교차로 바로 뒤 오른쪽 차로, 1.61 km).
  왼쪽 차로가 12~15 m 간격으로 같은 속도로 줄지어 가 8 m 틈이 없고, 교차로 안에서는 차선을 못 바꾼다.
  4 m 틈 끼어들기·감속해 틈 찾기를 시도하면 10판 중 6판 해결되지만 새 충돌이 생겨(로컬) v4에 넣지 않았다.
  환경 설계(다발지점 위치·확률) 문제로 보고 팀 결정 사항으로 남긴다.

## 시범 데이터 v4 (`~/work/cgc-demos/v4_clean/stage{1..6}`, 커밋 1c319f0, 단계별 200판)

| 단계 | 스텝 | 선생 완주 | 차선 변경 라벨 비율 |
|---|---|---|---|
| 1 | 139,816 | 100% | 0.14% |
| 2 | 223,531 | 100% | 0.30% |
| 3 | 236,253 | 98.5% | 0.39% |
| 4 | 236,647 | 97.5% | 0.35% |
| 5 | 245,409 | 99% | 0.30% |
| 6 | 344,201 | 82% | 0.77% |

## 3단계: BC → DAgger (Kaggle `cgc-gru-bc1-ts{0,1,2}`)

```bash
CGC_STAGE=1 python models/dongyeon/expert/train_bc.py --data <시범 폴더> --out r0 --epochs 30
CGC_STAGE=1 python models/dongyeon/expert/collect_demos.py --student r0/bc.pt --beta 0.5 --episodes 25 --seed-base 31000 --out d1
CGC_STAGE=1 python models/dongyeon/expert/train_bc.py --data <시범 폴더> d1 --init r0/bc.pt --out r1 --epochs 12
```

회차 β = 0.5, 0.2, 0, 0 (회차당 100판, 데이터는 누적). r0·r2·r4를 공통 평가.
주의: `demos.npz`의 `lane_cls`는 0=오른쪽이고 학생 차선 head는 0=왼쪽이라, `train_bc.py`는 `act[:,1]`에서 직접 만든다.

## 참고 문헌

- Ross, Gordon, Bagnell. A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning. AISTATS 2011. https://proceedings.mlr.press/v15/ross11a.html
- Chen, Zhou, Koltun, Krähenbühl. Learning by Cheating. CoRL 2019. https://arxiv.org/abs/1912.12294
- Weihs et al. Bridging the Imitation Gap by Adaptive Insubordination. NeurIPS 2021. https://arxiv.org/abs/2007.12173
- Wen, Lin, Darrell, Jayaraman, Gao. Fighting Copycat Agents in Behavioral Cloning from Observation Histories. NeurIPS 2020. https://arxiv.org/abs/2010.14876
- de Haan, Jayaraman, Levine. Causal Confusion in Imitation Learning. NeurIPS 2019. https://arxiv.org/abs/1905.11979
- Codevilla, Santana, López, Gaidon. Exploring the Limitations of Behavior Cloning for Autonomous Driving. ICCV 2019. https://arxiv.org/abs/1904.08980
- Bansal, Krizhevsky, Ogale. ChauffeurNet: Learning to Drive by Imitating the Best and Synthesizing the Worst. RSS 2019. https://arxiv.org/abs/1812.03079
- Rajeswaran et al. Learning Complex Dexterous Manipulation with Deep RL and Demonstrations (DAPG). RSS 2018. https://arxiv.org/abs/1709.10087
- Wołczyk et al. Fine-tuning Reinforcement Learning Models is Secretly a Forgetting Mitigation Problem. ICML 2024. https://arxiv.org/abs/2402.02868
- Actor-critic pretraining for PPO (arXiv 2602.23804, 2026, preprint). https://arxiv.org/abs/2602.23804
