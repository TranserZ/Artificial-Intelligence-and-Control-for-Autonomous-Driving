# 청계천 레벨 4 자율주행 버스 도입을 위한 SUMO 기반 시뮬레이션 환경 구축 (BC + KL 정규화 PPO 미세조정)

서울 청계천로를 달리는 자율주행 버스 노선(청계A01) 2.1 km를 교통 시뮬레이터 SUMO로 재현한 프로젝트입니다. 
버스 정책은 두 단계로 학습했습니다. 먼저 BC로 시연 데이터를 모방해 운전을 배우고, 그다음 KL 정규화 PPO(KL-regularized PPO)로 미세조정했습니다.
KL 정규화 PPO는 PPO로 보상을 높이면서, 출발점인 행동 복제 정책(기준 정책, reference policy)과의 KL 발산(KL divergence)에 벌점을 주어 정책이 기준 정책에서 너무 멀어지지 않게 하는 방법이며, 아래 두 연구에서 참고했습니다.

- InstructGPT: Ouyang et al., 2022, [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155). 사전학습된 언어 모델을 PPO로 미세조정할 때, 원래 모델과의 KL 발산에 벌점을 주었습니다.
- VPT: Baker et al., 2022, [Video PreTraining (VPT): Learning to Act by Watching Unlabeled Online Videos](https://arxiv.org/abs/2206.11795). 행동 복제로 학습한 정책을 강화학습으로 미세조정할 때, 원래 정책과의 KL 발산에 벌점을 주어 이미 배운 능력을 잊지 않게 했습니다.

이 방법을 쓴 이유는 다음과 같습니다.
1. BC로만 한다면 전문가를 뛰어넘을 수 없을 것이라 생각했습니다. 
2. PPO로만 미세조정해 보니, 학습이 진행될수록 모델이 처음의 BC 모델과 달라지면서 이미 잘하던 운전까지 무너졌습니다.

코드와 파일 이름에서는 이 방법을 줄여서 KL-PPO(`kl_ppo`, `klppo`)라고 부릅니다.

## 링크

| 항목 | 주소 |
|---|---|
| 프로젝트 페이지 | https://transerz.github.io/Artificial-Intelligence-and-Control-for-Autonomous-Driving/ |
| 코드 (GitHub) | https://github.com/TranserZ/Artificial-Intelligence-and-Control-for-Autonomous-Driving |
| 주행 데이터 (Hugging Face) | https://huggingface.co/datasets/CheonggyeRoad/Dataset |

## 최종 모델

| 항목 | 내용 |
|---|---|
| 모델 파일 | `results/klppo1_s5_s0/step250k.pt` |
| 학습 설정 | `results/klppo1_s5_s0/config.json` |
| 학습 방법 | 5단계 행동 복제 정책(`results/s5_student/s5_student.pt`)을 기준 정책으로 두고, 가속 출력만 KL 정규화 PPO로 25만 스텝 동안 미세조정했습니다. |
| 실행 설정 | `CGC_STAGE=5`, `OBS_SCAN=0` |
| 팀 완주율 목표 | 85% 이상입니다. 처음에는 95%였지만 중간발표 이후 최종 기준을 85%로 조정했습니다. |

모델을 돌릴 때는 반드시 `CGC_STAGE=5`와 `OBS_SCAN=0`을 설정해야 합니다.
설정하지 않으면 코드 기본값인 6단계 도로에서 돌게 되어, 최종 모델이 학습한 환경과 달라집니다.

`results/s5_student/s5_student.pt`는 미세조정의 출발점이자 KL 벌점의 기준이 되는 행동 복제 정책입니다. 최종 주행 모델이 아닙니다.
## 실행 환경

- 운영체제는 Windows 10 또는 11입니다. macOS와 Linux에서도 SUMO 설치 경로만 바꾸면 됩니다.
- Python 3.13을 사용합니다.
- PyTorch 2.15를 사용하며 CPU만으로 충분합니다. 나머지 패키지는 `requirements.txt`에 있습니다.
- SUMO는 **1.27.1**을 사용합니다. https://sumo.dlr.de/docs/Downloads.php 에서 설치할 수 있습니다.
  - `SUMO_HOME`을 SUMO 설치 폴더로 설정해야 합니다. Windows 설치 프로그램에서 "Set SUMO_HOME"을 체크하면 자동으로 설정됩니다.
  - `PYTHONPATH`에 `SUMO_HOME` 안의 `tools` 폴더를 추가해야 합니다. 파이썬이 SUMO를 조종하는 모듈(traci, sumolib)을 이 폴더에서 가져옵니다.
  - Windows에서는 `PYTHONUTF8=1`도 설정해 주세요. 한글과 기호가 들어간 로그를 파일로 저장할 때 생기는 인코딩 오류를 막아 줍니다.

PowerShell에서는 아래처럼 설정합니다. 이 방법은 지금 열린 창에만 적용됩니다. 계속 쓰려면 시스템 환경 변수에 같은 값을 등록하세요.

```powershell
$env:SUMO_HOME = "C:\Program Files (x86)\Eclipse\Sumo"
$env:PYTHONPATH = "$env:SUMO_HOME\tools"
$env:PYTHONUTF8 = "1"   # 한글과 기호 로그를 파일로 저장할 때 cp949 인코딩 오류 방지 (Windows)
sumo --version   # Eclipse SUMO sumo 1.27.1
```

bash(macOS, Linux, Windows Git Bash)에서는 아래처럼 설정합니다.

```bash
export SUMO_HOME=/usr/share/sumo        # 설치 경로에 맞게
export PYTHONPATH="$SUMO_HOME/tools"
export PYTHONUTF8=1
```

## 설치

```powershell
git clone https://github.com/TranserZ/Artificial-Intelligence-and-Control-for-Autonomous-Driving.git
cd Artificial-Intelligence-and-Control-for-Autonomous-Driving
py -3.13 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\Activate.ps1      # 이후 python = 이 가상환경
```

`Activate.ps1`을 실행할 때 "스크립트를 실행할 수 없습니다"라는 오류가 나면, 먼저 아래 명령을 실행한 뒤 다시 시도하세요.

```powershell
Set-ExecutionPolicy -Scope Process Bypass
```

bash에서는 아래 한 줄로 설치합니다.

`python3.13 -m venv .venv && .venv/bin/python -m pip install -r requirements.txt && source .venv/bin/activate`

아래 명령은 모두 **저장소 맨 위 폴더**에서 실행합니다.
도로망은 실행할 때마다 `data/osm`, `data/road_data`, `config/lines.yaml`을 바탕으로 새로 만들어집니다.

## 폴더 구조

| 폴더 | 들어 있는 것 |
|---|---|
| `environment/` | SUMO 주행 환경입니다. 도로, 신호, 정류장 설정(`road_config.py`), 관측과 보상 설정(`mdp_config.py`), 도로 생성(`road_builder.py`), 단계별 맵(`curriculum.py`), 어린이보호구역 표지 그림, 네트워크 파일이 있습니다. |
| `training/` | 학습 스크립트입니다. PPO와 KL 정규화 PPO 학습(`train.py`), 행동 복제 학습(`train_bc.py`)이 있습니다. |
| `evaluation/` | 평가 스크립트입니다. 여러 판을 동시에 평가하는 `drive_parallel.py`, SUMO 화면으로 한 판을 보는 `gui_drive_seed.py`, 모델 하나를 재생하는 `test.py`가 있습니다. |
| `data_collection/` | 주행 데이터 수집 스크립트(`collect_pomdp_data.py`)입니다. 데이터 컬럼 설명은 `docs/data_columns.md`에 있습니다. |
| `algorithms/` | PPO, KL 정규화 PPO(`kl_ppo.py`), 행동 복제 정책 코드입니다. 학습, 평가, 수집이 함께 사용합니다. |
| `utils/` | 신경망, 버퍼, 로거, 평가 지표 코드입니다. |
| `tools/` | 지도 데이터(OSM)를 SUMO 도로로 바꾸고, 시나리오와 신호, 노면 표시를 만드는 도구입니다. 주행 환경이 실행 중에 사용합니다. |
| `config/` | 버스 노선과 정류장 설정(`lines.yaml`)입니다. |
| `data/` | OSM 지도와 실제 도로 자료입니다. 차로 수, 신호 시간, 어린이보호구역 정보가 있습니다. |
| `docs/` | 데이터 컬럼 문서, 팀 실험 가이드, 최종 보고서가 있습니다. |
| `results/` | 최종 모델(`klppo1_s5_s0/`), 미세조정의 기준이 되는 행동 복제 정책(`s5_student/`), 보상 설정 확인용 파일(`exp-ppo-stage1/`)이 있습니다. |

## 평가

```powershell
$env:CGC_STAGE="5"; $env:OBS_SCAN="0"; python evaluation/drive_parallel.py --model results/klppo1_s5_s0/step250k.pt --seeds 900300-900399 --shield shadow --workers 4 --json-out results/local_evaluation/stage5.json
```

```bash
CGC_STAGE=5 OBS_SCAN=0 python evaluation/drive_parallel.py --model results/klppo1_s5_s0/step250k.pt --seeds 900300-900399 --shield shadow --workers 4 --json-out results/local_evaluation/stage5.json
```

1판만 빠르게 확인하는 명령입니다.

```powershell
$env:CGC_STAGE="5"; $env:OBS_SCAN="0"; python evaluation/drive_parallel.py --model results/klppo1_s5_s0/step250k.pt --seeds 900300 --shield shadow --workers 1
```

```bash
CGC_STAGE=5 OBS_SCAN=0 python evaluation/drive_parallel.py --model results/klppo1_s5_s0/step250k.pt --seeds 900300 --shield shadow --workers 1
```

SUMO 화면(SUMO-GUI)으로 버스가 달리는 모습을 한 판 보는 명령입니다.

```powershell
$env:CGC_STAGE="5"; $env:OBS_SCAN="0"; python evaluation/gui_drive_seed.py --stage 5 --seed 900377 --checkpoint results/klppo1_s5_s0/step250k.pt --delay 80
```

```bash
CGC_STAGE=5 OBS_SCAN=0 python evaluation/gui_drive_seed.py --stage 5 --seed 900377 --checkpoint results/klppo1_s5_s0/step250k.pt --delay 80
```

## 데이터 수집

최종 모델이 운전하면서 관측, 행동, 상태를 저장합니다.
결과는 세 가지 파일로 나옵니다. 컬럼 표(`.table.csv.gz`), 학습용 배열(`.npz`), 전체 기록(`.jsonl.gz`)입니다. 기본 저장 폴더는 `data/`입니다.
팀이 수집한 전체 주행 데이터는 Hugging Face에 있습니다.

```powershell
$env:CGC_STAGE="5"; $env:OBS_SCAN="0"; python data_collection/collect_pomdp_data.py --model results/klppo1_s5_s0/step250k.pt --seed 5000000 --episodes 1 --workers 1 --name s5_check
```

```bash
CGC_STAGE=5 OBS_SCAN=0 python data_collection/collect_pomdp_data.py --model results/klppo1_s5_s0/step250k.pt --seed 5000000 --episodes 1 --workers 1 --name s5_check
```

위 명령(시드 5000000, 1판)을 실행하면 1,021행, 167열짜리 컬럼 표가 만들어집니다.
버스는 목적지에 도착하고(`event.arrived` = 1), 마지막 주행거리(`state.ego.odometer`)는 2136.31 m입니다. 결과가 이와 같으면 설치가 제대로 된 것입니다.
여러 판을 수집하려면 `--episodes`와 `--workers`를 늘리면 됩니다.

## 학습

최종 모델을 만든 KL 정규화 PPO 미세조정 명령입니다. `--ref`가 기준 정책(행동 복제 정책)입니다.
5단계 행동 복제 정책에서 출발해 30만 스텝까지 학습하고, 2만 5천 스텝마다 모델을 저장합니다. 그중 `step250k.pt`를 최종 모델로 골랐습니다.

```powershell
$env:CGC_STAGE="5"; $env:OBS_SCAN="0"; python training/train.py --ref results/s5_student/s5_student.pt --explore-std 0.07772492733277317 --seed 0 --run-name klppo1_s5_s0_repro
```

```bash
CGC_STAGE=5 OBS_SCAN=0 python training/train.py --ref results/s5_student/s5_student.pt --explore-std 0.07772492733277317 --seed 0 --run-name klppo1_s5_s0_repro
```

학습을 시작할 때 두 가지를 검사합니다. 보상 설정 40개 항목이 `results/exp-ppo-stage1/config.json`과 같은지, `CGC_STAGE`와 도로 단계가 맞는지 확인하고, 다르면 학습을 멈춥니다.
결과는 `results/<run-name>/`에 저장됩니다.

행동 복제 정책은 수집한 시연 데이터(`.npz` 또는 `.jsonl.gz`)로 학습합니다. 옵션은 `--help`로 확인할 수 있습니다.

```powershell
$env:CGC_STAGE="5"; $env:OBS_SCAN="0"; python training/train_bc.py --help
```

```bash
CGC_STAGE=5 OBS_SCAN=0 python training/train_bc.py --help
```

## 환경변수

| 변수 | 최종 모델 | 코드 기본값 | 설명 |
|---|---|---|---|
| `CGC_STAGE` | `5` | 6 | 도로 단계입니다. 1 정류장, 2 신호, 3 끼어들기, 4 어린이보호구역, 5 오토바이, 6 불법주정차 순서로 요소가 하나씩 추가됩니다. |
| `OBS_SCAN` | `0` | 0 (`gui_drive_seed.py`만 1) | 1이면 옆 차로 관측을 보강합니다. 6단계 실험에서만 썼습니다. |
| `CGC_THROUGH_VPH` | 설정 안 함 (400) | 400 | 본선 교통량(시간당 차량 수)입니다. |

## 개발 과정
다음과 같은 사고 방식으로 모델을 만들어 나갔습니다.
BC에 PPO를 붙여 전문가의 성능을 넘어보고자 했으나 완주율이 87퍼센트에서 35퍼센트로 떨어졌습니다. 팀원의 실험 기록에서 가치망이 예측한 보상값의 오차가 튄 순간 운전이 무너졌고, 속도 보상 때문에 정류장을 놓쳤고, BC를 따라 하는 힘을 0까지 줄이는 것을 발견했습니다. 
<img width="1915" height="1074" alt="image" src="https://github.com/user-attachments/assets/8b13f045-f0c5-4957-a9a4-27d88e213431" />



첫번째 원인은 정책과 가치망이 신경망 앞부분을 함께 써서, 가치망을 학습할 때 BC로 배운 정책까지 흔들렸습니다. 그래서 정책과 가치망을 완전히 나누고, 정책을 고정한 채 가치망부터 먼저 학습시켰습니다.
그 결과 가치 예측 정확도(explained variance, 1에 가까울수록 정확)가 학습 내내 대부분 0.9 이상을 유지했습니다.
<img width="1919" height="1076" alt="image" src="https://github.com/user-attachments/assets/6ba32ff1-d5c7-412a-a1c9-f33a94de4b73" />



두 번째 원인은 속도 보상입니다. 빨리 달리면 점수를 주니까 왼쪽 차로로 나갔다가 정류장을 놓쳤습니다. 그래서 속도 보상을 없애고 차선은 BC 그대로 고정했습니다. 가속은 BC와 다르게 가속할수록 KL 페널티를 줘서 BC 근처에 묶었습니다. σ 실험 결과,  σ가 0.1을 넘으면 정차 중에도 흔들려 정류장을 놓쳐서 BC 값 0.079를 썼습니다.
<img width="1918" height="1073" alt="image" src="https://github.com/user-attachments/assets/5980e557-2c20-49cb-a2ba-910d996538d5" />



세 번째 원인은 BC를 따라 하는 힘이 0이 된 것입니다. KL 페널티 세기 β를 자동으로 조절했는데, 가운데 식처럼 BC와 너무 가까우면 β를 절반으로 줄입니다. 그대로 두면 0이 될 수 있어서 하한 0.01을 걸었습니다. 오른쪽 위 그래프를 보면 BC와의 차이가 기준보다 작아 β가 계속 줄었고, 아래 그래프처럼 0.01에서 멈춰 끝까지 0이 되지 않았습니다.
<img width="1919" height="1074" alt="image" src="https://github.com/user-attachments/assets/0d5be6e6-c278-4b4c-9c30-9a8f2b8b7d85" />



이 방법을 오토바이가 있는 5단계에 적용했습니다. BC 실패 대부분이 가속과 감속 문제여서, 차선은 그대로 두고 가속만 학습했습니다. 가운데 표는 모델을 고를 때 쓰지 않은 100판에서 BC와 비교한 결과입니다. 충돌 3퍼센트로 기준 5퍼센트를 통과했고, BC와 비슷한 수준을 지켜 무너지지 않았습니다. 다만 오른쪽처럼 남은 실패가 BC와 같은 곳에 몰려 있어, 다음은 DAgger가 필요합니다.
<img width="1919" height="1072" alt="image" src="https://github.com/user-attachments/assets/7bbaa355-635d-498a-b257-e9d0e46c412a" />

## 성능

아래 결과는 모두 결정적 정책과 안전 실드(safety shield) 감시 모드에서 측정했습니다.

- 결정적 정책: 매번 가장 가능성이 높은 행동을 고르는 방식입니다. 같은 시드로 돌리면 항상 같은 결과가 나옵니다.
- 안전 실드 감시 모드(`--shield shadow`): 안전 실드는 위험한 행동을 막는 규칙 기반 안전장치입니다. 감시 모드에서는 실드가 운전에 개입하지 않고, 개입했을 상황만 기록합니다. 따라서 아래 숫자는 정책이 혼자 운전한 결과입니다. 이를 기록한 이유는 수동 개입률(사람이 개입해야 하는 확률)을 기록하고 싶었기 때문입니다. 

| 평가 시드 | 판 수 | 충돌 | 공식 완주 | 완벽 완주 | 장기대기 발생 | 시간초과 | 이동 중 평균 속도 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 개발 시드 900100 ~ 900129 | 30 | 1 (3.3%) | 28 (93.3%) | 26 (86.7%) | 측정 안 함 | 측정 안 함 | 측정 안 함 |
| 독립 시드 900300 ~ 900399 | 100 | 3 (3%) | 87 (87%) | 84 (84%) | 63 (63%) | 7 (7%) | 19.6 km/h |

표의 각 항목은 다음과 같습니다.

- 개발 시드는 모델을 고르는 과정에서 계속 확인한 시드입니다. 독립 시드는 모델을 고른 뒤 처음 돌려 본 시드라서 실제 실력에 더 가깝습니다.
- 공식 완주는 충돌, 빨간불 위반, 차로 오류 없이 목적지에 도착한 판입니다. 정류장 정차를 놓쳤더라도 끝까지 도착하면 포함합니다.
- 완벽 완주는 공식 완주이면서 노선의 정류장 5곳 모두 제대로 정차한 판입니다. 정류장을 하나라도 놓치면 공식 완주에는 들어가지만 완벽 완주에는 들어가지 않습니다.
- 장기대기 발생은 버스가 60초(120스텝) 넘게 연속으로 멈춘 적이 한 번이라도 있는 판입니다. 속도가 0.1 m/s 미만이면 멈춘 것으로 봅니다. 정지선 15 m 안에서 빨간불을 기다리는 시간은 세지 않습니다. 장기대기가 있었더라도 결국 목적지에 도착한 판은 완주에 들어갑니다.
- 시간초과는 제한 시간 1,800초(3,600스텝) 안에 목적지에 도착하지 못한 판입니다. 시간초과 7판은 모두 장기대기가 있었던 판이라, 장기대기 63판 안에 포함됩니다.
- 이동 중 평균 속도는 멈춰 있던 시간을 빼고 계산한 평균 속도입니다.

데이터 수집 4,000판 기준 충돌률은 4.78%입니다. 세 결과 모두 충돌률 5% 미만 기준을 만족합니다.

자세한 내용은 아래 문서에 있습니다.

- [최종 과정 정리](docs/reports/2026-11-26_최종_과정정리.docx): 프로젝트 전체 과정과 결과를 정리한 보고서입니다.
- [5단계 장기대기 검증](docs/reports/2026-11-26_s5_장기대기.md): 독립 100판의 장기대기와 속도를 분석한 보고서입니다.
- [팀 실험 가이드](docs/experiment_protocol.md): 팀이 실험할 때 따른 규칙과 평가 기준입니다.

6단계 실험(불법주정차 구간 추가, `OBS_SCAN=1`)은 최종 결과에 포함되지 않습니다.


