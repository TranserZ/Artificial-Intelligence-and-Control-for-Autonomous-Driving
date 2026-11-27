# 청계로드 — 청계천로 자율주행 버스 (SUMO + KL-PPO)

청계천 자율주행 버스(청계A01) 노선 2.1 km를 SUMO로 재현하고, 모방학습(BC) 모델을 KL-PPO로 미세조정해 주행하는 프로젝트입니다.

- GitHub: https://github.com/TranserZ/Artificial-Intelligence-and-Control-for-Autonomous-Driving
- Hugging Face 데이터셋: TODO (링크 추가 예정)
- 프로젝트 페이지: TODO (링크 추가 예정)

## 최종 모델

| 항목 | 값 |
|---|---|
| 모델 파일 | `results/klppo1_s5_s0/step250k.pt` (학습 설정: `results/klppo1_s5_s0/config.json`) |
| 방법 | 5단계 BC 학생(`results/s5_student/s5_student.pt`)에서 출발해 가속만 KL-PPO로 25만 스텝 미세조정 |
| 환경 | **5단계(`CGC_STAGE=5`), `OBS_SCAN=0`에서 학습·평가함. 이 설정 없이 돌리면 다른 환경이 됨** (코드 기본값은 6단계) |
| 팀 완주율 목표 | **85% 이상** (초기 목표 95%, 중간발표 이후 최종 기준 85%로 조정) |

성능 (shield 감시 모드, 결정적 정책):

| 시드 | 충돌 | 공식 완주 | 완벽 완주 | 장기대기 또는 시간초과 | 이동 중 평균 속도 |
|---|---:|---:|---:|---:|---:|
| 개발 900100–900129 (30판) | 1/30 (3.3%) | 28/30 (93.3%) | 26/30 (86.7%) | 미측정 | 미측정 |
| 독립 900300–900399 (100판) | 3/100 (3%) | 87/100 (87%) | 84/100 (84%) | 63/100 (63%) | 19.6 km/h |

자세한 내용: [최종 과정 정리](docs/reports/2026-11-26_최종_과정정리.docx), [5단계 장기대기 검증](docs/reports/2026-11-26_s5_장기대기.md), [팀 실험 가이드](docs/experiment_protocol.md).
6단계(불법주정차, `OBS_SCAN=1`) 실험은 최종에 포함되지 않습니다.

## 실행 환경

- Windows 10/11 (macOS·Linux도 SUMO 설치 경로만 다름)
- Python 3.13
- PyTorch 2.15 (CPU로 충분), 나머지는 `requirements.txt`
- SUMO **1.27.1** — https://sumo.dlr.de/docs/Downloads.php 에서 설치
  - `SUMO_HOME`을 SUMO 설치 폴더로 설정 (Windows 인스톨러에서 "Set SUMO_HOME" 체크 시 자동)
  - `PYTHONPATH`에 `SUMO_HOME`의 `tools` 폴더를 추가 (traci·sumolib을 SUMO 설치본에서 가져옴)

PowerShell (현재 창에만 적용. 영구 적용은 시스템 환경 변수에 같은 값을 등록):

```powershell
$env:SUMO_HOME = "C:\Program Files (x86)\Eclipse\Sumo"
$env:PYTHONPATH = "$env:SUMO_HOME\tools"
sumo --version   # Eclipse SUMO sumo 1.27.1
```

bash (macOS·Linux):

```bash
export SUMO_HOME=/usr/share/sumo        # 설치 경로에 맞게
export PYTHONPATH="$SUMO_HOME/tools"
```

## 설치

```powershell
git clone https://github.com/TranserZ/Artificial-Intelligence-and-Control-for-Autonomous-Driving.git
cd Artificial-Intelligence-and-Control-for-Autonomous-Driving
py -3.13 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\Activate.ps1      # 이후 python = 이 가상환경
```

bash: `python3.13 -m venv .venv && .venv/bin/python -m pip install -r requirements.txt && source .venv/bin/activate`

아래 명령은 모두 **저장소 루트**에서 실행합니다. 도로망은 실행할 때마다 `data/osm`·`data/road_data`·`config/lines.yaml`로 새로 만들어집니다.

## 폴더 구조

```
environment/       SUMO 환경 (Gymnasium): 도로·신호·정류장 설정(road_config.py), MDP(mdp_config.py),
                   도로 생성(road_builder.py), 단계별 맵(curriculum.py), 학교 표지 그림, 네트워크 파일
training/          학습: train.py (PPO / KL-PPO), train_bc.py (모방학습)
evaluation/        평가: drive_parallel.py (병렬 평가), gui_drive_seed.py (SUMO-GUI 재생), test.py (단일 모델 재생)
data_collection/   collect_pomdp_data.py (POMDP 주행 데이터 수집, 컬럼 설명은 docs/data_columns.md)
algorithms/        PPO, KL-PPO, BC 정책 (학습·평가·수집이 함께 씀)
utils/             신경망, 버퍼, 로거, 평가 지표
tools/             OSM → SUMO 변환, 시나리오·신호·노면표시 생성 (environment가 실행 중에 사용)
config/            버스 노선·정류장 (lines.yaml)
data/              OSM 지도와 실측 도로 자료 (차로 수, 신호 시간, 보호구역 등)
docs/              데이터 컬럼 문서, 팀 실험 가이드, 최종 보고서
results/           klppo1_s5_s0/ (최종 모델), s5_student/ (KL-PPO 기준 BC 학생), exp-ppo-stage1/ (보상 확인용 설정)
```

## 평가

독립 100판 (시간이 오래 걸림, `--workers`는 CPU 코어 수에 맞게):

```powershell
$env:CGC_STAGE="5"; $env:OBS_SCAN="0"; python evaluation/drive_parallel.py --model results/klppo1_s5_s0/step250k.pt --seeds 900300-900399 --shield shadow --workers 4 --json-out results/local_evaluation/stage5.json
```

```bash
CGC_STAGE=5 OBS_SCAN=0 python evaluation/drive_parallel.py --model results/klppo1_s5_s0/step250k.pt --seeds 900300-900399 --shield shadow --workers 4 --json-out results/local_evaluation/stage5.json
```

1판만 빠르게 확인:

```powershell
$env:CGC_STAGE="5"; $env:OBS_SCAN="0"; python evaluation/drive_parallel.py --model results/klppo1_s5_s0/step250k.pt --seeds 900300 --shield shadow --workers 1
```

```bash
CGC_STAGE=5 OBS_SCAN=0 python evaluation/drive_parallel.py --model results/klppo1_s5_s0/step250k.pt --seeds 900300 --shield shadow --workers 1
```

SUMO-GUI로 한 판 보기:

```powershell
$env:CGC_STAGE="5"; $env:OBS_SCAN="0"; python evaluation/gui_drive_seed.py --stage 5 --seed 900377 --checkpoint results/klppo1_s5_s0/step250k.pt --delay 80
```

```bash
CGC_STAGE=5 OBS_SCAN=0 python evaluation/gui_drive_seed.py --stage 5 --seed 900377 --checkpoint results/klppo1_s5_s0/step250k.pt --delay 80
```

## 데이터 수집

최종 모델로 주행하며 관측·행동·상태를 저장합니다 (`.jsonl.gz`, `.npz`, 컬럼 표 `.table.csv.gz`). 기본 출력 폴더는 `data/`.

```powershell
$env:CGC_STAGE="5"; $env:OBS_SCAN="0"; python data_collection/collect_pomdp_data.py --model results/klppo1_s5_s0/step250k.pt --seed 5000000 --episodes 1 --workers 1 --name s5_check
```

```bash
CGC_STAGE=5 OBS_SCAN=0 python data_collection/collect_pomdp_data.py --model results/klppo1_s5_s0/step250k.pt --seed 5000000 --episodes 1 --workers 1 --name s5_check
```

위 명령(시드 5000000, 1판)은 1,021행·167열 컬럼 표를 만들고, 도착(`event.arrived` = 1)하며 마지막 `state.ego.odometer` = 2136.31 m 입니다. 여러 판은 `--episodes`와 `--workers`를 늘립니다.

## 학습

최종 모델의 KL-PPO 학습 (5단계 BC 학생에서 출발, 30만 스텝, 2만 5천 스텝마다 저장 → `step250k.pt`를 최종으로 선택):

```powershell
$env:CGC_STAGE="5"; $env:OBS_SCAN="0"; python training/train.py --ref results/s5_student/s5_student.pt --explore-std 0.07772492733277317 --seed 0 --run-name klppo1_s5_s0_repro
```

```bash
CGC_STAGE=5 OBS_SCAN=0 python training/train.py --ref results/s5_student/s5_student.pt --explore-std 0.07772492733277317 --seed 0 --run-name klppo1_s5_s0_repro
```

시작할 때 보상이 `results/exp-ppo-stage1/config.json`과 같은지(40키), `CGC_STAGE`와 도로 단계가 같은지 검사하고 다르면 멈춥니다. 결과는 `results/<run-name>/`에 저장됩니다.

모방학습(BC) 학생 학습은 수집한 데이터(`.npz`/`.jsonl.gz`)로 합니다. 옵션은 `--help`로 확인합니다.

```powershell
$env:CGC_STAGE="5"; $env:OBS_SCAN="0"; python training/train_bc.py --help
```

```bash
CGC_STAGE=5 OBS_SCAN=0 python training/train_bc.py --help
```

## 환경변수

| 변수 | 최종 모델 | 코드 기본값 | 설명 |
|---|---|---|---|
| `CGC_STAGE` | `5` | 6 | 도로 단계 (1 정류장 … 5 오토바이, 6 불법주정차) |
| `OBS_SCAN` | `0` | 0 | 1이면 옆 차로 관측 보강 (6단계 실험용) |
| `CGC_THROUGH_VPH` | 생략 (400) | 400 | 본선 교통량 (대/시간) |
