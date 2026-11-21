# Artificial Intelligence and Control for Autonomous Driving

> **자율주행 인공지능 및 제어 — Course Project Materials**

수업 프로젝트에 필요한 자료를 순서대로 정리합니다.

## 🗂 저장소 폴더 지도

| 폴더 | 무엇을 두나 |
|---|---|
| `train.py` · `test.py` · `train_bc.py` · `collect_pomdp_data.py` · `view_road.py` | 실행 파일 (학습 · 재생 · 모방학습 · 데이터 수집 · 도로 보기) |
| `env/` | 학습 환경 (도로 설정, 단계별 맵, SUMO 환경). 청계천로 도로는 학습 때 `env/sumo/`에 새로 만든다 |
| `algorithms/` · `utils/` | 학습 알고리즘(PPO·BC)과 공용 도구 (버퍼, 평가, 로그) |
| `config/` | 버스 노선·정류장 (`lines.yaml`) |
| `tools/` | 도로·시나리오 생성 도구. `tools/crossroad_data_collect/`: T-DATA 신호 수집기 (인증키 파일은 git 제외) |
| `scripts/` | 평가·검증 스크립트 |
| `data/` | 자료. `road_data/`: 도로 실측(신호·차로·정류장 좌표 등) · `osm/`: 지도 원본 · `huiyoung/`: 수집 궤적·평가 로그 |
| `docs/` | 설명과 기록. 환경 정리, 데이터 컬럼, `reports/` 날짜별 보고서 |
| `rules/` | 지켜야 할 것. `course_instruction/`: 수업 과제 안내 · `team/`: 팀 실험 규칙 · `setup/`: 설치·실행 환경 |
| `results/` | 학습 결과가 쌓이는 곳 (`train.py` 출력) |
| `backup/` | 지금은 안 쓰지만 보관하는 것: 옛 학습 결과, 옛 모델(`model_huiyoung`), `train.py` 자동 백업, 옛 도로 파일 |

새 파일은 위 규칙에 맞는 폴더에 둔다. 개인 이름 폴더는 최상위에 만들지 않는다.

---

## 📚 Project Materials

### 개인별 진행

### 0. Project Introduction

- [프로젝트 소개 및 전체 진행 방향](./rules/course_instruction/0_Project_Introduction.md)

### 1. SUMO Installation

- [Windows — SUMO 설치 및 환경 설정](./rules/course_instruction/1_SUMO_Windows_Setup.md)
- [macOS (Apple Silicon) — SUMO 설치 및 환경 설정](./rules/course_instruction/1_SUMO_macOS_AppleSilicon_Setup.md)

### 2. Road Environment Construction

- [도로 및 모델 POMDP 구조 이해](./rules/course_instruction/2_Road_Environment_Construction.md)

---

### 팀별 진행 (1-2명으로 구성, 자율 building)

### 3. Model Training — Behavior Cloning

- [모방학습 기반 자율주행 의사결정 모델 학습](./rules/course_instruction/3_Model_Training_BC.md)

### 팀별 진행 (자율 building)

### 4. Model Training — Reinforcement Learning

- [강화학습 기반 자율주행 의사결정 모델 학습](./rules/course_instruction/4_Model_Training_RL.md)

---
### 팀별 진행 (3-5명으로 구성, 자율 building)

### 5. Project Preparation

- [프로젝트 진행을 위한 코드 소개](./rules/course_instruction/5_Project_Preparation.md)

### 이후. 팀별 프로젝트 진행

#### 평가 기준

- [프로젝트 평가 기준 소개](./rules/course_instruction/6_Project_Evaluation.md)
- [동료 평가 표](https://docs.google.com/document/d/1czmE5ofrKckau2rMwK4-wXax-f5QrUia/edit?usp=sharing&ouid=107893289049883287038&rtpof=true&sd=true)

#### 프로젝트 컨설팅

- [팀별 컨설팅 진행 소개](./rules/course_instruction/7_Project_Consulting.md)
---
> 수업 자료는 `rules/course_instruction/` 폴더에서도 확인 가능합니다.
