# 5단계 데이터 수집 2번 (강동연)

시드 5,010,000 ~ 5,011,999, 2,000판, 2,931,022행.

## 조건

| 항목 | 값 |
|---|---|
| 모델 | `results/klppo1_s5_s0/step250k.pt` (KL-PPO, 5단계) |
| 코드 | 브랜치 `model/kl-ppo` 58ba5a8 + 이 브랜치의 `collect_pomdp_data.py` 로더 수정 |
| 환경 변수 | `CGC_STAGE=5`, `OBS_SCAN=0` |
| SUMO | 1.27.1 |
| 행동 | 결정적 (스크립트 기본값) |
| 실행 | macOS, Python 3.10, torch 2.14.1, 워커 8개 |

## 실행 명령

250판씩 8번 나눠 돌렸습니다. `k = 0..7`, 시드 시작은 `5010000 + 250*k` 입니다.

```bash
export CGC_STAGE=5 OBS_SCAN=0
python collect_pomdp_data.py --model results/klppo1_s5_s0/step250k.pt \
  --seed $((5010000 + 250*k)) --episodes 250 --workers 8 \
  --name s5_data_p2_c$k --out-dir data_dongyeon/s5_data_p2
```

## 파일

| 묶음 | 시드 | 행 수 |
|---|---|---|
| `s5_data_p2_c0` | 5010000 ~ 5010249 | 354,107 |
| `s5_data_p2_c1` | 5010250 ~ 5010499 | 370,392 |
| `s5_data_p2_c2` | 5010500 ~ 5010749 | 367,711 |
| `s5_data_p2_c3` | 5010750 ~ 5010999 | 373,263 |
| `s5_data_p2_c4` | 5011000 ~ 5011249 | 370,422 |
| `s5_data_p2_c5` | 5011250 ~ 5011499 | 361,684 |
| `s5_data_p2_c6` | 5011500 ~ 5011749 | 368,637 |
| `s5_data_p2_c7` | 5011750 ~ 5011999 | 364,806 |

묶음마다 `.table.csv.gz` (167열) 와 `.npz` 가 있습니다. `collect_progress.log` 는 묶음별 시작·종료 시각입니다.

## 합칠 때

- `episode` 열은 묶음마다 0 ~ 249로 다시 시작합니다. 판을 구분할 때는 `seed` 열을 쓰면 됩니다.
- `npz` 의 `detected_offsets` 도 묶음 안에서만 맞는 위치입니다. 이어 붙일 때는 앞 묶음의 `detected_xy` 길이만큼 더해야 합니다.
- 원본 `jsonl.gz` 는 용량 때문에 첫 묶음(c0, 1.5 GB)만 로컬에 남겼고 저장소에는 없습니다.

## 로더 수정

`collect_pomdp_data.py` 가 KL-PPO 체크포인트 형식을 읽지 못해 `--policy ppo` 일 때 `algorithms/kl_ppo.py` 의 `load_eval_policy` 로 읽도록 바꿨습니다. 그 밖의 수집 코드는 그대로입니다.

## 확인한 것

- 묶음마다 시드 250개가 빠짐없이 한 번씩 들어 있고, 전체 2,000개가 겹치지 않습니다.
- 묶음마다 표와 `npz` 의 행 수가 같습니다.
- 시드 5000000 한 판은 1,021행으로 팀 비교값과 같고, 워커 1개와 8개의 결과가 같습니다.
