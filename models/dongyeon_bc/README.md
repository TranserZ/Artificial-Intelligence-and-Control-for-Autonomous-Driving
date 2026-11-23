# 5·6단계 학생 (강동연)

민기 님 `models/bc_ppo` 파이프라인을 그대로 쓰고, 4단계 학생(`models/bc_ppo/s4_student.pt`, d2ecd99)에서 이어 학습했다.
bc_ppo 파일은 고치지 않았다. 6단계 선생님 보강은 `teacher56.py`가 클래스 메서드를 덮어쓰는 방식이다.

```
s4_student (민기, 4단계 93.3%) ─BC₀→ s5_student (5단계) ─BC₀→ s6_bc0 ─BC(시범 + DAgger 1회분 ×3, lane-weight 1)→ s6_student (6단계)
```

## 결과 (shield shadow = 학생 혼자 운전, 90판)

| 모델 | 단계 | 완주 | 충돌 | 적색 | 시간초과 | 수동개입률 |
|---|---|---|---|---|---|---|
| `s5_student.pt` | 5 (오토바이) | **87.8%** | 6.7% | 3.3% | 2.2% | 0.3% |
| `s6_student.pt` | 6 (불법주정차) | 34.4% | 62.2% | 0% | 3.3% | 1.6% |
| `s6_student_obsfix.pt` (`OBS_SCAN=1` 필요, DAgger 2회) | 6 (불법주정차) | **67.8%** | 22.2% | 2.2% | 7.8% | 0.9% |

평가 판: 개발 900100~900129 (30판) + 학습 범위 60판 (5단계 570000~, 6단계 670000~), 15판씩. 판별 json은 `results/`.

## 학습 방법

- **5단계**: 4단계 선생님(`s4_teacher.pt`, shield on + 정류장·신호·보호구역 규칙, `BCPPO_YELLOW_DECEL=2.4`, 넉넉한 게이트)
  시범 100판(100판 완주) → `s4_student`에서 BC₀ (30 epoch, brake-weight 10, lane-weight 3). 커널 `kaggle/s5v2.py`.
- **6단계 선생님**: 민기 님 선생님 그대로는 6단계 완주 1% (불법주정차 뒤에서 시간초과 99%).
  `teacher56.py`로 두 가지를 더했다.
  1. 회피 차선 명령: 같은 차로 앞 80 m에 불법주정차가 있으면 빈 옆 차로로 → 15%
  2. 막혔을 때 게이트: 주정차 차 뒤에서 3 m/s 미만이면 차선변경 검사를 앞 8 m, 뒤 8 m + 3 s × 다가오는 속도로 → **62%** (충돌 3%, 시간초과 35%)
  남은 시간초과는 대부분 주교 앞(e218924602_4, 교차로 바로 뒤 주정차 + 옆 차로 12~15 m 간격 대열) — 환경 쪽 문제로 보인다.
- **6단계 학생**: 선생님 시범 200판(성공 판만) → `s5_student`에서 BC₀ = 완주 5.6%·충돌 73%.
  진단(`kaggle/d6.py`): 충돌 26/30이 "주정차 차 뒤에 서 있다가 옆 차로 틈 없이 차선변경 → 직진차·오토바이와 측면 충돌".
  DAgger 1회(학생이 운전한 100판에 선생님 정답) → **24.4%** (시간초과 21% → 2%, 충돌은 그대로).
  DAgger 2회째는 11.1%로 떨어졌다 (학생이 일찍 부딪혀 기다리는 장면이 적게 모임).
  DAgger 1회분(학생이 막힌 장면)을 3배로 늘려 s6_bc0에서 다시 BC: lane-weight 3 → 17.8%, **lane-weight 1 → 34.4%** (현재 `s6_student.pt`, 커널 `kaggle/s6w.py`).

## 남은 문제: 옆 차로 관측 사각지대

옆 차로(±1) 관측은 getNeighbors만 쓰고 나란히 겹친 차(gap < 0)를 버린다. 버스가 서 있을 때 차선 직접 스캔과 비교하면
앞차 약 79%, 뒷차 약 41%를 관측이 놓쳤다. 선생님 게이트는 스캔으로 보므로, 학생은 못 보는 차 때문에 선생님이 기다리는 장면을 흉내 낼 수 없다.
보강 실험: `obsfix.py` (`OBS_SCAN=1`)로 옆 차로 관측을 스캔으로 보완하고 시범 200판부터 다시 → s5_student에서 BC₀ 46.7%·충돌 45.6%, DAgger 1회 BC1 58.9%·충돌 24.4% (`kaggle/s6o.py`) → DAgger 2회 BC2 **67.8%·충돌 22.2%** (`s6_student_obsfix.pt`), 3회 64.4% (`kaggle/s6o2.py`).
이 모델은 보강된 관측으로만 제대로 달린다 (`OBS_SCAN=1 python models/dongyeon_bc/teacher56.py eval ...`). 공식 평가에 쓰려면 env의 `_neighbor` 수정이 필요하고, 그건 팀 결정.

## 재현

```
CGC_STAGE=6 TEACHER_AVOID=1 BCPPO_YELLOW_DECEL=2.4 \
BCPPO_LC_GATE='{"rear_tau":3.0,"approach_horizon":4.0,"min_rear_gap":8.0}' \
python models/dongyeon_bc/teacher56.py collect|dagger|bc|eval ...   # bcppo.py 모드 그대로
CGC_STAGE=6 python models/bc_ppo/bcppo.py eval --ckpt models/dongyeon_bc/s6_student.pt --shield shadow --episodes 15 --seed-base 900100
```
Kaggle 커널 코드는 `kaggle/` (s5v2: 5단계, t6v3: 6단계 선생님 시범, s6: 6단계 BC₀, d6: 충돌 진단, s6d: DAgger).
