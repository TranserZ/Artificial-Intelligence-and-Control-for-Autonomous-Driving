#!/bin/bash
# 사용: bash run_person.sh <담당번호 1~4> <워커 수>
# 담당 n: 시드 5000000+(n-1)*10000 부터 2000판, episode (n-1)*2000 부터. 250판씩 끊어 저장
# (도중에 멈춰도 끝난 묶음은 남고, 다시 실행하면 이미 있는 묶음은 건너뛴다).
source ~/venvs/sumo-rl/bin/activate
cd "$(dirname "$0")"
export CGC_STAGE=5 OBS_SCAN=0 STOP_FIX=0 PLACE_FIX=0 CGC_THROUGH_VPH=400
N=$1; WK=${2:-6}; OUT=/home/min-gi/Desktop/2026_2/pomdp_s5_klppo
for b in 0 1 2 3 4 5 6 7; do
  SEED=$((5000000 + (N-1)*10000 + b*250)); EP=$(((N-1)*2000 + b*250))
  NAME=s5_klppo_p${N}_b${b}
  [ -f "$OUT/$NAME.npz" ] && { echo "skip $NAME"; continue; }
  echo "=== $NAME seed $SEED~$((SEED+249)) episode $EP~$((EP+249)) $(date +%T)"
  python collect_pomdp_data.py --model results/klppo1_s5_s0/step250k.pt --episodes 250 \
    --workers $WK --seed $SEED --episode-offset $EP --out-dir $OUT --name $NAME 2>&1 \
    | grep -vE "^\[신호\]|Retrying"
  [ -f "$OUT/$NAME.npz" ] || echo "FAILED $NAME"
done
echo "ALL DONE p$N $(date +%T)"
