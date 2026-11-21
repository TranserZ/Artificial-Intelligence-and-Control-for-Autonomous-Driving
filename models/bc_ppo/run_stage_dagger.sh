#!/bin/bash
# 한 단계 도로의 학생 만들기: 시범 수집 → BC₀ → (DAgger 수집 → BC 재학습) × ROUNDS, 매번 개발 30판 평가.
# 사용: models/bc_ppo/run_stage_dagger.sh <단계> <선생님.pt> <시작 학생.pt> [ROUNDS=3] [BRAKE=10]
set -u
STAGE=$1; TEACHER=$2; INIT=$3; ROUNDS=${4:-3}; BRAKE=${5:-10}
cd "$(dirname "$0")/../.."
source ~/venvs/sumo-rl/bin/activate
export CGC_STAGE=$STAGE
P=models/bc_ppo/bcppo.py; D=models/bc_ppo/data; R=models/bc_ppo/runs; s=s$STAGE
base=$((STAGE * 100000))      # 단계마다 다른 판 시드 (학습 판 범위)
summary=$R/${s}_summary.txt; : > $summary

par4() {  # par4 <모드> <시드 시작> <출력 접두사> [추가 인자...] : 25판 × 4 프로세스
  local mode=$1 sb=$2 out=$3; shift 3
  for k in 0 1 2 3; do
    python $P $mode --episodes 25 --seed-base $((sb + 25 * k)) --data ${out}_part$k.npz "$@" > ${out}_part$k.log 2>&1 &
  done; wait
  grep -h "저장" ${out}_part*.log | sed "s|$D/||" | cut -c1-150 | tee -a $summary
}
evaluate() {  # evaluate <run 폴더>
  python $P eval --ckpt $1/bc.pt --shield shadow --episodes 30 --json $1/eval_dev30_shadow.json > $1/eval.log 2>&1
  echo "$(basename $1): $(tail -1 $1/eval.log | sed 's|^.*\] ||')" | tee -a $summary
}

par4 collect $base $D/${s}_part --ckpt $TEACHER
DATA="$D/${s}_part0.npz $D/${s}_part1.npz $D/${s}_part2.npz $D/${s}_part3.npz"
BCPPO_THREADS=8 python $P bc --data $DATA --init $INIT --out $R/${s}_bc0 --epochs 30 --brake-weight $BRAKE > $R/${s}_bc0.log 2>&1
evaluate $R/${s}_bc0
prev=$R/${s}_bc0
for r in $(seq 1 $ROUNDS); do
  par4 dagger $((base + 10000 * r)) $D/${s}_dagger$r --ckpt $prev/bc.pt --teacher $TEACHER
  DATA="$DATA $D/${s}_dagger${r}_part0.npz $D/${s}_dagger${r}_part1.npz $D/${s}_dagger${r}_part2.npz $D/${s}_dagger${r}_part3.npz"
  BCPPO_THREADS=8 python $P bc --data $DATA --init $prev/bc.pt --out $R/${s}_bc$r --epochs 30 --brake-weight $BRAKE > $R/${s}_bc$r.log 2>&1
  evaluate $R/${s}_bc$r
  prev=$R/${s}_bc$r
done
echo "끝" | tee -a $summary
