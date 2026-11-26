#!/bin/bash
# eval5.sh <first> <last>  : 5단계 학생을 테스트 판으로 화면 없이 돌린다
cd "$(dirname "$0")"
export SUMO_HOME=~/src/sumo-1.27.1-mesa PYTHONPATH=~/src/sumo-1.27.1-mesa/tools
for s in $(seq $1 $2); do
  [ -f out/h5_$s/log.json ] && continue
  CGC_STAGE=5 ~/miniforge3/envs/sumo-rl/bin/python render_run.py --repo ~/work/cgc-s56 --ckpt models/dongyeon_bc/s5_student.pt --seed $s --out out/h5_$s --nogui 2>&1 | tail -1
done
