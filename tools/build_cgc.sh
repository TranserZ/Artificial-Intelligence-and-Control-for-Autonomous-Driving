#!/usr/bin/env bash
# 청계천로 도로망 + 시나리오를 처음부터 다시 만든다.
#   1) OSM → SUMO nod/edg   2) netconvert → net.xml   3) 경로·교통·GUI 설정 생성
# 사용: SUMO_HOME=~/src/sumo-1.27.1-mesa tools/build_cgc.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$SUMO_HOME/bin:$PATH"
python tools/osm_to_sumo.py
# --offset.disable-normalization: 좌표 평행이동을 끄지 않으면 GUI 시점·경로 좌표가 어긋난다
netconvert --node-files env/cheonggyecheon/cgc.nod.xml --edge-files env/cheonggyecheon/cgc.edg.xml \
  -o env/cheonggyecheon/cgc.net.xml --no-turnarounds true --junctions.join true --tls.join true \
  --junctions.corner-detail 5 --output.street-names true --offset.disable-normalization true \
  --sidewalks.guess true --sidewalks.guess.max-speed 14 --crossings.guess true --crossings.guess.speed-threshold 14
python tools/make_cgc_scenario.py "$@"
echo "완료: env/cheonggyecheon/cgc.sumocfg"
