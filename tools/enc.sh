#!/bin/bash
# enc.sh src start n fps out [extra-filter]
P=~/work/cgc-page/site/media
ffmpeg -y -loglevel error -framerate "$4" -start_number "$2" -i "out/$1/f_%05d.png" -f lavfi -i "color=c=0xE3E8EE:size=1280x720:rate=$4" \
  -filter_complex "[0]colorkey=0x338033:0.03:0.0[fg];[1][fg]overlay=shortest=1,format=yuv420p$6" -frames:v "$3" \
  -c:v libx264 -crf 27 -preset slow -movflags +faststart -an "$P/$5.mp4"
