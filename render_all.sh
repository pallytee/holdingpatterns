#!/bin/bash
set -e

FPS=30
FRAMES=300

render_mode () {
  MODE=$1

  echo "🧹 clearing $MODE frames"
  rm -rf frames_png_$MODE frames_svg_$MODE

  echo "🎨 rendering $MODE"
  python3 render.py --mode $MODE --frames $FRAMES

  echo "🎞 exporting $MODE.mp4"
  ffmpeg -y -framerate $FPS \
    -i frames_png_$MODE/${MODE}_frame_%04d.png \
    -c:v libx264 -pix_fmt yuv420p -movflags +faststart ${MODE}.mp4
}

render_mode regulator
render_mode saturator
render_mode fader

echo "✅ all films rendered"

