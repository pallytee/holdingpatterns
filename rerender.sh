#!/bin/bash
set -e

OUT="regulator.mp4"
FPS=30

echo "🧹 clearing old outputs"
rm -rf frames_svg frames_png
rm -f "$OUT"

mkdir -p frames_svg frames_png

echo "🎨 rendering frames (SVG → PNG)"
python3 render.py

echo "🎞 encoding video"
ffmpeg -y -framerate "$FPS" -i frames_png/frame_%04d.png \
  -c:v libx264 -pix_fmt yuv420p -movflags +faststart \
  "$OUT"

echo "✅ done → $OUT"

