#!/usr/bin/env bash
# Reproduce the hybrid prototype: generated plate -> paper pieces -> animated, narrated clip + comparison.
# Needs the storyvid setup from the top-level README (requirements + scripts/fetch_models.py) and ffmpeg.
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="$(cd .. && pwd)${PYTHONPATH:+:$PYTHONPATH}"
python3 extract_layers.py                       # -> masks.npy, masks_overlay.jpg
python3 make_layers.py                          # -> layers/*.png, layers_sheet.jpg
python3 -m storyvid.pipeline render hybrid_jingwei.json --out out   # -> out/hybrid_jingwei.mp4
python3 compare.py                              # -> out/compare_paperwing_vs_hybrid.mp4
