#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -r requirements.txt

.venv/bin/python -m pytest tests -q
.venv/bin/python bench/calibrate.py \
  --model Qwen/Qwen2.5-0.5B \
  --text data/pride_and_prejudice.txt \
  --split 12 \
  --out results/calibration_qwen25_0.5b.json
.venv/bin/python bench/plot_calibration.py
