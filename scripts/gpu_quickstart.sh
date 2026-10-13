#!/usr/bin/env bash
# UniAi on a real NVIDIA GPU (e.g. a free Google Colab T4):
#   !git clone -b claude/sleepy-turing-s23xsu https://github.com/SoloB3n/UniAi && cd UniAi && bash scripts/gpu_quickstart.sh
set -euo pipefail
cd "$(dirname "$0")/.."

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
pip install -q numpy

echo "== build: CPU engine, GPU engine, emulated GPU engine"
make -s uniai_c gpu gpu-emu

echo "== correctness: real GPU kernels vs Python (loss, every gradient, AdamW step)"
python3 tests/test_c_engine.py

echo "== data"
[ -f data/shakespeare.txt ] || ./data/download.sh
[ -f data/shakespeare_train.bin ] || python3 -m uniai prepare --data data/shakespeare.txt --out data/shakespeare

mkdir -p runs
echo "== same model as the CPU benchmark (0.94M params, 3000 steps)"
./uniai_gpu train -i data/shakespeare -o runs/gpu_small.bin -n 3000 -e 1000

echo "== a 10x larger model the CPU could not train in reasonable time"
./uniai_gpu train -i data/shakespeare -o runs/gpu_large.bin \
    -L 6 -H 6 -C 384 -T 256 -B 32 -n 5000 -e 1000 -l 6e-4 -p 0.1

python3 -m uniai sample --ckpt runs/gpu_large.bin \
    --tokenizer data/shakespeare_tokenizer.json --prompt "ROMEO:" --tokens 300
