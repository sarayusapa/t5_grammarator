#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

rm -f results.jsonl

for model in fullft lora qlora; do
  python3 generate.py --model "$model" --benchmark jfleg
  python3 score_jfleg.py --model "$model"

  python3 generate.py --model "$model" --benchmark bea_dev
  python3 score_bea.py --model "$model"
done

echo "=== results.jsonl ==="
cat results.jsonl
