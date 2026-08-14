#!/usr/bin/env bash
# Fetch the public benchmarks. Nothing here is redistributed in this repo.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/raw

echo "LoCoMo (Snap Inc. + UNC, arXiv 2402.17753) ..."
curl -sSL -o data/raw/locomo10.json \
  "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"

echo "LongMemEval, CLEANED variant (Wu et al., ICLR 2025, arXiv 2410.10813) ..."
echo "  note: the widely-cited xiaowu0162/longmemeval was DEPRECATED 2025-09-19."
curl -sSL -o data/raw/longmemeval_s_cleaned.json \
  "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json"
curl -sSL -o data/raw/longmemeval_oracle.json \
  "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_oracle.json"

echo "done. sizes:"
ls -la data/raw/
