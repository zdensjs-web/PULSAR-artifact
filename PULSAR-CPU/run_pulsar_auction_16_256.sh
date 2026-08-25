#!/usr/bin/env bash
set -euo pipefail

bit_clean_threshold="${1:-0.01}"

for width in 16 32 64 128 256; do
  echo "===== PULSAR auction: ${width} bits ====="
  ./run_pulsar_auction.sh "${width}" "${bit_clean_threshold}"
done
