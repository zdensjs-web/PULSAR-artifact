#!/usr/bin/env bash
set -euo pipefail

width="${1:-128}"
bit_clean_threshold="${2:-0.01}"

case "${width}" in
  16|32|64|128|256) ;;
  *) echo "word_bits must be one of 16, 32, 64, 128, or 256" >&2; exit 2 ;;
esac

threshold_tag="${bit_clean_threshold//./p}"
log="pulsar_auction_q33_scale43_q0_52_${width}bit_clean_${threshold_tag}.log"

OMP_NUM_THREADS="${OMP_NUM_THREADS:-12}" \
OMP_DYNAMIC=FALSE \
OMP_THREAD_LIMIT="${OMP_THREAD_LIMIT:-12}" \
OMP_PROC_BIND="${OMP_PROC_BIND:-close}" \
TCMALLOC_RELEASE_RATE="${TCMALLOC_RELEASE_RATE:-10}" \
TCMALLOC_MAX_TOTAL_THREAD_CACHE_BYTES="${TCMALLOC_MAX_TOTAL_THREAD_CACHE_BYTES:-268435456}" \
./build/bin/examples/pke/benchmark-pulsar-add-chain \
  "${width}" 1 0 0 auction "${bit_clean_threshold}" 2>&1 | tee "${log}"

echo "log: ${log}"
