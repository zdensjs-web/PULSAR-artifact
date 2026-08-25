#!/usr/bin/env bash
set -euo pipefail

width="${1:-128}"
rounds="${2:-10}"
initial_refresh="${3:-1}"
final_refresh="${4:-0}"
mode="${5:-scheduler}"
case "${width}" in
  16|32|64|128|256) ;;
  *)
    echo "error: word_bits must be one of 16, 32, 64, 128, or 256" >&2
    exit 2
    ;;
esac
if ! [[ "${rounds}" =~ ^[1-9][0-9]*$ ]]; then
  echo "error: rounds must be a positive integer" >&2
  exit 2
fi
mode_tag="${mode//-/_}"
log="pulsar_add_${mode_tag}_q33_scale43_q0_52_${width}bit_r${rounds}_i${initial_refresh}_f${final_refresh}.log"

OMP_NUM_THREADS="${OMP_NUM_THREADS:-12}" \
OMP_DYNAMIC=FALSE \
OMP_THREAD_LIMIT="${OMP_THREAD_LIMIT:-12}" \
OMP_PROC_BIND="${OMP_PROC_BIND:-close}" \
TCMALLOC_RELEASE_RATE="${TCMALLOC_RELEASE_RATE:-10}" \
TCMALLOC_MAX_TOTAL_THREAD_CACHE_BYTES="${TCMALLOC_MAX_TOTAL_THREAD_CACHE_BYTES:-268435456}" \
./build/bin/examples/pke/benchmark-pulsar-add-chain \
  "${width}" "${rounds}" "${initial_refresh}" "${final_refresh}" "${mode}" \
  2>&1 | tee "${log}"

echo "log: ${log}"
