#!/usr/bin/env bash
set -euo pipefail

rounds="${1:-64}"
bit_clean_threshold="${2:-0.01}"
threads="${3:-12}"

if ! [[ "${rounds}" =~ ^[1-9][0-9]*$ ]] ||
   (( rounds > 64 )); then
  echo "rounds must be an integer from 1 to 64" >&2
  exit 2
fi
if ! [[ "${threads}" =~ ^[1-9][0-9]*$ ]]; then
  echo "threads must be a positive integer" >&2
  exit 2
fi

threshold_tag="${bit_clean_threshold//./p}"
log="pulsar_sha256_q33_scale43_q0_52_r${rounds}_clean_${threshold_tag}_t${threads}.log"

OMP_NUM_THREADS="${threads}" \
OMP_DYNAMIC=FALSE \
OMP_THREAD_LIMIT="${threads}" \
OMP_PROC_BIND="${OMP_PROC_BIND:-close}" \
TCMALLOC_RELEASE_RATE="${TCMALLOC_RELEASE_RATE:-10}" \
TCMALLOC_MAX_TOTAL_THREAD_CACHE_BYTES="${TCMALLOC_MAX_TOTAL_THREAD_CACHE_BYTES:-268435456}" \
./build/bin/examples/pke/benchmark-pulsar-add-chain \
  32 "${rounds}" 0 0 sha256 "${bit_clean_threshold}" 2>&1 | tee "${log}"

echo "log: ${log}"
