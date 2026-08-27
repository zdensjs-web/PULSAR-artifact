#!/usr/bin/env bash
set -euo pipefail

width="${1:-256}"
exchange_rate="${2:-3}"
rate_scale="${3:-3}"
threads="${4:-${OMP_NUM_THREADS:-12}}"
repeats="${5:-1}"

case "${width}" in
  16|32|64|128|256) ;;
  *) echo "word width must be 16, 32, 64, 128, or 256" >&2; exit 2 ;;
esac

if ! [[ "${threads}" =~ ^[1-9][0-9]*$ ]]; then
  echo "threads must be a positive integer" >&2
  exit 2
fi
if ! [[ "${repeats}" =~ ^[1-9][0-9]*$ ]]; then
  echo "repeats must be a positive integer" >&2
  exit 2
fi

binary="./build/bin/examples/pke/benchmark-pulsar-vault"
if [[ ! -x "${binary}" ]]; then
  echo "missing ${binary}; reconfigure CMake and build benchmark-pulsar-vault first" >&2
  exit 1
fi

log="pulsar_vault_${width}bit_t${threads}.log"
OMP_NUM_THREADS="${threads}" \
OMP_DYNAMIC=FALSE \
OMP_THREAD_LIMIT="${threads}" \
OMP_PROC_BIND=close \
TCMALLOC_RELEASE_RATE="${TCMALLOC_RELEASE_RATE:-10}" \
TCMALLOC_MAX_TOTAL_THREAD_CACHE_BYTES="${TCMALLOC_MAX_TOTAL_THREAD_CACHE_BYTES:-268435456}" \
"${binary}" "${width}" "${exchange_rate}" "${rate_scale}" "${repeats}" \
  2>&1 | tee "${log}"

echo "log: ${log}"
