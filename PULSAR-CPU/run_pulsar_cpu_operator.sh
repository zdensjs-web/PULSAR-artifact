#!/usr/bin/env bash
set -euo pipefail

operation="${1:-add}"
width="${2:-128}"
threads="${3:-${OMP_NUM_THREADS:-12}}"
output_refresh="${4:-1}"

case "${operation}" in
  add|gt|eq|xor) ;;
  *) echo "operation must be add, gt, eq, or xor" >&2; exit 2 ;;
esac

case "${width}" in
  16|32|64|128|256) ;;
  *) echo "word width must be 16, 32, 64, 128, or 256" >&2; exit 2 ;;
esac

if ! [[ "${threads}" =~ ^[1-9][0-9]*$ ]]; then
  echo "threads must be a positive integer" >&2
  exit 2
fi
case "${output_refresh}" in
  0|1) ;;
  *) echo "output_refresh must be 0 or 1" >&2; exit 2 ;;
esac

binary="./build/bin/examples/pke/benchmark-pulsar-single-round"
if [[ ! -x "${binary}" ]]; then
  echo "missing ${binary}; configure and build benchmark-pulsar-single-round first" >&2
  exit 1
fi

log="pulsar_cpu_${operation}_${width}bit_t${threads}.log"
OMP_NUM_THREADS="${threads}" \
OMP_DYNAMIC=FALSE \
OMP_THREAD_LIMIT="${threads}" \
OMP_PROC_BIND=close \
TCMALLOC_RELEASE_RATE="${TCMALLOC_RELEASE_RATE:-10}" \
"${binary}" "${operation}" "${width}" "${output_refresh}" 2>&1 | tee "${log}"

echo "log: ${log}"
