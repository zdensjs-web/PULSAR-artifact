#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${root}/run_pulsar_cpu_operator.sh" mixed "${1:-128}" "${2:-${OMP_NUM_THREADS:-12}}" "${3:-1}"
