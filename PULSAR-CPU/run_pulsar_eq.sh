#!/usr/bin/env bash
set -euo pipefail
exec "$(dirname "$0")/run_pulsar_cpu_operator.sh" eq "$@"
