#!/usr/bin/env bash
# Start the demo web UI against the bundled binaries and weights/.
set -euo pipefail
cd "$(dirname "$0")"
export IRO_NUM_THREADS="${IRO_NUM_THREADS:-2}"
BIN=bin/irodori-onemkl
grep -qw avx512_vnni /proc/cpuinfo 2>/dev/null || { echo "note: CPU without AVX-512 VNNI, int8 will be slower; fp32 is recommended"; }
exec python3 demo/server.py --binary "$BIN" --weights weights --outputs outputs "$@"
