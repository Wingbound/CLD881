#!/usr/bin/env bash
# Entry point: checks the toolchain, then hands over to sweep.py.
#   ./run_sweep.sh            full run (smoke test first, then the sweep)
#   ./run_sweep.sh --smoke    smoke test only
#   ./run_sweep.sh --collect  rebuild summary.csv from existing logs
set -euo pipefail
cd "$(dirname "$0")"
for tool in g++ python3; do
  command -v "$tool" >/dev/null || { echo "ERROR: $tool not found in PATH"; exit 1; }
done
[ -f phaseField_poiseuille_linearField_AC.cpp ] || {
  echo "ERROR: phaseField_poiseuille_linearField_AC.cpp not found in $(pwd)"; exit 1; }
exec python3 sweep.py "$@"
