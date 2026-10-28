#!/usr/bin/env bash
# Single entry point for every experiment:
#   Release build -> correctness gate -> timing matrix + end-to-end pass -> charts.
#
#   ./benchmark/run_benchmark.sh                      # full matrix (about 3 min on an M4 Pro once DIV2K is cached)
#   ./benchmark/run_benchmark.sh --threads 1,2,4,8    # any option of benchmark/run_experiments.py
#
# Environment overrides: BUILD_DIR (default build), RESULTS_DIR (default results),
# PYTHON (default python3), JOBS (parallel compile jobs).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$ROOT/build}"
RESULTS_DIR="${RESULTS_DIR:-$ROOT/results}"
PYTHON="${PYTHON:-python3}"
JOBS="${JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)}"

echo "==> Release build in $BUILD_DIR"
cmake -S "$ROOT" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release
cmake --build "$BUILD_DIR" --parallel "$JOBS"

echo "==> Correctness gate and experiments -> $RESULTS_DIR"
"$PYTHON" "$ROOT/benchmark/run_experiments.py" \
    --build-dir "$BUILD_DIR" --output-dir "$RESULTS_DIR" ${1+"$@"}

if "$PYTHON" -c "import matplotlib" >/dev/null 2>&1; then
    echo "==> Charts -> $RESULTS_DIR"
    "$PYTHON" "$ROOT/benchmark/plot.py" --results "$RESULTS_DIR"
else
    echo "matplotlib is not installed for $PYTHON; skipped the charts (pip install -r requirements.txt)." >&2
fi
