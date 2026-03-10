#!/usr/bin/env bash
# Wrapper for docx-manual test scripts.
# Usage: ./run.sh [1|2|3|4|5|6|all] [args...]
#   ./run.sh 2 3       — run 2_ingest.py with arg 3 (first 3 sections)
#   ./run.sh all        — run 1→2→3→4→5→6 sequentially
#   ./run.sh 6          — run 6_verify_blocks.py

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
GRAPHITI_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

check_clean() {
    local pids
    pids=$(ps -ef | grep -E "examples/docx-manual/[0-9]_" | grep -v grep | awk '{print $2}' || true)
    if [ -n "$pids" ]; then
        echo "ERROR: Found running docx-manual test process(es):"
        ps -ef | grep -E "examples/docx-manual/[0-9]_" | grep -v grep
        echo ""
        echo "Kill them first:  kill $pids"
        exit 1
    fi
    echo "✓ No conflicting processes"
}

run_script() {
    local script="$1"
    shift
    echo ""
    echo "========== Running $script $* =========="
    uv run python "examples/docx-manual/$script" "$@"
    echo "========== Finished $script =========="
}

if [ $# -lt 1 ]; then
    echo "Usage: $0 [1|2|3|4|5|6|all] [args...]"
    exit 1
fi

step="$1"
shift

cd "$GRAPHITI_DIR"
check_clean

case "$step" in
    1) run_script "1_clear.py" "$@" ;;
    2) run_script "2_ingest.py" "$@" ;;
    3) run_script "3_search.py" "$@" ;;
    4) run_script "4_describes.py" "$@" ;;
    5) run_script "5_uncovered.py" "$@" ;;
    6) run_script "6_verify_blocks.py" "$@" ;;
    7) run_script "7_image_search.py" "$@" ;;
    8) run_script "8_entity_types.py" "$@" ;;
    all)
        run_script "1_clear.py"
        run_script "2_ingest.py" "$@"
        run_script "3_search.py"
        run_script "4_describes.py"
        run_script "5_uncovered.py"
        run_script "6_verify_blocks.py"
        echo ""
        echo "========== All done =========="
        ;;
    *) echo "Unknown step: $step (use 1-8/all)"; exit 1 ;;
esac
