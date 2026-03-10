#!/usr/bin/env bash
# Wrapper for sanguo test scripts.
# Usage: ./run.sh [1|2|3|4|all] [args...]
#   ./run.sh 2 3       — run 2_ingest.py with arg 3
#   ./run.sh all 3     — run 1→2→3→4 sequentially, passing args to 2_ingest
#   ./run.sh 3         — run 3_search.py

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
GRAPHITI_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Check for conflicting python processes
check_clean() {
    local pids
    pids=$(ps -ef | grep -E "examples/sanguo/[0-9]_" | grep -v grep | awk '{print $2}' || true)
    if [ -n "$pids" ]; then
        echo "ERROR: Found running sanguo test process(es):"
        ps -ef | grep -E "examples/sanguo/[0-9]_" | grep -v grep
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
    uv run python "examples/sanguo/$script" "$@"
    echo "========== Finished $script =========="
}

if [ $# -lt 1 ]; then
    echo "Usage: $0 [1|2|3|4|all] [args...]"
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
    all)
        # Default to 3 paragraphs if no count specified
        if [ $# -eq 0 ]; then
            set -- 3
        fi
        run_script "1_clear.py"
        run_script "2_ingest.py" "$@"
        run_script "3_search.py"
        run_script "4_describes.py"
        run_script "5_uncovered.py"
        echo ""
        echo "========== All done =========="
        ;;
    *) echo "Unknown step: $step (use 1/2/3/4/5/all)"; exit 1 ;;
esac
