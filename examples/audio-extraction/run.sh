#!/usr/bin/env bash
# Wrapper for audio-extraction test scripts.
# Usage: ./run.sh [0|1|2|3|4|5|all] [args...]
#   ./run.sh 0          — extract audio & transcribe (re-run if needed)
#   ./run.sh 1          — clear Neo4j + rebuild S3 Vectors indices
#   ./run.sh 2          — ingest all paragraphs
#   ./run.sh 2 2        — ingest first 2 paragraphs only
#   ./run.sh 3          — search tests
#   ./run.sh 4          — show DescribesEdges
#   ./run.sh 5          — show narrative excerpts
#   ./run.sh all        — run 1→2→3→4→5 sequentially

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
GRAPHITI_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

check_clean() {
    local pids
    pids=$(ps -ef | grep -E "examples/audio-extraction/[0-9]_" | grep -v grep | awk '{print $2}' || true)
    if [ -n "$pids" ]; then
        echo "ERROR: Found running audio-extraction test process(es):"
        ps -ef | grep -E "examples/audio-extraction/[0-9]_" | grep -v grep
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
    uv run python "examples/audio-extraction/$script" "$@"
    echo "========== Finished $script =========="
}

if [ $# -lt 1 ]; then
    echo "Usage: $0 [0|1|2|3|4|5|all] [args...]"
    exit 1
fi

step="$1"
shift

cd "$GRAPHITI_DIR"
check_clean

case "$step" in
    0) run_script "0_transcribe.py" "$@" ;;
    1) run_script "1_clear.py" "$@" ;;
    2) run_script "2_ingest.py" "$@" ;;
    3) run_script "3_search.py" "$@" ;;
    4) run_script "4_describes.py" "$@" ;;
    5) run_script "5_narratives.py" "$@" ;;
    all)
        run_script "1_clear.py"
        run_script "2_ingest.py" "$@"
        run_script "3_search.py"
        run_script "4_describes.py"
        run_script "5_narratives.py"
        echo ""
        echo "========== All done =========="
        ;;
    *) echo "Unknown step: $step (use 0-5/all)"; exit 1 ;;
esac
