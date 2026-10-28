#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$script_dir"

if [[ -n "${PYTHON:-}" ]]; then
    python_bin="$PYTHON"
elif [[ -n "${VIRTUAL_ENV:-}" || -n "${CONDA_PREFIX:-}" ]]; then
    python_bin="python"
elif [[ -x "$script_dir/.venv/bin/python" ]]; then
    python_bin="$script_dir/.venv/bin/python"
elif [[ -f "$script_dir/.venv/Scripts/python.exe" ]]; then
    python_bin="$script_dir/.venv/Scripts/python.exe"
elif command -v python >/dev/null 2>&1; then
    python_bin="python"
elif command -v python3 >/dev/null 2>&1; then
    python_bin="python3"
else
    printf 'Python was not found. Install Python and requirements.txt first.\n' >&2
    exit 1
fi

if ! command -v "$python_bin" >/dev/null 2>&1; then
    printf 'Python interpreter was not found: %s\n' "$python_bin" >&2
    exit 1
fi

"$python_bin" -u "$script_dir/download_weights.py"
exec "$python_bin" -u "$script_dir/run_demo.py" \
    --evaluation-labels "$script_dir/examples/evaluation_labels.csv" "$@"
