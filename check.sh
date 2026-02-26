#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"

if [[ ! -x "$PYTHON_BIN" ]]; then
	PYTHON_BIN="python3"
fi

cd "$ROOT_DIR"

echo "Running unit tests..."
"$PYTHON_BIN" -m unittest discover -s tests -v

echo
echo "All checks passed."
