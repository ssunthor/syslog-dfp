#!/usr/bin/env bash
# Test runner entrypoint for the syslog-DFP test container.
# Controls which pytest tier runs via TEST_TIER; passes PYTEST_ARGS through.
set -euo pipefail

# Install pytest if not already present in the image.
if ! python -c "import pytest" &>/dev/null 2>&1; then
    echo "Installing pytest..."
    pip install pytest -q
fi

# morpheus_app.py creates FEEDBACK_DIR at import time; ensure the path exists
# so Tier 3 tests that import the module don't fail on a missing directory.
mkdir -p /app/models/dfp-models/feedback

echo "======================================================"
echo " syslog-DFP test suite  |  TIER=${TEST_TIER:-all}"
echo "======================================================"

case "${TEST_TIER:-all}" in
    unit)
        exec python -m pytest tests/ -m "not gpu and not morpheus" -v "${PYTEST_ARGS:-}"
        ;;
    gpu)
        exec python -m pytest tests/ -m "gpu and not morpheus" -v "${PYTEST_ARGS:-}"
        ;;
    morpheus)
        exec python -m pytest tests/ -m "morpheus" -v "${PYTEST_ARGS:-}"
        ;;
    all|*)
        exec python -m pytest tests/ -v "${PYTEST_ARGS:-}"
        ;;
esac
