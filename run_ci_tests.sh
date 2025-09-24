#!/bin/bash
# CI Test Strategy Script
# Implements the staged pytest matrix from Future_instruction.txt

set -e  # Exit on any failure

echo "🧪 CI Test Strategy - Staged pytest matrix"
echo "=========================================="

# Stage 1: Fast unit tests only (no integration)
echo ""
echo "📋 Stage 1: Unit Tests (fast, no external dependencies)"
echo "------------------------------------------------------"
echo "Command: pytest -m 'not integration' --maxfail=5 --durations=20"
pytest -m "not integration" --maxfail=5 --durations=20

echo ""
echo "✅ Stage 1 PASSED - Unit tests completed successfully"

# Stage 2: Full integration suite (Redis, API, connectors, Alpaca dry_run)
echo ""
echo "📋 Stage 2: Integration Tests (Redis, API, connectors)"
echo "-------------------------------------------------------"
echo "Command: pytest -m 'integration' --maxfail=5 --durations=20"
pytest -m "integration" --maxfail=5 --durations=20

echo ""
echo "✅ Stage 2 PASSED - Integration tests completed successfully"

# Stage 3: Soak/long-running tests (selective, for nightly builds)
echo ""
echo "📋 Stage 3: Soak Tests (long-running, nightly builds)"
echo "------------------------------------------------------"
echo "Command: pytest -m 'soak' --maxfail=5 --durations=20"
echo "Note: These are typically run in nightly builds due to duration"

if [[ "${RUN_SOAK_TESTS:-0}" == "1" ]]; then
    echo "RUN_SOAK_TESTS=1 detected - running soak tests..."
    pytest -m "soak" --maxfail=5 --durations=20
    echo "✅ Stage 3 PASSED - Soak tests completed successfully"
else
    echo "⏭️  Stage 3 SKIPPED - Set RUN_SOAK_TESTS=1 to run soak tests"
fi

echo ""
echo "🎉 CI Test Strategy completed successfully!"
echo "All stages passed. Ready for deployment."