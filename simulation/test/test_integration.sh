#!/bin/bash
# Integration test: run the simulation headless and check output.
# Needs Xvfb: xvfb-run bash test/test_integration.sh
set -e

BINARY="./build/3d_fluid_simulation_car"
MODEL="assets/3d-files/ahmed_25deg_m.obj"
OUTDIR=$(mktemp -d)
PASSED=0
FAILED=0

pass() { echo "  PASS: $1"; PASSED=$((PASSED + 1)); }
fail() { echo "  FAIL: $1"; FAILED=$((FAILED + 1)); }

echo "=== Integration Tests ==="
echo ""

# Check binary exists
if [ ! -x "$BINARY" ]; then
    echo "Binary not found at $BINARY, building..."
    mkdir -p build && cd build && cmake .. && make -j4
    cd ..
fi

# Test 1: basic run produces frames
echo "test: 5s render produces frames and Cd"
OUTPUT=$($BINARY \
    --wind=1.0 --duration=5 \
    --model=$MODEL --output=$OUTDIR 2>&1)
FRAMES=$(ls "$OUTDIR"/frame_*.ppm 2>/dev/null | wc -l)
if [ "$FRAMES" -gt 0 ]; then
    pass "got $FRAMES frames"
else
    fail "no frames produced"
fi

# Test 2: stdout contains Cd lines
echo "test: stdout has Cd output"
if echo "$OUTPUT" | grep -q "Cd="; then
    pass "Cd= found in output"
else
    fail "no Cd= in output"
fi

# Test 3: no error messages
echo "test: no errors in output"
if echo "$OUTPUT" | grep -qi "error\|crash\|fault\|abort"; then
    fail "error found in output"
else
    pass "clean output"
fi

# Test 4: effective Re is printed
echo "test: Re is reported"
if echo "$OUTPUT" | grep -q "Effective Re"; then
    pass "Re reported"
else
    fail "no Re in output"
fi

# Cleanup
rm -rf "$OUTDIR"

echo ""
echo "=== Results: $PASSED passed, $FAILED failed ==="
[ "$FAILED" -eq 0 ] || exit 1
