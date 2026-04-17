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
# Use a small grid so the convergence phase fits within the duration.
# Default 128x64x64 needs ~684 frames to start Cd output, but 5s
# at 60fps is only 300 frames. 64x32x32 brings it under budget.
echo "test: 5s render produces frames and Cd"
OUTPUT=$($BINARY \
    --wind=1.0 --duration=5 --grid=64x32x32 \
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

# Test 3: no fatal error messages.
# Shader-not-found warnings are non-fatal (features degrade gracefully).
echo "test: no errors in output"
ERROR_LINES=$(echo "$OUTPUT" | grep -i "error\|crash\|fault\|abort" \
    | grep -vi "shader file not found\|shader file is empty\|OpenGL error at" || true)
if [ -n "$ERROR_LINES" ]; then
    echo "  Matched lines:"
    echo "$ERROR_LINES" | head -5
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

# Test 5: --superres graceful fallback without weights
echo "test: --superres graceful fallback without weights"
SR_OUTPUT=$($BINARY --wind=1.0 --duration=2 --grid=32x16x16 \
    --model=$MODEL --output=$OUTDIR --superres 2>&1)
if echo "$SR_OUTPUT" | grep -q "falling back to native resolution"; then
    pass "superres fallback works"
else
    fail "no fallback message"
fi
# Should still produce frames and Cd despite SR failing
if echo "$SR_OUTPUT" | grep -q "Cd="; then
    pass "Cd still computed with SR fallback"
else
    fail "no Cd with SR fallback"
fi

# Cleanup
rm -rf "$OUTDIR"

echo ""
echo "=== Results: $PASSED passed, $FAILED failed ==="
[ "$FAILED" -eq 0 ] || exit 1
