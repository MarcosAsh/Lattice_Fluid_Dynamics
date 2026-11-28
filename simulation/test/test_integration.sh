#!/bin/bash
# Integration test: run the simulation headless and check output.
# Needs Xvfb: xvfb-run bash test/test_integration.sh [binary]
set -e

BINARY="${1:-./build/3d_fluid_simulation_car}"
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
    --model=$MODEL --output=$OUTDIR --superres \
    --sr-weights=/nonexistent/weights.bin 2>&1)
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

# Test 6: run manifest lands in the output directory
echo "test: manifest written with --output"
MANIFEST="$OUTDIR/manifest.json"
if [ -s "$MANIFEST" ] && grep -q '"schema_version": 1' "$MANIFEST"; then
    pass "manifest.json written"
else
    fail "no manifest.json"
fi
if command -v python3 >/dev/null 2>&1; then
    if python3 - "$MANIFEST" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
for k in ("git", "gl", "parameters", "lattice", "model", "run", "results",
          "convergence", "start_time_utc", "wall_seconds"):
    assert k in m, k
assert m["results"]["cd_samples"] > 0
assert m["lattice"]["tau"] > 0.5
assert m["model"]["solid_cells"] > 0
assert m["run"]["output_frames"] > 0
c = m["convergence"]
assert c["enabled"] is False
assert c["method"].startswith("mser5"), c["method"]
for k in ("samples", "rejected_samples", "truncated_samples", "kept_samples",
          "stationary", "reason", "cd_mean", "cd_sem", "cd_ci95_half_width",
          "effective_sample_size", "lag1_autocorrelation"):
    assert k in c, k
assert c["samples"] + c["rejected_samples"] == m["results"]["cd_samples"]
if c["cd_mean"] is None:
    assert c["reason"], "no statistics and no reason"
else:
    assert c["cd_ci95_half_width"] >= 0 and c["effective_sample_size"] > 0
PY
    then
        pass "manifest parses and has the fields"
    else
        fail "manifest fields"
    fi
fi

# Test 7: --converge stops early and reports
echo "test: --converge reports and records convergence"
CONV_OUTPUT=$($BINARY --wind=1.0 --grid=32x16x16 --model=$MODEL \
    --converge=0.02 --converge-window=4 --max-duration=10 \
    --manifest="$OUTDIR/conv.json" 2>&1)
if echo "$CONV_OUTPUT" | grep -q "^Convergence: "; then
    pass "convergence reported"
else
    fail "no convergence report"
fi
if grep -q '"enabled": true' "$OUTDIR/conv.json" 2>/dev/null; then
    pass "convergence in manifest"
else
    fail "convergence missing from manifest"
fi
if echo "$CONV_OUTPUT" | grep -q "^Drag statistics: " &&
    ! echo "$CONV_OUTPUT" | grep -A4 "^Drag statistics: " | grep -q "Cd="; then
    pass "drag statistics reported"
else
    fail "no drag statistics, or they contain Cd="
fi

# Test 8: a restarted run ends exactly where a straight run ends
echo "test: --restart continues a run bitwise"
CK_ARGS="--wind=1.0 --grid=32x16x16 --model=$MODEL"
$BINARY $CK_ARGS --duration=7 --checkpoint="$OUTDIR/a.ckpt" \
    --manifest="$OUTDIR/a.json" >"$OUTDIR/a.log" 2>&1 || true
$BINARY $CK_ARGS --duration=9 --restart="$OUTDIR/a.ckpt" \
    --checkpoint="$OUTDIR/b.ckpt" --manifest="$OUTDIR/b.json" \
    >"$OUTDIR/b.log" 2>&1 || true
$BINARY $CK_ARGS --duration=9 --checkpoint="$OUTDIR/c.ckpt" \
    --checkpoint-interval=500 --manifest="$OUTDIR/c.json" \
    >"$OUTDIR/c.log" 2>&1 || true
if [ -s "$OUTDIR/b.ckpt" ] && cmp -s "$OUTDIR/b.ckpt" "$OUTDIR/c.ckpt"; then
    pass "final checkpoints of restarted and straight runs are identical"
else
    fail "restarted run differs from the straight run"
fi
if command -v python3 >/dev/null 2>&1; then
    if python3 - "$OUTDIR/a.json" "$OUTDIR/b.json" "$OUTDIR/c.json" <<'PY'
import json, sys
a, b, c = (json.load(open(p)) for p in sys.argv[1:])
assert b["restarted_from"]["step"] == a["checkpoint"]["last_step"] > 0
assert b["restarted_from"]["drag_statistics"] == "continued"
assert c["restarted_from"] is None and c["checkpoint"]["written"] == 2
assert b["results"]["cd_samples"] > a["results"]["cd_samples"] > 0
for k in ("results", "convergence", "run"):
    assert b[k] == c[k], (k, b[k], c[k])
PY
    then
        pass "restarted manifest matches the straight run"
    else
        fail "restarted manifest differs"
    fi
fi
set +e
$BINARY $CK_ARGS --grid=32x16x8 --duration=9 --restart="$OUTDIR/a.ckpt" \
    >"$OUTDIR/d.log" 2>&1
RC=$?
head -c 3000 "$OUTDIR/a.ckpt" >"$OUTDIR/t.ckpt"
$BINARY $CK_ARGS --duration=9 --restart="$OUTDIR/t.ckpt" >"$OUTDIR/t.log" 2>&1
RC_T=$?
set -e
if [ $RC -eq 2 ] && grep -q "grid 32x16x16 does not match 32x16x8" "$OUTDIR/d.log" &&
    [ $RC_T -eq 2 ] && grep -q "truncated" "$OUTDIR/t.log"; then
    pass "other grid and truncated file refused"
else
    fail "bad checkpoint not refused (rc $RC, $RC_T)"
fi

# Cleanup
rm -rf "$OUTDIR"

echo ""
echo "=== Results: $PASSED passed, $FAILED failed ==="
[ "$FAILED" -eq 0 ] || exit 1
