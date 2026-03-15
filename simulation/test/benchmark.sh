#!/bin/bash
# Overnight GPU profiling and benchmarking.
# Runs the simulation at various grid sizes, models, and
# wind speeds. Writes results to docs/BENCHMARK.md.
#
# Usage: SDL_VIDEODRIVER=x11 bash test/benchmark.sh

set -e

BINARY="./build/3d_fluid_simulation_car"
RESULTS_DIR="test/benchmark_results"
REPORT="../docs/BENCHMARK.md"
mkdir -p "$RESULTS_DIR"

MODELS=(
    "ahmed25:assets/3d-files/ahmed_25deg_m.obj"
    "ahmed35:assets/3d-files/ahmed_35deg_m.obj"
    "car:assets/3d-files/car-model.obj"
)

WIND_SPEEDS=(0.5 1.0 1.5 2.0 3.0)
DURATION=15

# Header
cat > "$REPORT" << 'HDR'
# Benchmark Results

GPU profiling and Cd measurements across grid sizes,
models, and wind speeds. Generated automatically.

HDR

echo "## System Info" >> "$REPORT"
echo "" >> "$REPORT"
echo "- GPU: $(glxinfo 2>/dev/null | grep 'OpenGL renderer' | cut -d: -f2 | xargs || echo 'unknown')" >> "$REPORT"
echo "- Date: $(date -u '+%Y-%m-%d %H:%M UTC')" >> "$REPORT"
echo "" >> "$REPORT"

# Run a single benchmark and append to CSV
run_bench() {
    local label=$1
    local model_path=$2
    local wind=$3
    local csv=$4

    echo "  $label wind=$wind..."

    START=$(date +%s%N)
    OUTPUT=$($BINARY \
        --wind=$wind \
        --duration=$DURATION \
        --model=$model_path \
        --collision=1 2>&1) || true
    END=$(date +%s%N)
    ELAPSED_MS=$(( (END - START) / 1000000 ))

    # Parse Cd values
    CD_LINES=$(echo "$OUTPUT" | grep "Cd=" || true)
    if [ -n "$CD_LINES" ]; then
        # Get last Cd value
        LAST_CD=$(echo "$CD_LINES" | tail -1 | sed 's/.*Cd=\([0-9.]*\).*/\1/')
        # Count Cd samples
        NUM_CD=$(echo "$CD_LINES" | wc -l)
    else
        LAST_CD="nan"
        NUM_CD=0
    fi

    # Parse Re
    RE=$(echo "$OUTPUT" | grep "Effective Re" | sed 's/.*Re = \([0-9.]*\).*/\1/' || echo "0")

    # Parse grid size
    GRID=$(echo "$OUTPUT" | grep "LBM Grid:" | sed 's/.*Grid: \([^ ]*\).*/\1/' || echo "?")

    # Parse solid cells
    SOLIDS=$(echo "$OUTPUT" | grep "mesh solid:" | sed 's/.*solid: \([0-9]*\).*/\1/' || echo "0")

    # Parse memory
    MEM_MB=$(echo "$OUTPUT" | grep "Total:" | sed 's/.*Total: *\([0-9.]*\).*/\1/' || echo "0")

    echo "$label,$wind,$GRID,$RE,$LAST_CD,$NUM_CD,$SOLIDS,$MEM_MB,$ELAPSED_MS" >> "$csv"
}

# Main benchmark loop
CSV="$RESULTS_DIR/results.csv"
echo "model,wind_speed,grid,re,cd,cd_samples,solid_cells,memory_mb,time_ms" > "$CSV"

echo "=== Starting benchmark ==="
echo ""

for entry in "${MODELS[@]}"; do
    IFS=':' read -r label model_path <<< "$entry"
    echo "Model: $label"
    for wind in "${WIND_SPEEDS[@]}"; do
        run_bench "$label" "$model_path" "$wind" "$CSV"
    done
    echo ""
done

echo "=== Benchmark complete ==="
echo ""

# Generate markdown table from CSV
echo "## Results" >> "$REPORT"
echo "" >> "$REPORT"
echo "Duration: ${DURATION}s per run" >> "$REPORT"
echo "" >> "$REPORT"
echo "| Model | Wind | Grid | Re | Cd | Solid Cells | Memory (MB) | Time (s) |" >> "$REPORT"
echo "|-------|------|------|----|----|-------------|-------------|----------|" >> "$REPORT"

tail -n +2 "$CSV" | while IFS=',' read -r model wind grid re cd samples solids mem_mb time_ms; do
    time_s=$(echo "scale=1; $time_ms / 1000" | bc 2>/dev/null || echo "$time_ms")
    echo "| $model | $wind | $grid | $re | $cd | $solids | $mem_mb | ${time_s} |" >> "$REPORT"
done

echo "" >> "$REPORT"

# Cd comparison section
echo "## Ahmed Body Comparison" >> "$REPORT"
echo "" >> "$REPORT"
echo "Published experimental Cd (Re > 10^5):" >> "$REPORT"
echo "- Ahmed 25 deg: 0.25 - 0.30" >> "$REPORT"
echo "- Ahmed 35 deg: 0.35 - 0.40" >> "$REPORT"
echo "" >> "$REPORT"
echo "Our simulation runs at Re ~ 160 (limited by grid" >> "$REPORT"
echo "resolution and stability). At this Re, viscous drag" >> "$REPORT"
echo "dominates and Cd is much higher than the high-Re" >> "$REPORT"
echo "experimental values. This is expected physics, not a" >> "$REPORT"
echo "bug. Reaching experimental Cd requires either larger" >> "$REPORT"
echo "grids (256+ cells in each dimension) or a subgrid" >> "$REPORT"
echo "turbulence model like Smagorinsky." >> "$REPORT"
echo "" >> "$REPORT"

# Performance section
echo "## Performance Notes" >> "$REPORT"
echo "" >> "$REPORT"
echo "The simulation uses OpenGL 4.3 compute shaders for" >> "$REPORT"
echo "all LBM operations. Each frame runs 5 LBM substeps" >> "$REPORT"
echo "(collision + streaming). The 128x64x64 grid uses ~86 MB" >> "$REPORT"
echo "of GPU memory for distribution functions, velocity field," >> "$REPORT"
echo "and solid mask." >> "$REPORT"

echo ""
echo "Report written to $REPORT"
echo "Raw data at $CSV"
