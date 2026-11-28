#!/bin/bash
# Coarse 64x32x32 and fine 128x64x64 runs for the super-resolution check
# usage: run_cases.sh BUILD_DIR OUT_DIR, run from simulation/
set -u
B=$1/3d_fluid_simulation_car
OUT=$2
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT"
uv run --no-sync python "$HERE/make_sphere.py" "$OUT/sphere.obj"

run() { # name model extra re
    local name=$1 model=$2 extra=$3 re=$4
    for g in coarse fine; do
        # fine runs twice the steps so both cover the same convective time
        if [ $g = coarse ]; then grid=64x32x32 dur=30 iv=600; else grid=128x64x64 dur=60 iv=1200; fi
        d=$OUT/$name/$g
        mkdir -p "$d"
        "$B" --model="$model" $extra --grid=$grid --reynolds="$re" --strict-reynolds \
            --duration=$dur --vtk-output="$d" --vtk-interval=$iv --vtk-stats \
            --manifest="$d/manifest.json" >"$d/log.txt" 2>&1
        echo "$name $g rc=$?"
    done
}

run sphere_re20 "$OUT/sphere.obj" "--scale=0.5" 20
run sphere_re50 "$OUT/sphere.obj" "--scale=0.5" 50
run ahmed25_re140 assets/3d-files/ahmed_25deg_m.obj "" 140
run car_re140 assets/3d-files/car-model.obj "" 140
