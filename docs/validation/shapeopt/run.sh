#!/bin/sh
# Laptop run behind verify.md and dataset.csv, from the repository root
set -e
cmake -B build -S simulation -DCMAKE_BUILD_TYPE=Release
cmake --build build -j8
uv run python ml/cli.py gen-shapes --config "$PWD/docs/validation/shapeopt/laptop_config.yaml" \
    --local --binary build/3d_fluid_simulation_car --out build/shapeopt/data --timeout 600
for i in 0 1 2 3 4; do
    uv run python ml/cli.py train-shapeopt --data build/shapeopt/data/shapeopt_data.sobin \
        --seed "$i" --out "build/shapeopt/s$i"
done
uv run python ml/cli.py optimize \
    --weights build/shapeopt/s0/shapeopt_model_best.pt build/shapeopt/s1/shapeopt_model_best.pt \
    build/shapeopt/s2/shapeopt_model_best.pt build/shapeopt/s3/shapeopt_model_best.pt \
    build/shapeopt/s4/shapeopt_model_best.pt \
    --base-model ahmed25 --output build/shapeopt/ensemble.obj --name ensemble \
    --binary build/3d_fluid_simulation_car --verify-out build/shapeopt/verify \
    --grid 64x32x32 --grid 128x64x64 --timeout 1800
uv run python ml/cli.py optimize --weights build/shapeopt/s0/shapeopt_model_best.pt \
    --base-model ahmed25 --output build/shapeopt/single.obj --name single --no-verify
# reuses the ensemble's four runs and adds two for the single shape
uv run python ml/cli.py verify build/shapeopt/ensemble.json build/shapeopt/single.json \
    --binary build/3d_fluid_simulation_car --out build/shapeopt/verify \
    --grid 64x32x32 --grid 128x64x64 --timeout 1800
