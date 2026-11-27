# Releasing

Steps to cut a release, using 2.0.0 as the example.

## Before tagging

1. Start from an up-to-date `master` with a clean tree and green CI.
2. Set the version in `simulation/CMakeLists.txt`:

   ```cmake
   project(3d_fluid_simulation_car
       VERSION 2.0.0
   ```

3. Add the version and date to `CITATION.cff`:

   ```yaml
   version: 2.0.0
   date-released: "2026-12-01"
   ```

4. In `CHANGELOG.md`, replace `Unreleased` in the `2.0.0` heading with the
   date, in `YYYY-MM-DD` form. Add an empty `## [Unreleased]` section above
   it for the next round of changes.
5. Build and check the version string:

   ```bash
   cmake -B build -S simulation -DCMAKE_BUILD_TYPE=Release
   cmake --build build -j"$(nproc)"
   ./build/3d_fluid_simulation_car --version
   ```

6. Commit the three files as `Release 2.0.0`, push to `master` and wait
   for CI.

## Tag

```bash
git tag -a v2.0.0 -m "v2.0.0"
git push origin v2.0.0
```

The tag must start with `v`. That is what starts the release workflow.

## What the release workflow does

`.github/workflows/release.yml` runs on `ubuntu-latest` for any `v*` tag.
It:

1. installs the build dependencies and Mesa,
2. configures and builds a Release build,
3. runs `ctest` under `xvfb-run` on Mesa's software renderer,
4. packs the binary, `shaders/`, `assets/`, `README.md` and `LICENSE` into
   `lattice-v2.0.0-linux-x86_64.tar.gz`,
5. creates the GitHub release with that tarball and notes generated from
   the merged pull requests.

The test step runs every CTest label. CI leaves out `-L validation`, but
the release workflow does not. If `uv` is on the runner, the slow
validation suite runs on the software renderer too. Watch the first run.

The generated notes list pull requests only. Replace them with the
changelog section:

```bash
gh release edit v2.0.0 --notes-file notes.md
```

where `notes.md` holds the `2.0.0` section of `CHANGELOG.md`.

## Check the uploaded binary

On a Linux machine with an OpenGL 4.3 GPU:

```bash
gh release download v2.0.0 -R MarcosAsh/Lattice_Fluid_Dynamics \
    -p 'lattice-v2.0.0-linux-x86_64.tar.gz'
tar xzf lattice-v2.0.0-linux-x86_64.tar.gz
cd lattice-v2.0.0
./3d_fluid_simulation_car --version
./3d_fluid_simulation_car --model=assets/3d-files/ahmed_25deg_m.obj \
    --converge=0.02 --max-duration=60 --manifest=run.json
echo "exit $?"
```

Check that:

- `--version` prints `2.0.0 (v2.0.0)`. A commit hash in the brackets
  means the runner did not see the tag. A `-dirty` suffix means the
  build tree was modified.
- The run exits with 0. Status 3 means NaN and 4 means stability resets.
- `run.json` has version `2.0.0`, `"dirty": false` and a `convergence`
  section.

Run it from inside the unpacked directory, since the binary loads
`shaders/` and `assets/` from the working directory.

## After the release

If the project is archived on Zenodo, check that the new version got a
DOI and add it to `CITATION.cff` and the README.
