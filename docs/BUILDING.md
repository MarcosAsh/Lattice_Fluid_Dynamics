# Building

## Requirements

- C11 compiler (GCC, Clang, or MSVC)
- CMake 3.16+
- OpenGL 4.3+ capable GPU and drivers
- SDL2 and SDL2_ttf
- GLEW

## Linux

### Arch

```bash
sudo pacman -S base-devel cmake sdl2-compat sdl2_ttf glew mesa
```

### Ubuntu / Debian

```bash
sudo apt install build-essential cmake pkg-config \
    libsdl2-dev libsdl2-ttf-dev libglew-dev \
    libgl1-mesa-dev
```

### Fedora

```bash
sudo dnf install cmake gcc SDL2-devel SDL2_ttf-devel \
    glew-devel mesa-libGL-devel
```

### Build

```bash
cd simulation
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

### Run

```bash
cd simulation
./build/3d_fluid_simulation_car --wind=1.0 --duration=10
```

On Wayland, you might need:

```bash
SDL_VIDEODRIVER=x11 ./build/3d_fluid_simulation_car
```

## macOS

```bash
brew install cmake sdl2 sdl2_ttf glew
cd simulation
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(sysctl -n hw.ncpu)
```

macOS uses the Accelerate framework for BLAS if available.

## Windows

Install Visual Studio 2022 with C++ workload, then use vcpkg:

```powershell
vcpkg install sdl2 sdl2-ttf glew
cd simulation
mkdir build && cd build
cmake .. -DCMAKE_TOOLCHAIN_FILE=[vcpkg-root]/scripts/buildsystems/vcpkg.cmake
cmake --build . --config Release
```

Or use MSYS2/MinGW:

```bash
pacman -S mingw-w64-x86_64-cmake mingw-w64-x86_64-SDL2 \
    mingw-w64-x86_64-SDL2_ttf mingw-w64-x86_64-glew
cd simulation
mkdir -p build && cd build
cmake .. -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release
mingw32-make -j4
```

## Website

```bash
cd website
npm install
npm run dev
```

Set `MODAL_RENDER_ENDPOINT` in `.env.local` to connect to the
Modal backend. Without it, the site runs in demo mode.

## CLI options

```
--wind=SPEED      Wind speed 0-5 (default: 1.0)
--viz=MODE        Visualization mode 0-6 (default: 1)
--collision=MODE  0=off, 1=AABB, 2=per-triangle (default: 1)
--duration=SECS   Headless render duration (0=interactive)
--output=PATH     Output directory for frames
--model=PATH      Path to OBJ model file
--reynolds=RE     Target Reynolds number (0=default viscosity)
--scale=SCALE     Model scale factor
```
