cmake_minimum_required(VERSION 3.22)
project(3d_fluid_simulation_car
    VERSION 1.5.0
    DESCRIPTION "GPU Lattice Boltzmann wind tunnel"
    LANGUAGES C)

set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)
# getopt_long, setenv and M_PI need the GNU dialect
set(CMAKE_C_EXTENSIONS ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(LATTICE_WARNINGS "Enable compiler warnings" ON)
option(LATTICE_WERROR "Treat compiler warnings as errors" OFF)
set(LATTICE_SANITIZE "" CACHE STRING
    "Sanitizers to build with, e.g. address,undefined")
option(BUILD_TESTING "Build the tests" ON)

if(LATTICE_SANITIZE)
    add_compile_options(-fsanitize=${LATTICE_SANITIZE}
                        -fno-sanitize-recover=all
                        -fno-omit-frame-pointer)
    add_link_options(-fsanitize=${LATTICE_SANITIZE})
endif()

find_package(PkgConfig REQUIRED)
pkg_check_modules(SDL2 REQUIRED IMPORTED_TARGET sdl2)
pkg_check_modules(GL REQUIRED IMPORTED_TARGET gl)
# Optional EGL for headless GPU rendering
pkg_check_modules(EGL IMPORTED_TARGET egl)

function(lattice_warnings target)
    if(NOT LATTICE_WARNINGS OR NOT CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
        return()
    endif()
    target_compile_options(${target} PRIVATE
        -Wall -Wextra -Wpedantic -Wshadow -Wformat=2 -Wstrict-prototypes
        -Wmissing-prototypes -Wundef -Wvla)
    if(LATTICE_WERROR)
        target_compile_options(${target} PRIVATE -Werror)
    endif()
endfunction()

# Vendored glad GL loader, built without our warnings
add_library(lattice_glad STATIC lib/glad/src/gl.c)
target_include_directories(lattice_glad SYSTEM PUBLIC lib/glad/include)
if(EGL_FOUND)
    target_sources(lattice_glad PRIVATE lib/glad/src/egl.c)
endif()

# OBJ loading and voxelization, no GL
add_library(lattice_mesh STATIC
    obj-file-loader/lib/model_loader.c
    src/voxelize.c)
target_include_directories(lattice_mesh PUBLIC lib obj-file-loader/lib)
target_link_libraries(lattice_mesh PUBLIC m)
lattice_warnings(lattice_mesh)

# CLI parsing and drag post-processing, no GL
add_library(lattice_util STATIC
    src/cli.c
    src/drag_metrics.c
    src/ml_predict.c)
target_include_directories(lattice_util PUBLIC lib)
target_link_libraries(lattice_util PUBLIC m)
lattice_warnings(lattice_util)

# GL context, LBM solver and GPU helpers
add_library(lattice_gl STATIC
    src/gl_context.c
    src/gl_helpers.c
    src/lbm.c
    src/model_bounds.c
    src/opengl_utils.c
    src/superres.c
    src/view_matrix.c
    src/vti_export.c)
target_include_directories(lattice_gl PUBLIC lib)
target_link_libraries(lattice_gl PUBLIC
    lattice_glad lattice_mesh PkgConfig::SDL2 PkgConfig::GL m
    ${CMAKE_DL_LIBS})
if(EGL_FOUND)
    target_compile_definitions(lattice_gl PRIVATE HAVE_EGL)
    target_link_libraries(lattice_gl PUBLIC PkgConfig::EGL)
    message(STATUS "EGL found -- headless GPU enabled")
else()
    message(STATUS "EGL not found -- SDL fallback only")
endif()
lattice_warnings(lattice_gl)

add_executable(3d_fluid_simulation_car
    src/main.c
    src/event_handlers.c
    src/render_model.c)
target_link_libraries(3d_fluid_simulation_car PRIVATE lattice_gl lattice_util)
lattice_warnings(3d_fluid_simulation_car)

# Standalone voxelizer CLI: OBJ -> .voxbin for ML dataset pipelines
add_executable(voxelize_obj src/voxelize_main.c)
target_link_libraries(voxelize_obj PRIVATE lattice_mesh)
lattice_warnings(voxelize_obj)

if(BUILD_TESTING)
    enable_testing()

    add_executable(test_voxelize test/test_voxelize.c)
    target_link_libraries(test_voxelize PRIVATE lattice_mesh)
    lattice_warnings(test_voxelize)

    add_executable(test_cli test/test_cli.c)
    target_link_libraries(test_cli PRIVATE lattice_util)
    lattice_warnings(test_cli)

    # Needs a GL 4.3 context
    add_executable(test_lbm test/test_lbm.c)
    target_link_libraries(test_lbm PRIVATE lattice_gl)
    lattice_warnings(test_lbm)

    add_test(NAME voxelize_unit_tests COMMAND test_voxelize)
    add_test(NAME cli_unit_tests COMMAND test_cli)
    add_test(NAME cli_help COMMAND 3d_fluid_simulation_car --help)
    set_tests_properties(voxelize_unit_tests cli_unit_tests cli_help
        PROPERTIES LABELS cpu)
    set_tests_properties(cli_help PROPERTIES
        PASS_REGULAR_EXPRESSION "Usage:")

    # Shaders and assets are loaded relative to simulation/
    add_test(NAME lbm_unit_tests COMMAND test_lbm)
    add_test(NAME integration
        COMMAND bash test/test_integration.sh
                $<TARGET_FILE:3d_fluid_simulation_car>)
    set_tests_properties(lbm_unit_tests integration PROPERTIES
        LABELS gl
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        TIMEOUT 1800)
endif()
