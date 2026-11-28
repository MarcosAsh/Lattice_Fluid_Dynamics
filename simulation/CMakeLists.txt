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

# Build info for --version
find_package(Git QUIET)
set(LATTICE_VERSION_H ${CMAKE_CURRENT_BINARY_DIR}/generated/lattice_version.h)
add_custom_target(lattice_version
    COMMAND ${CMAKE_COMMAND}
        -DGIT_EXECUTABLE=${GIT_EXECUTABLE}
        -DSOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}
        -DVERSION=${PROJECT_VERSION}
        -DINPUT=${CMAKE_CURRENT_SOURCE_DIR}/cmake/lattice_version.h.in
        -DOUTPUT=${LATTICE_VERSION_H}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/lattice_version.cmake
    BYPRODUCTS ${LATTICE_VERSION_H}
    VERBATIM)

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

# Checkpoint files, no GL
add_library(lattice_io STATIC src/checkpoint.c src/surface.c)
target_include_directories(lattice_io PUBLIC lib)
target_link_libraries(lattice_io PUBLIC m)
lattice_warnings(lattice_io)

# CLI parsing and drag post-processing, no GL
add_library(lattice_util STATIC
    src/cli.c
    src/convergence.c
    src/drag_metrics.c
    src/manifest.c
    src/ml_predict.c)
target_include_directories(lattice_util
    PUBLIC lib
    PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/generated)
target_link_libraries(lattice_util PUBLIC m)
add_dependencies(lattice_util lattice_version)
lattice_warnings(lattice_util)

# GL context, LBM solver and GPU helpers
add_library(lattice_gl STATIC
    src/gl_context.c
    src/gl_helpers.c
    src/lbm.c
    src/lbm_surface.c
    src/model_bounds.c
    src/opengl_utils.c
    src/superres.c
    src/view_matrix.c
    src/vti_export.c)
target_include_directories(lattice_gl PUBLIC lib)
target_link_libraries(lattice_gl PUBLIC
    lattice_glad lattice_mesh lattice_io PkgConfig::SDL2 PkgConfig::GL m
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

    add_executable(test_convergence test/test_convergence.c)
    target_link_libraries(test_convergence PRIVATE lattice_util)
    lattice_warnings(test_convergence)

    add_executable(test_checkpoint test/test_checkpoint.c)
    target_link_libraries(test_checkpoint PRIVATE lattice_io)
    lattice_warnings(test_checkpoint)

    add_executable(test_surface test/test_surface.c)
    target_link_libraries(test_surface PRIVATE lattice_io)
    lattice_warnings(test_surface)

    add_executable(test_manifest test/test_manifest.c)
    target_link_libraries(test_manifest PRIVATE lattice_util)
    lattice_warnings(test_manifest)

    # Needs a GL 4.3 context
    add_executable(test_lbm test/test_lbm.c)
    target_link_libraries(test_lbm PRIVATE lattice_gl)
    lattice_warnings(test_lbm)

    add_test(NAME voxelize_unit_tests COMMAND test_voxelize)
    add_test(NAME cli_unit_tests COMMAND test_cli)
    add_test(NAME convergence_unit_tests COMMAND test_convergence)
    add_test(NAME checkpoint_unit_tests COMMAND test_checkpoint)
    add_test(NAME surface_unit_tests COMMAND test_surface)
    add_test(NAME manifest_unit_tests COMMAND test_manifest)
    set_tests_properties(manifest_unit_tests PROPERTIES
        LABELS cpu FIXTURES_SETUP manifest_file)
    find_package(Python3 COMPONENTS Interpreter)
    if(Python3_FOUND)
        add_test(NAME manifest_json_parses
            COMMAND ${Python3_EXECUTABLE} -c
                "import json,sys; d=json.load(open(sys.argv[1])); assert d['schema_version'] == 1"
                test_manifest.json)
        set_tests_properties(manifest_json_parses PROPERTIES
            LABELS cpu FIXTURES_REQUIRED manifest_file)
    endif()
    add_test(NAME cli_help COMMAND 3d_fluid_simulation_car --help)
    add_test(NAME cli_version COMMAND 3d_fluid_simulation_car --version)
    set_tests_properties(voxelize_unit_tests cli_unit_tests
        convergence_unit_tests checkpoint_unit_tests surface_unit_tests
        cli_help cli_version
        PROPERTIES LABELS cpu)
    set_tests_properties(cli_help PROPERTIES
        PASS_REGULAR_EXPRESSION "Usage:")
    set_tests_properties(cli_version PROPERTIES
        PASS_REGULAR_EXPRESSION
        "^3d_fluid_simulation_car ${PROJECT_VERSION} \\([^)]+\\)")

    # Shaders and assets are loaded relative to simulation/
    add_test(NAME lbm_unit_tests COMMAND test_lbm)
    add_test(NAME integration
        COMMAND bash test/test_integration.sh
                $<TARGET_FILE:3d_fluid_simulation_car>)
    set_tests_properties(lbm_unit_tests integration PROPERTIES
        LABELS gl
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        TIMEOUT 1800)

    # GPU validation suite, slow: ctest -L validation
    add_subdirectory(validation)
endif()
