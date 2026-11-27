cmake_minimum_required(VERSION 3.22)
project(lattice_ml LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(LATTICE_ML_NATIVE "Tune for this CPU with -march=native (not portable)" OFF)
option(LATTICE_ML_WARNINGS "Enable compiler warnings" ON)
option(LATTICE_ML_WERROR "Treat compiler warnings as errors" OFF)
option(BUILD_TESTING "Build the tests" ON)

function(lattice_ml_options target)
    target_compile_features(${target} PUBLIC cxx_std_20)
    set_target_properties(${target} PROPERTIES CXX_EXTENSIONS OFF)
    if(LATTICE_ML_NATIVE)
        target_compile_options(${target} PRIVATE -march=native)
    endif()
    if(LATTICE_ML_WARNINGS AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(${target} PRIVATE
            -Wall -Wextra -Wpedantic -Wshadow)
        if(LATTICE_ML_WERROR)
            target_compile_options(${target} PRIVATE -Werror)
        endif()
    endif()
endfunction()

add_library(lattice_ml STATIC
    src/autodiff.cpp
    src/layers/ad_linear.cpp
    src/layers/ad_swiglu.cpp
    src/loss.cpp
    src/memory_pool.cpp
    src/optimizer.cpp
    src/quantization.cpp
    src/tensor.cpp
    src/weights_io.cpp)
target_include_directories(lattice_ml PUBLIC include)
lattice_ml_options(lattice_ml)

if(BUILD_TESTING)
    enable_testing()
    add_executable(test_ml test_ml.cpp)
    target_link_libraries(test_ml PRIVATE lattice_ml)
    lattice_ml_options(test_ml)
    add_test(NAME ml_framework_tests COMMAND test_ml)
endif()
