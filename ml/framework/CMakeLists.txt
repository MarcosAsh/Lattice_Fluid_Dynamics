cmake_minimum_required(VERSION 3.16)
project(lattice_ml CXX)

set(CMAKE_CXX_STANDARD 17)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE
    "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native")

file(GLOB_RECURSE LIB_SOURCES "src/*.cpp")

add_library(lattice_ml STATIC ${LIB_SOURCES})
target_include_directories(lattice_ml PUBLIC include)
