cmake_minimum_required(VERSION 3.22)
project(lattice_ml_tools LANGUAGES CXX)

enable_testing()
add_subdirectory(framework)

add_executable(train train.cpp)
target_link_libraries(train PRIVATE lattice_ml)
lattice_ml_options(train)
