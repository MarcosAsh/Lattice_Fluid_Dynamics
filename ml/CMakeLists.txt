cmake_minimum_required(VERSION 3.16)
project(lattice_ml_tools CXX)

add_subdirectory(framework)

add_executable(train train.cpp)
target_link_libraries(train lattice_ml)
