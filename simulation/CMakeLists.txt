cmake_minimum_required(VERSION 3.16) 
project(3d_fluid_simulation_car C)

set(CMAKE_C_STANDARD 11)

# Use pkg-config to find libraries
find_package(PkgConfig REQUIRED)
pkg_check_modules(SDL2 REQUIRED sdl2)
pkg_check_modules(SDL2_TTF REQUIRED SDL2_ttf)
pkg_check_modules(GLEW REQUIRED glew)
pkg_check_modules(GL REQUIRED gl)

enable_testing()

add_executable(3d_fluid_simulation_car
    src/main.c
    src/coloring.c
    src/fluid_cube.c
    obj-file-loader/lib/model_loader.c
    src/render_model.c
    src/particle_system.c
    src/opengl_utils.c
    src/lbm.c
)

# Link libraries
target_link_libraries(3d_fluid_simulation_car 
    ${SDL2_LIBRARIES}
    ${SDL2_TTF_LIBRARIES}
    ${GLEW_LIBRARIES}
    ${GL_LIBRARIES}
    m
)

# Include directories
target_include_directories(3d_fluid_simulation_car PRIVATE 
    ${SDL2_INCLUDE_DIRS}
    ${SDL2_TTF_INCLUDE_DIRS}
    ${GLEW_INCLUDE_DIRS}
    ${GL_INCLUDE_DIRS}
)

# Link directories
target_link_directories(3d_fluid_simulation_car PRIVATE
    ${SDL2_LIBRARY_DIRS}
    ${SDL2_TTF_LIBRARY_DIRS}
    ${GLEW_LIBRARY_DIRS}
    ${GL_LIBRARY_DIRS}
)

# LBM unit tests
add_executable(test_lbm
    test/test_lbm.c
    src/lbm.c
    src/opengl_utils.c
)
target_include_directories(test_lbm PRIVATE
    ${SDL2_INCLUDE_DIRS}
    ${GLEW_INCLUDE_DIRS}
    ${GL_INCLUDE_DIRS}
)
target_link_libraries(test_lbm
    ${SDL2_LIBRARIES}
    ${GLEW_LIBRARIES}
    ${GL_LIBRARIES}
    m
)
target_link_directories(test_lbm PRIVATE
    ${SDL2_LIBRARY_DIRS}
    ${GLEW_LIBRARY_DIRS}
    ${GL_LIBRARY_DIRS}
)
add_test(NAME lbm_unit_tests COMMAND test_lbm)
