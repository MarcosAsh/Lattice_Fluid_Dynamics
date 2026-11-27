add_executable(lbm_validate lbm_validate.c)
target_link_libraries(lbm_validate PRIVATE lattice_gl)
lattice_warnings(lbm_validate)

# Quick GPU suite: ctest -L validation. Drop --quick for the full report set
find_program(UV_EXECUTABLE uv)
if(UV_EXECUTABLE)
    add_test(NAME validation_suite
        COMMAND ${UV_EXECUTABLE} run --no-project --with numpy --with matplotlib
                python validation/run_validation.py --quick
                --build ${CMAKE_BINARY_DIR}
                --out ${CMAKE_BINARY_DIR}/validation-out)
    set_tests_properties(validation_suite PROPERTIES
        LABELS validation
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        TIMEOUT 7200)
endif()
