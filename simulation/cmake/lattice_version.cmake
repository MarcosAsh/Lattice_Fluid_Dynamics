# Writes lattice_version.h, touching it only when the git state changes
set(GIT_DESCRIBE "unknown")
if(GIT_EXECUTABLE)
    execute_process(
        COMMAND ${GIT_EXECUTABLE} describe --tags --always --dirty
        WORKING_DIRECTORY ${SOURCE_DIR}
        OUTPUT_VARIABLE out
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
        RESULT_VARIABLE rc)
    if(rc EQUAL 0 AND out)
        set(GIT_DESCRIBE ${out})
    endif()
endif()
configure_file(${INPUT} ${OUTPUT} @ONLY)
