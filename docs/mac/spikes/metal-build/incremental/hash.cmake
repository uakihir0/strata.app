file(SHA256 ${METALLIB} h)
file(WRITE ${INC}.tmp "#define STRATA_METALLIB_SHA256 \"${h}\"\n")
execute_process(COMMAND ${CMAKE_COMMAND} -E copy_if_different ${INC}.tmp ${INC})
file(REMOVE ${INC}.tmp)
