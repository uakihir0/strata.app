# cmake -P script for the strata_version test (docs/mac/08-build-and-distribution.md, section 6.3).  Inputs: STRATA
# (the engine), METALLIB, VERSION, METAL_REV.  Checks what setup.py and tools/metal/check_asset.py rely on: the
# --version line, one engine= tag, and a metallib_sha256= tag equal to the file next to the engine.
execute_process(COMMAND ${STRATA} --version OUTPUT_VARIABLE _out RESULT_VARIABLE _rc)
if(NOT _rc EQUAL 0 OR NOT _out STREQUAL "strata ${VERSION} metal ${METAL_REV}\n")
  message(FATAL_ERROR "strata --version: exit ${_rc}, printed '${_out}'")
endif()
file(STRINGS ${STRATA} _tags REGEX "^(engine|metallib_sha256)=")
file(SHA256 ${METALLIB} _h)
if(NOT _tags STREQUAL "engine=${VERSION};metallib_sha256=${_h}")
  message(FATAL_ERROR "tags in ${STRATA}: '${_tags}', want engine=${VERSION} and metallib_sha256=${_h}")
endif()
message(STATUS "strata ${VERSION} metal ${METAL_REV}, metallib ${_h}")
