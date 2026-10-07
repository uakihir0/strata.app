if(NOT DEFINED STRATA OR STRATA STREQUAL "")
  message(FATAL_ERROR "STRATA must name the engine executable")
endif()

execute_process(
  COMMAND "${STRATA}" --list-devices
  RESULT_VARIABLE _result
  OUTPUT_VARIABLE _stdout
  ERROR_VARIABLE _stderr)
if(NOT _result EQUAL 0)
  message(FATAL_ERROR "strata --list-devices failed (${_result}):\n${_stderr}\n${_stdout}")
endif()

foreach(_field device_name= gpu_core_count= working_set_bytes= max_buffer_length= max_threadgroup_memory_length=
               families= physical_memory_bytes= macos_version=)
  string(FIND "${_stdout}" "${_field}" _position)
  if(_position EQUAL -1)
    message(FATAL_ERROR "strata --list-devices omitted ${_field}:\n${_stdout}")
  endif()
endforeach()
if(NOT _stdout MATCHES "gpu_core_count=[1-9][0-9]*" OR
   NOT _stdout MATCHES "working_set_bytes=[1-9][0-9]*" OR
   NOT _stdout MATCHES "max_buffer_length=[1-9][0-9]*" OR
   NOT _stdout MATCHES "max_threadgroup_memory_length=[1-9][0-9]*" OR
   NOT _stdout MATCHES "physical_memory_bytes=[1-9][0-9]*" OR
   NOT _stdout MATCHES "macos_version=[0-9]+\\.[0-9]+")
  message(FATAL_ERROR "strata --list-devices returned an empty or malformed hardware fact:\n${_stdout}")
endif()
if(NOT _stdout MATCHES "families=[^\n]*apple7" OR NOT _stdout MATCHES "families=[^\n]*metal4")
  message(FATAL_ERROR "strata --list-devices omitted a required GPU family:\n${_stdout}")
endif()

if(NOT DEFINED BUILD_DIR OR BUILD_DIR STREQUAL "")
  message(FATAL_ERROR "BUILD_DIR must name the configured build directory")
endif()
set(_existing_trace "${BUILD_DIR}/metal/list-devices-existing.gputrace")
file(MAKE_DIRECTORY "${BUILD_DIR}/metal")
file(WRITE "${_existing_trace}" "existing test trace")
execute_process(
  COMMAND "${CMAKE_COMMAND}" -E env "STRATA_METAL_CAPTURE=${_existing_trace}" "${STRATA}" --list-devices
  RESULT_VARIABLE _capture_result
  OUTPUT_VARIABLE _capture_stdout
  ERROR_VARIABLE _capture_stderr)
file(REMOVE "${_existing_trace}")
if(NOT _capture_result EQUAL 0)
  message(FATAL_ERROR "STRATA_METAL_CAPTURE must not interfere with --list-devices "
                      "(${_capture_result}):\n${_capture_stderr}\n${_capture_stdout}")
endif()
