if(NOT DEFINED STRATA OR NOT DEFINED FIXTURE_TOOL OR NOT DEFINED PYTHON OR
   NOT DEFINED FIXTURE_HELPER OR NOT DEFINED BUILD_ROOT OR NOT DEFINED TEST_ROOT)
  message(FATAL_ERROR
    "STRATA, FIXTURE_TOOL, PYTHON, FIXTURE_HELPER, BUILD_ROOT and TEST_ROOT are required")
endif()

string(RANDOM LENGTH 32 ALPHABET 0123456789abcdef _fixture_run_id)
set(_test_dir "${TEST_ROOT}-${_fixture_run_id}")
if(EXISTS "${_test_dir}" OR IS_SYMLINK "${_test_dir}")
  message(FATAL_ERROR "refusing to reuse an existing WeightStore CLI test directory")
endif()
file(MAKE_DIRECTORY "${_test_dir}")
set(_input "${_test_dir}/input")
execute_process(
  COMMAND "${PYTHON}" "${FIXTURE_HELPER}"
          --create-weight-cli-child "${BUILD_ROOT}" "${_test_dir}" input "${FIXTURE_TOOL}"
  RESULT_VARIABLE _fixture_rc
  OUTPUT_VARIABLE _fixture_out
  ERROR_VARIABLE _fixture_err)
if(NOT _fixture_rc EQUAL 0)
  message(FATAL_ERROR "could not create sparse WeightStore fixture (${_fixture_rc}): ${_fixture_err}")
endif()

set(_protected_fixture "${_test_dir}/protected-fixture")
file(MAKE_DIRECTORY "${_protected_fixture}")
file(WRITE "${_protected_fixture}/sentinel" "preserve this file\n")
file(SHA256 "${_protected_fixture}/sentinel" _sentinel_before)
execute_process(
  COMMAND "${PYTHON}" "${FIXTURE_HELPER}"
          --create-weight-cli-child "${BUILD_ROOT}" "${_test_dir}" protected-fixture
          "${FIXTURE_TOOL}"
  RESULT_VARIABLE _nonempty_fixture_rc
  OUTPUT_VARIABLE _nonempty_fixture_out
  ERROR_VARIABLE _nonempty_fixture_err)
file(SHA256 "${_protected_fixture}/sentinel" _sentinel_after)
if(_nonempty_fixture_rc EQUAL 0 OR
   NOT "${_nonempty_fixture_err}" MATCHES "must be an empty directory" OR
   NOT _sentinel_after STREQUAL _sentinel_before)
  message(FATAL_ERROR "fixture generator must refuse a non-empty destination without changing it: "
                      "${_nonempty_fixture_out}${_nonempty_fixture_err}")
endif()

set(_symlink_fixture "${_test_dir}/symlink-fixture")
execute_process(
  COMMAND "${CMAKE_COMMAND}" -E create_symlink "${_input}" "${_symlink_fixture}"
  RESULT_VARIABLE _fixture_symlink_rc
  ERROR_VARIABLE _fixture_symlink_err)
if(NOT _fixture_symlink_rc EQUAL 0)
  message(FATAL_ERROR "could not create the fixture destination symlink: ${_fixture_symlink_err}")
endif()
file(SIZE "${_input}/tiny.gguf" _native_size_before_fixture_symlink)
execute_process(
  COMMAND "${PYTHON}" "${FIXTURE_HELPER}"
          --create-weight-cli-child "${BUILD_ROOT}" "${_test_dir}" symlink-fixture
          "${FIXTURE_TOOL}"
  RESULT_VARIABLE _symlink_fixture_rc
  OUTPUT_VARIABLE _symlink_fixture_out
  ERROR_VARIABLE _symlink_fixture_message)
file(SIZE "${_input}/tiny.gguf" _native_size_after_fixture_symlink)
if(_symlink_fixture_rc EQUAL 0 OR
   NOT "${_symlink_fixture_message}" MATCHES "must not be a symlink" OR
   NOT _native_size_after_fixture_symlink EQUAL _native_size_before_fixture_symlink)
  message(FATAL_ERROR "fixture generator must refuse a symlink destination without changing its target: "
                      "${_symlink_fixture_out}${_symlink_fixture_message}")
endif()

set(_native "${_input}/tiny.gguf")
set(_pack "${_input}/pack")
set(_mtp "${_input}/mtp-rt")
set(_default_pack "${_test_dir}/pack/full")
file(COPY "${_pack}/" DESTINATION "${_default_pack}")
file(SIZE "${_native}" _native_size_before)
execute_process(
  COMMAND "${STRATA}" --write-geometry "${_native}" --native "${_native}" --pack "${_pack}"
  RESULT_VARIABLE _native_collision_rc
  OUTPUT_VARIABLE _native_collision_out
  ERROR_VARIABLE _native_collision_err)
if(NOT _native_collision_rc EQUAL 2 OR NOT _native_collision_err MATCHES "output aliases input")
  message(FATAL_ERROR "geometry export must refuse to overwrite its GGUF input: "
                      "${_native_collision_out}${_native_collision_err}")
endif()
file(SIZE "${_native}" _native_size_after)
if(NOT _native_size_after EQUAL _native_size_before)
  message(FATAL_ERROR "geometry export changed its GGUF input")
endif()

file(SHA256 "${_pack}/index.txt" _index_sha_before)
execute_process(
  COMMAND "${STRATA}" --write-geometry "${_pack}/index.txt" --native "${_native}" --pack "${_pack}"
  RESULT_VARIABLE _pack_collision_rc
  OUTPUT_VARIABLE _pack_collision_out
  ERROR_VARIABLE _pack_collision_err)
if(NOT _pack_collision_rc EQUAL 2 OR NOT _pack_collision_err MATCHES "output aliases input")
  message(FATAL_ERROR "geometry export must refuse to overwrite pack inputs: "
                      "${_pack_collision_out}${_pack_collision_err}")
endif()
file(SHA256 "${_pack}/index.txt" _index_sha_after)
if(NOT _index_sha_after STREQUAL _index_sha_before)
  message(FATAL_ERROR "geometry export changed the pack index")
endif()

file(SHA256 "${_default_pack}/index.txt" _default_index_sha_before)
execute_process(
  COMMAND "${STRATA}" --write-geometry "${_default_pack}/index.txt" --native "${_native}"
  WORKING_DIRECTORY "${_test_dir}"
  RESULT_VARIABLE _default_pack_collision_rc
  OUTPUT_VARIABLE _default_pack_collision_out
  ERROR_VARIABLE _default_pack_collision_err)
if(NOT _default_pack_collision_rc EQUAL 2 OR
   NOT _default_pack_collision_err MATCHES "output aliases input")
  message(FATAL_ERROR "geometry export must protect the implicit pack/full input: "
                      "${_default_pack_collision_out}${_default_pack_collision_err}")
endif()
file(SHA256 "${_default_pack}/index.txt" _default_index_sha_after)
if(NOT _default_index_sha_after STREQUAL _default_index_sha_before)
  message(FATAL_ERROR "geometry export changed the implicit pack index")
endif()

set(_native_alias "${_input}/native-alias.gguf")
execute_process(
  COMMAND "${CMAKE_COMMAND}" -E create_symlink "${_native}" "${_native_alias}"
  RESULT_VARIABLE _alias_rc
  ERROR_VARIABLE _alias_err)
if(NOT _alias_rc EQUAL 0)
  message(FATAL_ERROR "could not create the GGUF alias fixture: ${_alias_err}")
endif()
execute_process(
  COMMAND "${STRATA}" --write-geometry "${_native_alias}" --native "${_native}" --pack "${_pack}"
  RESULT_VARIABLE _alias_collision_rc
  OUTPUT_VARIABLE _alias_collision_out
  ERROR_VARIABLE _alias_collision_err)
if(NOT _alias_collision_rc EQUAL 2 OR NOT _alias_collision_err MATCHES "output aliases input")
  message(FATAL_ERROR "geometry export must refuse a symlink alias to an input: "
                      "${_alias_collision_out}${_alias_collision_err}")
endif()

set(_no_mtp_spec_rc 0)
execute_process(
  COMMAND "${STRATA}" --plan-only --native "${_native}" --pack "${_pack}" --spec 4
          --plan-w-mib 131072 --plan-ram-mib 131072
  RESULT_VARIABLE _no_mtp_spec_rc
  OUTPUT_VARIABLE _no_mtp_spec_out
  ERROR_VARIABLE _no_mtp_spec_err)
if(NOT _no_mtp_spec_rc EQUAL 2 OR
   NOT _no_mtp_spec_err MATCHES "requires geometry that includes a validated --mtp runtime")
  message(FATAL_ERROR "speculative plan without MTP data must return argument error 2: "
                      "${_no_mtp_spec_out}${_no_mtp_spec_err}")
endif()

execute_process(
  COMMAND "${STRATA}" --plan-only --native "${_native}" --pack "${_pack}"
          --mtp "${_mtp}" --spec 4 --mtp-window 32768 --max-context 262144
          --plan-w-mib 131072 --plan-ram-mib 131072
  RESULT_VARIABLE _real_rc
  OUTPUT_VARIABLE _real_out
  ERROR_VARIABLE _real_err)
if(NOT _real_rc EQUAL 0 OR NOT _real_out MATCHES "^PLAN ")
  message(FATAL_ERROR "real-file plan failed (${_real_rc}): ${_real_err}${_real_out}")
endif()

set(_geometry "${_test_dir}/tiny.geometry")
set(_sentinel "${_test_dir}/sentinel")
file(WRITE "${_sentinel}" "sentinel must not be overwritten\n")
execute_process(
  COMMAND "${CMAKE_COMMAND}" -E create_symlink "${_sentinel}" "${_geometry}.tmp"
  RESULT_VARIABLE _symlink_rc
  ERROR_VARIABLE _symlink_err)
if(NOT _symlink_rc EQUAL 0)
  message(FATAL_ERROR "could not create the temporary-path symlink fixture: ${_symlink_err}")
endif()
execute_process(
  COMMAND "${STRATA}" --write-geometry "${_geometry}" --native "${_native}" --pack "${_pack}"
          --mtp "${_mtp}"
          --geometry-family qwen --geometry-model tiny
  RESULT_VARIABLE _write_rc
  OUTPUT_VARIABLE _write_out
  ERROR_VARIABLE _write_err)
if(NOT _write_rc EQUAL 0 OR NOT EXISTS "${_geometry}")
  message(FATAL_ERROR "geometry export failed (${_write_rc}): ${_write_err}${_write_out}")
endif()
if(NOT IS_SYMLINK "${_geometry}.tmp")
  message(FATAL_ERROR "geometry export replaced the pre-existing .tmp symlink")
endif()
file(READ "${_sentinel}" _sentinel_contents)
if(NOT _sentinel_contents STREQUAL "sentinel must not be overwritten\n")
  message(FATAL_ERROR "geometry export modified the pre-existing .tmp symlink target")
endif()
file(GLOB _temporary_files "${_geometry}.tmp.*")
if(_temporary_files)
  message(FATAL_ERROR "geometry export left a unique temporary file behind: ${_temporary_files}")
endif()

execute_process(
  COMMAND "${STRATA}" --write-geometry "${_geometry}" --native "${_native}" --pack "${_pack}"
          --mtp "${_mtp}"
          --geometry-family qwen --geometry-model tiny
  RESULT_VARIABLE _overwrite_rc
  OUTPUT_VARIABLE _overwrite_out
  ERROR_VARIABLE _overwrite_err)
if(NOT _overwrite_rc EQUAL 0)
  message(FATAL_ERROR "geometry export could not replace an existing target (${_overwrite_rc}): "
                      "${_overwrite_err}${_overwrite_out}")
endif()

execute_process(
  COMMAND "${STRATA}" --plan-only --geometry "${_geometry}"
          --spec 4 --mtp-window 32768 --max-context 262144
          --plan-w-mib 131072 --plan-ram-mib 131072
  RESULT_VARIABLE _geometry_rc
  OUTPUT_VARIABLE _geometry_out
  ERROR_VARIABLE _geometry_err)
if(NOT _geometry_rc EQUAL 0 OR NOT _geometry_out MATCHES "^PLAN ")
  message(FATAL_ERROR "geometry-file plan failed (${_geometry_rc}): ${_geometry_err}${_geometry_out}")
endif()
if(NOT _real_out STREQUAL _geometry_out)
  message(FATAL_ERROR "real-file and exported-geometry PLAN lines differ:\nreal: ${_real_out}\ngeometry: ${_geometry_out}")
endif()

execute_process(
  COMMAND "${STRATA}" --plan-only --native "${_native}" --pack "${_pack}" --mtp "${_mtp}"
          --spec 4 --mtp-window 262144 --max-context 262144
          --plan-w-mib 131072 --plan-ram-mib 131072
  RESULT_VARIABLE _full_window_rc
  OUTPUT_VARIABLE _full_window_out
  ERROR_VARIABLE _full_window_err)
if(NOT _full_window_rc EQUAL 0 OR NOT _full_window_out MATCHES "mtp_bind_mib=102")
  message(FATAL_ERROR "a full-context MTP window must increase planned bind scratch: "
                      "${_full_window_out}${_full_window_err}")
endif()

execute_process(
  COMMAND "${STRATA}" --plan-only --native "${_native}" --pack "${_pack}" --mtp "${_mtp}"
          --spec 4 --mtp-window 262145 --max-context 262144
          --plan-w-mib 131072 --plan-ram-mib 131072
  RESULT_VARIABLE _larger_window_rc
  OUTPUT_VARIABLE _larger_window_out
  ERROR_VARIABLE _larger_window_err)
if(NOT _larger_window_rc EQUAL 0 OR NOT _larger_window_out STREQUAL _full_window_out)
  message(FATAL_ERROR "a window at or above context must use the full-context scratch: "
                      "${_larger_window_out}${_larger_window_err}")
endif()

execute_process(
  COMMAND "${STRATA}" --plan-only --native "${_native}" --pack "${_pack}" --mtp "${_mtp}"
          --spec 4 --mtp-window 4294967296 --max-context 262144
          --plan-w-mib 131072 --plan-ram-mib 131072
  RESULT_VARIABLE _overflow_window_rc
  OUTPUT_VARIABLE _overflow_window_out
  ERROR_VARIABLE _overflow_window_err)
if(NOT _overflow_window_rc EQUAL 2 OR NOT _overflow_window_err MATCHES "plan option is out of range")
  message(FATAL_ERROR "an MTP window that exceeds the runtime integer width must be rejected: "
                      "${_overflow_window_out}${_overflow_window_err}")
endif()
