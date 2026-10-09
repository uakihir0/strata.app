if(NOT DEFINED STRATA OR NOT DEFINED FIXTURE_TOOL OR NOT DEFINED PYTHON OR
   NOT DEFINED FIXTURE_HELPER OR NOT DEFINED BUILD_ROOT OR NOT DEFINED TEST_ROOT)
  message(FATAL_ERROR
    "STRATA, FIXTURE_TOOL, PYTHON, FIXTURE_HELPER, BUILD_ROOT and TEST_ROOT are required")
endif()

if(NOT _STRATA_WEIGHT_STORE_CLI_CHILD)
  set(_build_root_input "${BUILD_ROOT}")
  set(_test_root_input "${TEST_ROOT}")
  cmake_path(ABSOLUTE_PATH _build_root_input NORMALIZE
             OUTPUT_VARIABLE _build_root_abs)
  cmake_path(ABSOLUTE_PATH _test_root_input NORMALIZE
             OUTPUT_VARIABLE _test_root_abs)
  set(_expected_test_root
      "${_build_root_abs}/metal/weight-store-cli/fixtures")
  if(NOT _test_root_abs STREQUAL _expected_test_root)
    message(FATAL_ERROR "refusing cleanup outside the dedicated WeightStore CLI build directory")
  endif()

  set(_child_common_args
      "-DSTRATA=${STRATA}"
      "-DFIXTURE_TOOL=${FIXTURE_TOOL}"
      "-DPYTHON=${PYTHON}"
      "-DFIXTURE_HELPER=${FIXTURE_HELPER}"
      "-DBUILD_ROOT=${_build_root_abs}"
      "-DTEST_ROOT=${_test_root_abs}"
      "-D_STRATA_WEIGHT_STORE_CLI_CHILD=TRUE")

  string(RANDOM LENGTH 32 ALPHABET 0123456789abcdef _fixture_run_id)
  set(_test_dir "${_test_root_abs}-${_fixture_run_id}")
  if(EXISTS "${_test_dir}" OR IS_SYMLINK "${_test_dir}")
    message(FATAL_ERROR "refusing to reuse an existing WeightStore CLI test directory")
  endif()
  execute_process(
    COMMAND "${PYTHON}" "${FIXTURE_HELPER}"
            --create-weight-cli-test-root "${_build_root_abs}"
            "fixtures-${_fixture_run_id}"
    RESULT_VARIABLE _create_root_rc
    OUTPUT_VARIABLE _create_root_out
    ERROR_VARIABLE _create_root_err)
  if(NOT _create_root_rc EQUAL 0)
    message(FATAL_ERROR
      "could not securely create the WeightStore CLI test directory "
      "(${_create_root_rc}): ${_create_root_err}${_create_root_out}")
  endif()
  string(STRIP "${_create_root_out}" _test_root_identity)
  if(NOT _test_root_identity MATCHES "^([0-9]+) +([0-9]+)$")
    message(FATAL_ERROR "fixture helper returned an invalid directory identity")
  endif()
  set(_test_root_device "${CMAKE_MATCH_1}")
  set(_test_root_inode "${CMAKE_MATCH_2}")
  execute_process(
    COMMAND "${CMAKE_COMMAND}" ${_child_common_args}
            "-D_STRATA_WEIGHT_STORE_CLI_RUN_ID=${_fixture_run_id}"
            -P "${CMAKE_CURRENT_LIST_FILE}"
    RESULT_VARIABLE _test_rc
    OUTPUT_VARIABLE _test_out
    ERROR_VARIABLE _test_err)
  execute_process(
    COMMAND "${PYTHON}" "${FIXTURE_HELPER}"
            --remove-weight-cli-test-root "${_build_root_abs}"
            "fixtures-${_fixture_run_id}" "${_test_root_device}"
            "${_test_root_inode}"
    RESULT_VARIABLE _remove_root_rc
    OUTPUT_VARIABLE _remove_root_out
    ERROR_VARIABLE _remove_root_err)
  if(NOT _remove_root_rc EQUAL 0)
    message(FATAL_ERROR
      "could not securely remove the WeightStore CLI test directory "
      "(${_remove_root_rc}): ${_remove_root_err}${_remove_root_out}")
  endif()
  if(NOT _test_rc EQUAL 0)
    message(FATAL_ERROR "WeightStore CLI test failed (${_test_rc}): ${_test_err}${_test_out}")
  endif()

  string(RANDOM LENGTH 32 ALPHABET 0123456789abcdef _failure_run_id)
  set(_failure_dir "${_test_root_abs}-${_failure_run_id}")
  if(EXISTS "${_failure_dir}" OR IS_SYMLINK "${_failure_dir}")
    message(FATAL_ERROR "refusing to reuse an existing cleanup-test directory")
  endif()
  execute_process(
    COMMAND "${PYTHON}" "${FIXTURE_HELPER}"
            --create-weight-cli-test-root "${_build_root_abs}"
            "fixtures-${_failure_run_id}"
    RESULT_VARIABLE _create_failure_root_rc
    OUTPUT_VARIABLE _create_failure_root_out
    ERROR_VARIABLE _create_failure_root_err)
  if(NOT _create_failure_root_rc EQUAL 0)
    message(FATAL_ERROR
      "could not securely create the injected-failure test directory "
      "(${_create_failure_root_rc}): "
      "${_create_failure_root_err}${_create_failure_root_out}")
  endif()
  string(STRIP "${_create_failure_root_out}" _failure_root_identity)
  if(NOT _failure_root_identity MATCHES "^([0-9]+) +([0-9]+)$")
    message(FATAL_ERROR "cleanup test helper returned an invalid directory identity")
  endif()
  set(_failure_root_device "${CMAKE_MATCH_1}")
  set(_failure_root_inode "${CMAKE_MATCH_2}")
  set(_cleanup_sentinel
      "${_build_root_abs}/metal/weight-store-cli/cleanup-sentinel-${_failure_run_id}")
  if(EXISTS "${_cleanup_sentinel}" OR IS_SYMLINK "${_cleanup_sentinel}")
    message(FATAL_ERROR "refusing to reuse an existing cleanup sentinel")
  endif()
  file(WRITE "${_cleanup_sentinel}" "preserve cleanup sentinel\n")
  execute_process(
    COMMAND "${CMAKE_COMMAND}" ${_child_common_args}
            "-D_STRATA_WEIGHT_STORE_CLI_RUN_ID=${_failure_run_id}"
            "-D_STRATA_WEIGHT_STORE_CLI_SENTINEL=${_cleanup_sentinel}"
            -D_STRATA_WEIGHT_STORE_CLI_INJECT_FAILURE=TRUE
            -P "${CMAKE_CURRENT_LIST_FILE}"
    RESULT_VARIABLE _failure_rc
    OUTPUT_VARIABLE _failure_out
    ERROR_VARIABLE _failure_err)
  execute_process(
    COMMAND "${PYTHON}" "${FIXTURE_HELPER}"
            --remove-weight-cli-test-root "${_build_root_abs}"
            "fixtures-${_failure_run_id}" "${_failure_root_device}"
            "${_failure_root_inode}"
    RESULT_VARIABLE _remove_failure_root_rc
    OUTPUT_VARIABLE _remove_failure_root_out
    ERROR_VARIABLE _remove_failure_root_err)
  if(NOT _remove_failure_root_rc EQUAL 0)
    message(FATAL_ERROR
      "could not securely remove the injected-failure test directory "
      "(${_remove_failure_root_rc}): "
      "${_remove_failure_root_err}${_remove_failure_root_out}")
  endif()
  if(_failure_rc EQUAL 0 OR
     NOT "${_failure_err}" MATCHES "injected WeightStore CLI failure")
    message(FATAL_ERROR
      "cleanup lifecycle self-test did not observe the injected failure: "
      "${_failure_out}${_failure_err}")
  endif()
  if(NOT EXISTS "${_cleanup_sentinel}" OR IS_SYMLINK "${_cleanup_sentinel}")
    message(FATAL_ERROR "cleanup followed an external test symlink")
  endif()
  file(READ "${_cleanup_sentinel}" _cleanup_sentinel_content)
  if(NOT _cleanup_sentinel_content STREQUAL "preserve cleanup sentinel\n")
    message(FATAL_ERROR "cleanup changed the external sentinel")
  endif()
  file(REMOVE "${_cleanup_sentinel}")
  return()
endif()

if(NOT DEFINED _STRATA_WEIGHT_STORE_CLI_RUN_ID)
  message(FATAL_ERROR "invalid WeightStore CLI test run identity")
endif()
string(LENGTH "${_STRATA_WEIGHT_STORE_CLI_RUN_ID}" _fixture_run_id_length)
if(NOT _fixture_run_id_length EQUAL 32 OR
   NOT _STRATA_WEIGHT_STORE_CLI_RUN_ID MATCHES "^[0-9a-f]+$")
  message(FATAL_ERROR "invalid WeightStore CLI test run identity")
endif()

set(_test_dir "${TEST_ROOT}-${_STRATA_WEIGHT_STORE_CLI_RUN_ID}")
if(NOT IS_DIRECTORY "${_test_dir}" OR IS_SYMLINK "${_test_dir}")
  message(FATAL_ERROR "missing or unsafe owned WeightStore CLI test directory")
endif()
set(_test_owner_path "${_test_dir}/.strata-owned-fixture")
if(NOT EXISTS "${_test_owner_path}" OR IS_SYMLINK "${_test_owner_path}")
  message(FATAL_ERROR "missing or unsafe WeightStore CLI directory ownership marker")
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
if(_STRATA_WEIGHT_STORE_CLI_INJECT_FAILURE)
  if(DEFINED _STRATA_WEIGHT_STORE_CLI_SENTINEL)
    file(CREATE_LINK "${_STRATA_WEIGHT_STORE_CLI_SENTINEL}"
         "${_test_dir}/external-sentinel-link" SYMBOLIC RESULT _sentinel_link_result)
    if(NOT _sentinel_link_result STREQUAL "0")
      message(FATAL_ERROR "could not create the cleanup sentinel link: ${_sentinel_link_result}")
    endif()
  endif()
  message(FATAL_ERROR "injected WeightStore CLI failure after fixture generation")
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

file(WRITE "${_pack}/experts.bin" "legacy PC expert pack\n")
execute_process(
  COMMAND "${STRATA}" --plan-only --native "${_native}" --pack "${_pack}"
          --mtp "${_mtp}" --spec 4 --mtp-window 32768 --max-context 262144
          --plan-w-mib 131072 --plan-ram-mib 131072
  RESULT_VARIABLE _legacy_experts_rc
  OUTPUT_VARIABLE _legacy_experts_out
  ERROR_VARIABLE _legacy_experts_err)
if(NOT _legacy_experts_rc EQUAL 0 OR
   NOT _legacy_experts_err MATCHES "strata: ignoring pack experts.bin on macOS" OR
   NOT _legacy_experts_out STREQUAL _real_out)
  message(FATAL_ERROR "Mac CLI must ignore and report pack experts.bin without changing the plan: "
                      "${_legacy_experts_err}${_legacy_experts_out}")
endif()
file(REMOVE "${_pack}/experts.bin")

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
