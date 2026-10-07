if(NOT DEFINED STRATA OR NOT DEFINED TEST_DIR)
  message(FATAL_ERROR "STRATA and TEST_DIR are required")
endif()

file(MAKE_DIRECTORY "${TEST_DIR}")
set(_geometry "${TEST_DIR}/valid.geometry")
file(WRITE "${_geometry}" [=[strata-geometry 1
family=test
model=fixture
shard1_name=model.gguf
shard1_bytes=1
n_layers=1
n_expert=1
nonexpert_bytes=1024
wrapped_unread_bytes=0
dense_pack_bytes=0
mtp_bytes=0
mtp_bind_bytes=0
n_qsa_layers=1
n_kv_head=1
head_dim=1
n_embd=4
vocab=16
types=q4_0
layer 0 blob=4096 gu=q4_0 d=q4_0
]=])

execute_process(
  COMMAND "${STRATA}" --plan-only --geometry "${_geometry}" --plan-w-mib 65536 --plan-ram-mib 65536
  RESULT_VARIABLE _fit_rc OUTPUT_VARIABLE _fit_out ERROR_VARIABLE _fit_err)
if(NOT _fit_rc EQUAL 0 OR NOT _fit_out MATCHES "PLAN fits=1 mode=full gpu=simulated")
  message(FATAL_ERROR "valid geometry must fit without opening Metal (exit ${_fit_rc}): ${_fit_out}${_fit_err}")
endif()

execute_process(
  COMMAND "${STRATA}" --plan-only --geometry "${_geometry}" --plan-w-mib 65536 --plan-ram-mib 65536
          --spec 4
  RESULT_VARIABLE _missing_mtp_rc OUTPUT_VARIABLE _missing_mtp_out ERROR_VARIABLE _missing_mtp_err)
if(NOT _missing_mtp_rc EQUAL 2 OR
   NOT _missing_mtp_err MATCHES "requires geometry that includes a validated --mtp runtime")
  message(FATAL_ERROR "speculation without MTP geometry must return argument error 2: "
                      "${_missing_mtp_out}${_missing_mtp_err}")
endif()

execute_process(
  COMMAND "${STRATA}" --plan-only --geometry "${_geometry}" --plan-w-mib 1 --plan-ram-mib 65536
  RESULT_VARIABLE _no_fit_rc OUTPUT_VARIABLE _no_fit_out ERROR_VARIABLE _no_fit_err)
if(NOT _no_fit_rc EQUAL 3 OR NOT _no_fit_out MATCHES "PLAN fits=0")
  message(FATAL_ERROR "an insufficient working set must return a no-fit plan (exit ${_no_fit_rc}): "
                      "${_no_fit_out}${_no_fit_err}")
endif()

file(READ "${_geometry}" _unsupported_geometry_text)
string(REPLACE "q4_0" "unknown_quant" _unsupported_geometry_text "${_unsupported_geometry_text}")
set(_unsupported_geometry "${TEST_DIR}/unsupported-type.geometry")
file(WRITE "${_unsupported_geometry}" "${_unsupported_geometry_text}")
execute_process(
  COMMAND "${STRATA}" --plan-only --geometry "${_unsupported_geometry}"
          --plan-w-mib 65536 --plan-ram-mib 65536
  RESULT_VARIABLE _unsupported_rc OUTPUT_VARIABLE _unsupported_out ERROR_VARIABLE _unsupported_err)
if(NOT _unsupported_rc EQUAL 3 OR
   NOT _unsupported_out MATCHES "PLAN fits=0" OR
   NOT _unsupported_out MATCHES "reason=unsupported_type_unknown_quant")
  message(FATAL_ERROR "an unsupported tensor type must produce a no-fit PLAN (exit ${_unsupported_rc}): "
                      "${_unsupported_out}${_unsupported_err}")
endif()

set(_unknown_version "${TEST_DIR}/unknown-version.geometry")
file(WRITE "${_unknown_version}" "strata-geometry 2\n")
execute_process(
  COMMAND "${STRATA}" --plan-only --geometry "${_unknown_version}"
          --plan-w-mib 65536 --plan-ram-mib 65536
  RESULT_VARIABLE _version_rc OUTPUT_VARIABLE _version_out ERROR_VARIABLE _version_err)
if(NOT _version_rc EQUAL 2 OR
   NOT _version_err MATCHES "unsupported geometry file version 2")
  message(FATAL_ERROR "an unknown geometry version must return exit 2 with its version (exit ${_version_rc}): "
                      "${_version_out}${_version_err}")
endif()

execute_process(
  COMMAND "${STRATA}" --plan-only --geometry "${_geometry}" --plan-w-mib 65536 --plan-ram-mib 65536
          --prefill auto:0
  RESULT_VARIABLE _prefill_rc OUTPUT_VARIABLE _prefill_out ERROR_VARIABLE _prefill_err)
if(NOT _prefill_rc EQUAL 2 OR NOT _prefill_err MATCHES "at least 1024")
  message(FATAL_ERROR "an invalid auto-prefill cap must return argument error 2 (exit ${_prefill_rc}): "
                      "${_prefill_out}${_prefill_err}")
endif()

execute_process(
  COMMAND "${STRATA}" --plan-only --geometry "${_geometry}" --plan-w-mib 65536 --plan-ram-mib 65536
          --max-context 262145
  RESULT_VARIABLE _context_rc OUTPUT_VARIABLE _context_out ERROR_VARIABLE _context_err)
if(NOT _context_rc EQUAL 2 OR NOT _context_err MATCHES "out of range")
  message(FATAL_ERROR "an out-of-range context must return argument error 2 (exit ${_context_rc}): "
                      "${_context_out}${_context_err}")
endif()

execute_process(
  COMMAND "${STRATA}" --plan-only --geometry "${_geometry}" --plan-w-mib 65536 --plan-ram-mib 65536
          --metal-capture "${TEST_DIR}/unused-trace"
  RESULT_VARIABLE _capture_rc OUTPUT_VARIABLE _capture_out ERROR_VARIABLE _capture_err)
if(NOT _capture_rc EQUAL 2 OR NOT _capture_err MATCHES "--metal-capture cannot be combined with --plan-only")
  message(FATAL_ERROR "plan-only must reject a capture request (exit ${_capture_rc}): "
                      "${_capture_out}${_capture_err}")
endif()
