if(NOT DEFINED STRATA)
  message(FATAL_ERROR "STRATA is required")
endif()

execute_process(COMMAND "${STRATA}" RESULT_VARIABLE _empty_rc OUTPUT_VARIABLE _empty_out ERROR_VARIABLE _empty_err)
if(NOT _empty_rc EQUAL 2 OR NOT _empty_err MATCHES "choose --self-test")
  message(FATAL_ERROR "empty invocation must fail with guidance (exit ${_empty_rc}): ${_empty_err}${_empty_out}")
endif()

execute_process(COMMAND "${STRATA}" --storage-check
                RESULT_VARIABLE _storage_missing_native_rc
                OUTPUT_VARIABLE _storage_missing_native_out
                ERROR_VARIABLE _storage_missing_native_err)
if(NOT _storage_missing_native_rc EQUAL 2 OR
   NOT _storage_missing_native_err MATCHES "needs --native")
  message(FATAL_ERROR "storage check without a model must fail during argument parsing (exit "
                      "${_storage_missing_native_rc}): "
                      "${_storage_missing_native_err}${_storage_missing_native_out}")
endif()

execute_process(COMMAND "${STRATA}" --storage-check --native unavailable.gguf --plan-only
                RESULT_VARIABLE _storage_plan_conflict_rc
                OUTPUT_VARIABLE _storage_plan_conflict_out
                ERROR_VARIABLE _storage_plan_conflict_err)
if(NOT _storage_plan_conflict_rc EQUAL 2 OR
   NOT _storage_plan_conflict_err MATCHES "--storage-check cannot be combined")
  message(FATAL_ERROR "storage check must reject the light plan mode before opening Metal (exit "
                      "${_storage_plan_conflict_rc}): "
                      "${_storage_plan_conflict_err}${_storage_plan_conflict_out}")
endif()

execute_process(COMMAND "${STRATA}" --storage-check --native unavailable.gguf --serve
                RESULT_VARIABLE _storage_serve_conflict_rc
                OUTPUT_VARIABLE _storage_serve_conflict_out
                ERROR_VARIABLE _storage_serve_conflict_err)
if(NOT _storage_serve_conflict_rc EQUAL 2 OR
   NOT _storage_serve_conflict_err MATCHES "--storage-check cannot be combined with --serve")
  message(FATAL_ERROR "storage check must reject inference options before opening Metal (exit "
                      "${_storage_serve_conflict_rc}): "
                      "${_storage_serve_conflict_err}${_storage_serve_conflict_out}")
endif()

execute_process(COMMAND "${STRATA}" --list-devices --plan-only --native unavailable.gguf
                RESULT_VARIABLE _mixed_rc OUTPUT_VARIABLE _mixed_out ERROR_VARIABLE _mixed_err)
if(NOT _mixed_rc EQUAL 2 OR NOT _mixed_err MATCHES "--plan-only cannot be combined with --list-devices")
  message(FATAL_ERROR "list-devices and plan-only must reject this incompatible mode combination (exit ${_mixed_rc}): "
                      "${_mixed_err}${_mixed_out}")
endif()

execute_process(COMMAND "${STRATA}" --list-devices --self-test
                RESULT_VARIABLE _conflict_rc OUTPUT_VARIABLE _conflict_out ERROR_VARIABLE _conflict_err)
if(NOT _conflict_rc EQUAL 2 OR NOT _conflict_err MATCHES "cannot be combined")
  message(FATAL_ERROR "list-devices and self-test must conflict (exit ${_conflict_rc}): "
                      "${_conflict_err}${_conflict_out}")
endif()

execute_process(COMMAND "${STRATA}" --list-devices --metal-capture trace.gputrace
                RESULT_VARIABLE _capture_rc OUTPUT_VARIABLE _capture_out ERROR_VARIABLE _capture_err)
if(NOT _capture_rc EQUAL 2 OR NOT _capture_err MATCHES "--metal-capture cannot be combined with --list-devices")
  message(FATAL_ERROR "list-devices must reject an explicit capture request (exit ${_capture_rc}): "
                      "${_capture_err}${_capture_out}")
endif()
