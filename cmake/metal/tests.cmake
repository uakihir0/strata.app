# The Mac tests and the test labels (docs/mac/08-build-and-distribution.md, section 6).
#
#   host    no GPU needed: runs on every CI runner, including the paravirtualized GPU of a hosted Mac
#   metal   needs a real Apple GPU (M1 or newer)
#   model   needs the model files (STRATA_TEST_MODELS); run on the reference Mac only
#
# strata_metal_test() labels a test; every other test in the root folder, upstream ones included, is labelled
# host at the end of the root CMakeLists.txt.

function(_strata_label_rest)
  get_property(_tests DIRECTORY ${CMAKE_SOURCE_DIR} PROPERTY TESTS)
  foreach(_t IN LISTS _tests)
    get_test_property(${_t} LABELS _l)
    if(NOT _l)
      set_tests_properties(${_t} PROPERTIES LABELS host)
    endif()
  endforeach()
endfunction()
cmake_language(DEFER DIRECTORY ${CMAKE_SOURCE_DIR} CALL _strata_label_rest)

if(NOT STRATA_BUILD_TESTS)
  return()
endif()
enable_testing()

# strata_metal_test(<name> <host|metal|model> <command> [args...])
function(strata_metal_test name label)
  if(NOT label MATCHES "^(host|metal|model)$")
    message(FATAL_ERROR "strata_metal_test(${name}): label must be host, metal or model, not '${label}'")
  endif()
  add_test(NAME ${name} COMMAND ${ARGN})
  set_tests_properties(${name} PROPERTIES LABELS ${label})
endfunction()

# ---- ggml's CPU backend as the reference for the dequantizers and the kernels (section 6.1) ----
# Built from the pinned llama.cpp with nothing but the scalar and NEON code: no Metal, BLAS, Accelerate, OpenMP or
# -mcpu=native, so its results do not depend on the Mac that runs the test.
if(STRATA_GGML_DIR)
  set(_strata_llama ${STRATA_GGML_DIR})
else()
  file(STRINGS ${CMAKE_SOURCE_DIR}/third_party/ggml/VERSION.txt _pin LIMIT_COUNT 1)
  string(REGEX MATCH "^[0-9a-f]+" _pin "${_pin}")
  include(FetchContent)
  FetchContent_Declare(strata_llamacpp
    GIT_REPOSITORY https://github.com/ggml-org/llama.cpp.git
    GIT_TAG ${_pin}
    GIT_SHALLOW FALSE
    SOURCE_SUBDIR strata-populate-only)   # a folder that does not exist: download only, add nothing
  FetchContent_MakeAvailable(strata_llamacpp)
  set(_strata_llama ${strata_llamacpp_SOURCE_DIR})
endif()
enable_language(C)
foreach(_o GGML_METAL GGML_CUDA GGML_HIP GGML_BLAS GGML_ACCELERATE GGML_OPENMP GGML_NATIVE GGML_CCACHE
           GGML_BACKEND_DL GGML_BUILD_TESTS
           GGML_BUILD_EXAMPLES BUILD_SHARED_LIBS)
  set(${_o} OFF CACHE BOOL "" FORCE)
endforeach()
set(GGML_STATIC ON CACHE BOOL "" FORCE)
add_subdirectory(${_strata_llama}/ggml ${CMAKE_BINARY_DIR}/ggml-oracle EXCLUDE_FROM_ALL)
add_library(strata_ggml_oracle INTERFACE)
target_link_libraries(strata_ggml_oracle INTERFACE ggml-base ggml-cpu)
# both are needed (ggml-cpu alone loses ggml's include folder) and ld warns that ggml-base appears twice
target_link_options(strata_ggml_oracle INTERFACE LINKER:-no_warn_duplicate_libraries)

# ---- the Mac twins of upstream tests that live in the GPU block of the root CMakeLists.txt ----
add_executable(platform_memory_test src/platform/memory_test.cpp)
target_link_libraries(platform_memory_test PRIVATE strata_metal_platform strata_metal_options)
strata_metal_test(platform_memory_test host platform_memory_test)

add_executable(ple_reader_test src/ngram/ple_reader_test.cpp)
target_link_libraries(ple_reader_test PRIVATE strata_metal_platform strata_metal_options)
strata_metal_test(ple_reader_selftest host ple_reader_test --selftest)

add_executable(ple_oracle_model_test tests/metal/ple_oracle_model_test.cpp)
target_include_directories(ple_oracle_model_test PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(ple_oracle_model_test PRIVATE strata_metal_platform strata_metal_options)
strata_metal_test(ple_oracle_layout_test host ple_oracle_model_test --selftest)
strata_metal_test(ple_oracle_model_test model ple_oracle_model_test --models)
set_tests_properties(ple_oracle_model_test PROPERTIES TIMEOUT 1800)

# ---- tests/metal ----
add_executable(dequant_oracle_test tests/metal/dequant_oracle_test.cpp)
target_link_libraries(dequant_oracle_test PRIVATE strata_artifact strata_ggml_oracle strata_metal_options)
strata_metal_test(dequant_oracle_test host dequant_oracle_test)

add_executable(codebooks_test tests/metal/codebooks_test.cpp)
target_include_directories(codebooks_test PRIVATE ${CMAKE_SOURCE_DIR}/src/metal/kernels/common
                           ${_strata_llama}/ggml/src)
strata_metal_test(codebooks host codebooks_test)

add_executable(metal_manifest_test tests/metal/metal_manifest_test.cpp)
target_include_directories(metal_manifest_test PRIVATE ${STRATA_METAL_GENERATED_DIR})
target_link_libraries(metal_manifest_test PRIVATE strata_metal_host strata_metal_options)
set_source_files_properties(tests/metal/metal_manifest_test.cpp PROPERTIES OBJECT_DEPENDS ${STRATA_METALLIB_SHA256_INC})
add_dependencies(metal_manifest_test strata_metallib)
strata_metal_test(metal_manifest_test host metal_manifest_test ${STRATA_METALLIB})

add_executable(metal_context_test tests/metal/metal_context_test.cpp)
target_include_directories(metal_context_test PRIVATE ${STRATA_METAL_GENERATED_DIR})
target_link_libraries(metal_context_test PRIVATE strata_metal_host strata_metal_options)
set_source_files_properties(tests/metal/metal_context_test.cpp PROPERTIES OBJECT_DEPENDS ${STRATA_METALLIB_SHA256_INC})
add_dependencies(metal_context_test strata_metallib)
strata_metal_test(metal_context_test metal metal_context_test ${STRATA_METALLIB} ${CMAKE_BINARY_DIR}/metal/test-cache)

add_executable(metal_math_test tests/metal/metal_math_test.mm)
target_link_libraries(metal_math_test PRIVATE strata_metal_host strata_metal_options)
strata_metal_test(metal_math_test metal metal_math_test)
set_tests_properties(metal_math_test PROPERTIES TIMEOUT 180)

add_library(strata_metal_oracles STATIC
  tests/metal/oracles/common.cpp
  tests/metal/oracles/q8_1.cpp
  tests/metal/oracles/q8_0_mv.cpp
  tests/metal/oracles/q4_0_mv.cpp
  tests/metal/oracles/q5_0_mv.cpp
  tests/metal/oracles/q2_0_mv.cpp
  tests/metal/oracles/iq4_nl_mv.cpp
  tests/metal/oracles/iq4_xs_mv.cpp
  tests/metal/oracles/iq2_xxs_mv.cpp
  tests/metal/oracles/iq2_xs_mv.cpp
  tests/metal/oracles/iq2_s_mv.cpp
  tests/metal/oracles/iq3_xxs_mv.cpp
  tests/metal/oracles/iq3_s_mv.cpp
  tests/metal/oracles/iq1_m_mv.cpp
  tests/metal/oracles/moe_route.cpp
  tests/metal/oracles/moe_plan.cpp
  tests/metal/oracles/moe_swiglu.cpp
  tests/metal/oracles/moe_gu_q2_0.cpp
  tests/metal/oracles/moe_gu_iq1_m.cpp
  tests/metal/oracles/moe_gu_iq2_xxs.cpp
  tests/metal/oracles/moe_gu_iq2_xs.cpp
  tests/metal/oracles/moe_gu_iq2_s.cpp
  tests/metal/oracles/moe_gu_iq3_xxs.cpp
  tests/metal/oracles/moe_gu_iq3_s.cpp
  tests/metal/oracles/moe_gu_iq4_xs.cpp
  tests/metal/oracles/moe_down_q2_0.cpp
  tests/metal/oracles/moe_down_iq4_nl.cpp
  tests/metal/oracles/moe_combine.cpp
  tests/metal/oracles/moe_shared_expert.cpp
  tests/metal/oracles/q4_k_mv.cpp
  tests/metal/oracles/q5_k_mv.cpp
  tests/metal/oracles/q3_k_mv.cpp
  tests/metal/oracles/q6_k_mv.cpp
  tests/metal/oracles/bf16_mv.cpp
  tests/metal/oracles/gdn.cpp
  tests/metal/oracles/attention.cpp
  tests/metal/oracles/qsa.cpp
  tests/metal/synth.cpp)
target_include_directories(strata_metal_oracles PUBLIC ${CMAKE_SOURCE_DIR}/tests/metal
                           PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(strata_metal_oracles PRIVATE strata_metal_options)
target_compile_options(strata_metal_oracles PRIVATE -ffp-contract=off)

add_library(strata_metal_twins STATIC tests/metal/twin.cpp)
set_source_files_properties(tests/metal/twin.cpp PROPERTIES LANGUAGE OBJCXX)
target_include_directories(strata_metal_twins PUBLIC ${CMAKE_SOURCE_DIR}/tests/metal)
target_link_libraries(strata_metal_twins PUBLIC strata_metal_host strata_metal_oracles
                      PRIVATE strata_metal_options)

add_executable(metal_twin_selftest tests/metal/metal_twin_selftest.mm)
target_include_directories(metal_twin_selftest PRIVATE ${STRATA_METAL_GENERATED_DIR})
target_link_libraries(metal_twin_selftest PRIVATE strata_metal_twins strata_metal_options)
set_source_files_properties(tests/metal/metal_twin_selftest.mm
                            PROPERTIES OBJECT_DEPENDS ${STRATA_METALLIB_SHA256_INC})
add_dependencies(metal_twin_selftest strata_metallib)
strata_metal_test(metal_twin_selftest metal metal_twin_selftest ${STRATA_METALLIB})
set_tests_properties(metal_twin_selftest PROPERTIES TIMEOUT 120)

add_executable(metal_mv_test tests/metal/metal_mv_test.mm)
target_include_directories(metal_mv_test PRIVATE ${STRATA_METAL_GENERATED_DIR})
target_link_libraries(metal_mv_test PRIVATE
  strata_metal_twins strata_metal_engine strata_metal_options)
set_source_files_properties(tests/metal/metal_mv_test.mm
                            PROPERTIES OBJECT_DEPENDS ${STRATA_METALLIB_SHA256_INC})
add_dependencies(metal_mv_test strata_metallib)
strata_metal_test(metal_mv_test metal metal_mv_test ${STRATA_METALLIB})
set_tests_properties(metal_mv_test PROPERTIES
  TIMEOUT 120
  ENVIRONMENT "STRATA_METAL_DISPATCH=serial")
strata_metal_test(metal_mv_test_concurrent metal
                  metal_mv_test ${STRATA_METALLIB})
set_tests_properties(metal_mv_test_concurrent PROPERTIES
  TIMEOUT 120
  ENVIRONMENT "STRATA_METAL_DISPATCH=concurrent")

add_executable(metal_moe_test tests/metal/metal_moe_test.mm)
target_include_directories(metal_moe_test PRIVATE ${STRATA_METAL_GENERATED_DIR})
target_link_libraries(metal_moe_test PRIVATE
  strata_metal_twins strata_metal_engine strata_metal_options)
set_source_files_properties(tests/metal/metal_moe_test.mm
                            PROPERTIES OBJECT_DEPENDS ${STRATA_METALLIB_SHA256_INC})
add_dependencies(metal_moe_test strata_metallib)
strata_metal_test(metal_moe_test metal metal_moe_test ${STRATA_METALLIB})
set_tests_properties(metal_moe_test PROPERTIES
  TIMEOUT 180
  ENVIRONMENT "STRATA_METAL_DISPATCH=serial")
strata_metal_test(metal_moe_test_concurrent metal
                  metal_moe_test ${STRATA_METALLIB} --input-set fixture)
set_tests_properties(metal_moe_test_concurrent PROPERTIES
  TIMEOUT 180
  ENVIRONMENT "STRATA_METAL_DISPATCH=concurrent")

add_executable(metal_sampler_parity_test tests/metal/metal_sampler_parity_test.mm)
target_include_directories(metal_sampler_parity_test PRIVATE ${STRATA_METAL_GENERATED_DIR})
target_link_libraries(metal_sampler_parity_test PRIVATE
  strata_metal_twins strata_metal_engine strata_metal_options)
set_source_files_properties(tests/metal/metal_sampler_parity_test.mm
                            PROPERTIES OBJECT_DEPENDS ${STRATA_METALLIB_SHA256_INC})
add_dependencies(metal_sampler_parity_test strata_metallib)
strata_metal_test(metal_sampler_parity_test metal
                  metal_sampler_parity_test ${STRATA_METALLIB})
strata_metal_test(metal_sampler_parity_test_concurrent metal
                  metal_sampler_parity_test ${STRATA_METALLIB})
set_tests_properties(metal_sampler_parity_test_concurrent PROPERTIES
  TIMEOUT 180
  ENVIRONMENT "STRATA_METAL_DISPATCH=concurrent")
strata_metal_test(samp_coupled_eq_host metal
                  metal_sampler_parity_test ${STRATA_METALLIB} --coupled)
set_tests_properties(metal_sampler_parity_test samp_coupled_eq_host
                     PROPERTIES TIMEOUT 180)

add_executable(metal_embed_test tests/metal/metal_embed_test.mm)
target_include_directories(metal_embed_test PRIVATE ${STRATA_METAL_GENERATED_DIR})
target_link_libraries(metal_embed_test PRIVATE
  strata_metal_engine strata_metal_twins strata_ggml_oracle strata_metal_options)
set_source_files_properties(tests/metal/metal_embed_test.mm
                            PROPERTIES OBJECT_DEPENDS ${STRATA_METALLIB_SHA256_INC})
add_dependencies(metal_embed_test strata_metallib)
strata_metal_test(metal_embed_test metal metal_embed_test ${STRATA_METALLIB})
set_tests_properties(metal_embed_test PROPERTIES
  TIMEOUT 120
  ENVIRONMENT "STRATA_METAL_DISPATCH=serial")
strata_metal_test(metal_embed_test_concurrent metal
                  metal_embed_test ${STRATA_METALLIB})
set_tests_properties(metal_embed_test_concurrent PROPERTIES
  TIMEOUT 120
  ENVIRONMENT "STRATA_METAL_DISPATCH=concurrent")

add_executable(metal_prefill_test tests/metal/metal_prefill_test.mm)
target_include_directories(metal_prefill_test PRIVATE ${STRATA_METAL_GENERATED_DIR})
target_link_libraries(metal_prefill_test PRIVATE
  strata_metal_engine strata_metal_twins strata_metal_options)
set_source_files_properties(tests/metal/metal_prefill_test.mm
                            PROPERTIES OBJECT_DEPENDS ${STRATA_METALLIB_SHA256_INC})
add_dependencies(metal_prefill_test strata_metallib)
strata_metal_test(metal_prefill_test metal metal_prefill_test ${STRATA_METALLIB})
set_tests_properties(metal_prefill_test PROPERTIES
  TIMEOUT 120
  ENVIRONMENT "STRATA_METAL_DISPATCH=serial")
strata_metal_test(metal_prefill_test_concurrent metal
                  metal_prefill_test ${STRATA_METALLIB})
set_tests_properties(metal_prefill_test_concurrent PROPERTIES
  TIMEOUT 120
  ENVIRONMENT "STRATA_METAL_DISPATCH=concurrent")

add_executable(metal_ple_test tests/metal/metal_ple_test.mm)
target_include_directories(metal_ple_test PRIVATE ${STRATA_METAL_GENERATED_DIR})
target_link_libraries(metal_ple_test PRIVATE
  strata_metal_twins strata_metal_engine strata_metal_options)
set_source_files_properties(tests/metal/metal_ple_test.mm
                            PROPERTIES OBJECT_DEPENDS ${STRATA_METALLIB_SHA256_INC})
add_dependencies(metal_ple_test strata_metallib)
strata_metal_test(metal_ple_test metal metal_ple_test ${STRATA_METALLIB})
set_tests_properties(metal_ple_test PROPERTIES
  TIMEOUT 120
  ENVIRONMENT "STRATA_METAL_DISPATCH=serial")
strata_metal_test(metal_ple_test_concurrent metal
                  metal_ple_test ${STRATA_METALLIB})
set_tests_properties(metal_ple_test_concurrent PROPERTIES
  TIMEOUT 120
  ENVIRONMENT "STRATA_METAL_DISPATCH=concurrent")

add_executable(metal_gr_test tests/metal/metal_gr_test.mm)
target_include_directories(metal_gr_test PRIVATE ${STRATA_METAL_GENERATED_DIR})
target_link_libraries(metal_gr_test PRIVATE
  strata_metal_twins strata_metal_engine strata_metal_options)
set_source_files_properties(tests/metal/metal_gr_test.mm
                            PROPERTIES OBJECT_DEPENDS ${STRATA_METALLIB_SHA256_INC})
add_dependencies(metal_gr_test strata_metallib)
strata_metal_test(metal_gr_test metal metal_gr_test ${STRATA_METALLIB})
set_tests_properties(metal_gr_test PROPERTIES
  TIMEOUT 120
  ENVIRONMENT "STRATA_METAL_DISPATCH=serial")
strata_metal_test(metal_gr_test_concurrent metal
                  metal_gr_test ${STRATA_METALLIB})
set_tests_properties(metal_gr_test_concurrent PROPERTIES
  TIMEOUT 120
  ENVIRONMENT "STRATA_METAL_DISPATCH=concurrent")

add_executable(metal_gdn_test tests/metal/metal_gdn_test.mm)
target_include_directories(metal_gdn_test PRIVATE ${STRATA_METAL_GENERATED_DIR})
target_link_libraries(metal_gdn_test PRIVATE
  strata_metal_twins strata_metal_engine strata_metal_options)
set_source_files_properties(tests/metal/metal_gdn_test.mm
                            PROPERTIES OBJECT_DEPENDS ${STRATA_METALLIB_SHA256_INC})
add_dependencies(metal_gdn_test strata_metallib)
strata_metal_test(metal_gdn_test metal metal_gdn_test ${STRATA_METALLIB})
set_tests_properties(metal_gdn_test PROPERTIES TIMEOUT 300)
strata_metal_test(metal_gdn_test_concurrent metal
                  metal_gdn_test ${STRATA_METALLIB} --input-set fixture)
set_tests_properties(metal_gdn_test_concurrent PROPERTIES
  TIMEOUT 300
  ENVIRONMENT "STRATA_METAL_DISPATCH=concurrent")
strata_metal_test(metal_window_gdn_commit_batch_test metal
                  metal_gdn_test ${STRATA_METALLIB}
                  --window-gdn-commit-batch)
strata_metal_test(metal_window_gdn_commit_batch_test_concurrent metal
                  metal_gdn_test ${STRATA_METALLIB}
                  --window-gdn-commit-batch)
set_tests_properties(metal_window_gdn_commit_batch_test_concurrent PROPERTIES
  TIMEOUT 300
  ENVIRONMENT "STRATA_METAL_DISPATCH=concurrent")

add_executable(metal_qsa_test tests/metal/metal_qsa_test.mm)
target_include_directories(metal_qsa_test PRIVATE ${STRATA_METAL_GENERATED_DIR})
target_link_libraries(metal_qsa_test PRIVATE
  strata_metal_twins strata_metal_engine strata_metal_options)
set_source_files_properties(tests/metal/metal_qsa_test.mm
                            PROPERTIES OBJECT_DEPENDS ${STRATA_METALLIB_SHA256_INC})
add_dependencies(metal_qsa_test strata_metallib)
strata_metal_test(metal_qsa_test metal metal_qsa_test ${STRATA_METALLIB})
set_tests_properties(metal_qsa_test PROPERTIES
  TIMEOUT 300
  ENVIRONMENT "STRATA_METAL_DISPATCH=serial")
strata_metal_test(metal_qsa_test_concurrent metal
                  metal_qsa_test ${STRATA_METALLIB})
set_tests_properties(metal_qsa_test_concurrent PROPERTIES
  TIMEOUT 300
  ENVIRONMENT "STRATA_METAL_DISPATCH=concurrent")
strata_metal_test(metal_qsa_indexer_commit_batch_test metal
                  metal_qsa_test ${STRATA_METALLIB}
                  --indexer-commit-batch)
set_tests_properties(metal_qsa_indexer_commit_batch_test PROPERTIES
  TIMEOUT 300
  ENVIRONMENT "STRATA_METAL_DISPATCH=serial")
strata_metal_test(metal_qsa_indexer_commit_batch_test_concurrent metal
                  metal_qsa_test ${STRATA_METALLIB}
                  --indexer-commit-batch)
set_tests_properties(metal_qsa_indexer_commit_batch_test_concurrent PROPERTIES
  TIMEOUT 300
  ENVIRONMENT "STRATA_METAL_DISPATCH=concurrent")

add_executable(metal_attn_test tests/metal/metal_attn_test.mm)
target_include_directories(metal_attn_test PRIVATE ${STRATA_METAL_GENERATED_DIR})
target_link_libraries(metal_attn_test PRIVATE
  strata_metal_twins strata_metal_engine strata_metal_options)
set_source_files_properties(tests/metal/metal_attn_test.mm
                            PROPERTIES OBJECT_DEPENDS ${STRATA_METALLIB_SHA256_INC})
add_dependencies(metal_attn_test strata_metallib)
strata_metal_test(metal_attn_test metal metal_attn_test ${STRATA_METALLIB})
set_tests_properties(metal_attn_test PROPERTIES
  TIMEOUT 300
  ENVIRONMENT "STRATA_METAL_DISPATCH=serial")
strata_metal_test(metal_attn_test_concurrent metal
                  metal_attn_test ${STRATA_METALLIB})
set_tests_properties(metal_attn_test_concurrent PROPERTIES
  TIMEOUT 300
  ENVIRONMENT "STRATA_METAL_DISPATCH=concurrent")

add_executable(metal_qsa_selection_test
               tests/metal/metal_qsa_selection_test.mm)
target_include_directories(metal_qsa_selection_test PRIVATE
  ${STRATA_METAL_GENERATED_DIR})
target_link_libraries(metal_qsa_selection_test PRIVATE
  strata_metal_twins strata_metal_engine strata_metal_options)
set_source_files_properties(tests/metal/metal_qsa_selection_test.mm
                            PROPERTIES OBJECT_DEPENDS ${STRATA_METALLIB_SHA256_INC})
add_dependencies(metal_qsa_selection_test strata_metallib)
strata_metal_test(metal_qsa_selection_test metal metal_qsa_selection_test
                  ${STRATA_METALLIB})
set_tests_properties(metal_qsa_selection_test PROPERTIES
  TIMEOUT 300
  ENVIRONMENT "STRATA_METAL_DISPATCH=serial")
strata_metal_test(metal_qsa_selection_test_concurrent metal
                  metal_qsa_selection_test ${STRATA_METALLIB})
set_tests_properties(metal_qsa_selection_test_concurrent PROPERTIES
  TIMEOUT 300
  ENVIRONMENT "STRATA_METAL_DISPATCH=concurrent")

add_executable(sampler_tail_test tests/metal/sampler_tail_test.cpp
               src/metal/engine/sampler_tail.cpp)
target_include_directories(sampler_tail_test PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(sampler_tail_test PRIVATE strata_metal_options)
strata_metal_test(sampler_tail_test host sampler_tail_test)

strata_metal_test(metal_moe_real_inputs_unavailable host metal_moe_test
                  ${STRATA_METALLIB} --input-set real)
set_tests_properties(metal_moe_real_inputs_unavailable
                     PROPERTIES SKIP_RETURN_CODE 77)

add_executable(metal_fault_test tests/metal/metal_fault_test.mm)
target_include_directories(metal_fault_test PRIVATE ${STRATA_METAL_GENERATED_DIR})
target_link_libraries(metal_fault_test PRIVATE strata_metal_host strata_metal_options)
set_source_files_properties(tests/metal/metal_fault_test.mm PROPERTIES OBJECT_DEPENDS ${STRATA_METALLIB_SHA256_INC})
add_dependencies(metal_fault_test strata_metallib)
strata_metal_test(metal_fault_test metal metal_fault_test ${STRATA_METALLIB})
set_tests_properties(metal_fault_test PROPERTIES
  TIMEOUT 120
  ENVIRONMENT "MTL_SHADER_VALIDATION=1;MTL_DEBUG_LAYER=1;STRATA_METAL_ERRORS=1")

add_executable(metal_buffers_test tests/metal/metal_buffers_test.mm)
target_link_libraries(metal_buffers_test PRIVATE strata_metal_host strata_metal_options)
strata_metal_test(metal_buffers_test metal metal_buffers_test)

add_executable(metal_events_test tests/metal/metal_events_test.mm)
target_link_libraries(metal_events_test PRIVATE strata_metal_host strata_metal_options)
strata_metal_test(metal_events_test metal metal_events_test)
set_tests_properties(metal_events_test PROPERTIES TIMEOUT 30)

add_executable(metal_encoder_test tests/metal/metal_encoder_test.mm)
target_include_directories(metal_encoder_test PRIVATE ${STRATA_METAL_GENERATED_DIR})
target_link_libraries(metal_encoder_test PRIVATE strata_metal_host strata_metal_options)
set_source_files_properties(tests/metal/metal_encoder_test.mm
                            PROPERTIES OBJECT_DEPENDS ${STRATA_METALLIB_SHA256_INC})
add_dependencies(metal_encoder_test strata_metallib)
strata_metal_test(metal_encoder_test metal metal_encoder_test ${STRATA_METALLIB})
set_tests_properties(metal_encoder_test PROPERTIES TIMEOUT 120)

# The test registry must match both the documented catalogue and CTest's configured list.
find_package(Python3 3.10 COMPONENTS Interpreter REQUIRED)
strata_metal_test(test_map host ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tools/metal/test_map.py)
set_tests_properties(test_map PROPERTIES ENVIRONMENT "STRATA_BUILD_DIR=${CMAKE_BINARY_DIR}")

# the shipped engine: identity and the hash it embeds, on the binary itself (host: --version opens no device)
strata_metal_test(strata_version host ${CMAKE_COMMAND} -DSTRATA=$<TARGET_FILE:strata> -DMETALLIB=${STRATA_METALLIB}
                  -DVERSION=${PROJECT_VERSION} -DMETAL_REV=${STRATA_METAL_REV}
                  -P ${CMAKE_CURRENT_LIST_DIR}/check_engine.cmake)

strata_metal_test(metal_list_devices_test metal ${CMAKE_COMMAND} -DSTRATA=$<TARGET_FILE:strata>
                  -DBUILD_DIR=${CMAKE_BINARY_DIR}
                  -P ${CMAKE_SOURCE_DIR}/tests/metal/check_list_devices.cmake)
strata_metal_test(metal_self_test metal $<TARGET_FILE:strata> --self-test)
set_tests_properties(metal_self_test PROPERTIES ENVIRONMENT "STRATA_METAL_ARCHIVE=0" TIMEOUT 120)

add_executable(options_dispositions_test tests/metal/options_dispositions_test.cpp)
target_link_libraries(options_dispositions_test PRIVATE strata_metal_cli_options)
strata_metal_test(options_dispositions_test host options_dispositions_test)
strata_metal_test(metal_cli_modes_test host ${CMAKE_COMMAND} -DSTRATA=$<TARGET_FILE:strata>
                  -P ${CMAKE_SOURCE_DIR}/tests/metal/check_cli_modes.cmake)
strata_metal_test(metal_geometry_plan_cli_test host ${CMAKE_COMMAND} -DSTRATA=$<TARGET_FILE:strata>
                  -DTEST_DIR=${CMAKE_BINARY_DIR}/metal/geometry-plan-cli
                  -P ${CMAKE_SOURCE_DIR}/tests/metal/check_geometry_plan_cli.cmake)

add_executable(metal_plan_test tests/metal/metal_plan_test.cpp src/metal/engine/geometry.cpp
               src/metal/engine/expert_profile.cpp src/metal/engine/plan.cpp
               src/metal/engine/kv_layout.cpp src/metal/engine/mtp_kv_layout.cpp
               src/metal/engine/mtp_params_layout.cpp
               src/metal/engine/session_state_layout.cpp
               src/metal/engine/window_scratch_layout.cpp)
target_link_libraries(metal_plan_test PRIVATE strata_metal_options)
target_include_directories(metal_plan_test PRIVATE ${CMAKE_SOURCE_DIR}/include)
strata_metal_test(metal_plan_test host metal_plan_test)

add_executable(metal_kv_layout_test tests/metal/metal_kv_layout_test.cpp
               src/metal/engine/kv_layout.cpp src/metal/engine/mtp_kv_layout.cpp)
target_link_libraries(metal_kv_layout_test PRIVATE strata_metal_options)
target_include_directories(metal_kv_layout_test PRIVATE ${CMAKE_SOURCE_DIR}/include)
strata_metal_test(metal_kv_layout_test host metal_kv_layout_test)

add_executable(metal_kv_store_test tests/metal/metal_kv_store_test.mm)
target_link_libraries(metal_kv_store_test PRIVATE strata_metal_engine strata_metal_options)
strata_metal_test(metal_kv_store_test metal metal_kv_store_test)

add_executable(metal_session_state_test tests/metal/metal_session_state_test.mm)
target_link_libraries(metal_session_state_test PRIVATE strata_metal_engine strata_metal_options)
strata_metal_test(metal_session_state_test metal metal_session_state_test)

add_executable(metal_window_scratch_storage_test
               tests/metal/metal_window_scratch_storage_test.mm)
target_link_libraries(metal_window_scratch_storage_test PRIVATE
                      strata_metal_engine strata_metal_options)
strata_metal_test(metal_window_scratch_storage_test metal
                  metal_window_scratch_storage_test)

add_executable(metal_mtp_kv_store_test tests/metal/metal_mtp_kv_store_test.mm)
target_link_libraries(metal_mtp_kv_store_test PRIVATE strata_metal_engine strata_metal_options)
strata_metal_test(metal_mtp_kv_store_test metal metal_mtp_kv_store_test)

add_executable(metal_mtp_test tests/metal/metal_mtp_test.mm)
target_include_directories(metal_mtp_test PRIVATE ${STRATA_METAL_GENERATED_DIR})
target_link_libraries(metal_mtp_test PRIVATE
  strata_metal_twins strata_metal_engine strata_metal_options)
set_source_files_properties(tests/metal/metal_mtp_test.mm
                            PROPERTIES OBJECT_DEPENDS ${STRATA_METALLIB_SHA256_INC})
add_dependencies(metal_mtp_test strata_metallib)
strata_metal_test(metal_mtp_test metal metal_mtp_test ${STRATA_METALLIB})
set_tests_properties(metal_mtp_test PROPERTIES TIMEOUT 120)

add_executable(metal_mtp_gr_test tests/metal/metal_mtp_gr_test.mm)
target_include_directories(metal_mtp_gr_test PRIVATE ${STRATA_METAL_GENERATED_DIR})
target_link_libraries(metal_mtp_gr_test PRIVATE
  strata_metal_twins strata_metal_engine strata_metal_options)
set_source_files_properties(tests/metal/metal_mtp_gr_test.mm
                            PROPERTIES OBJECT_DEPENDS ${STRATA_METALLIB_SHA256_INC})
add_dependencies(metal_mtp_gr_test strata_metallib)
strata_metal_test(metal_mtp_gr_test metal metal_mtp_gr_test ${STRATA_METALLIB})
set_tests_properties(metal_mtp_gr_test PROPERTIES TIMEOUT 120)
strata_metal_test(metal_mtp_gr_test_concurrent metal ${CMAKE_COMMAND} -E env
  STRATA_METAL_DISPATCH=concurrent $<TARGET_FILE:metal_mtp_gr_test>
  ${STRATA_METALLIB})
set_tests_properties(metal_mtp_gr_test_concurrent PROPERTIES TIMEOUT 120)

add_executable(metal_mtp_kv_projection_test
  tests/metal/metal_mtp_kv_projection_test.mm)
target_include_directories(metal_mtp_kv_projection_test PRIVATE
  ${STRATA_METAL_GENERATED_DIR})
target_link_libraries(metal_mtp_kv_projection_test PRIVATE
  strata_metal_twins strata_metal_engine strata_metal_options)
set_source_files_properties(tests/metal/metal_mtp_kv_projection_test.mm
                            PROPERTIES OBJECT_DEPENDS ${STRATA_METALLIB_SHA256_INC})
add_dependencies(metal_mtp_kv_projection_test strata_metallib)
strata_metal_test(metal_mtp_kv_projection_test metal
  metal_mtp_kv_projection_test ${STRATA_METALLIB})
set_tests_properties(metal_mtp_kv_projection_test PROPERTIES TIMEOUT 120)
strata_metal_test(metal_mtp_kv_projection_test_concurrent metal
  ${CMAKE_COMMAND} -E env STRATA_METAL_DISPATCH=concurrent
  $<TARGET_FILE:metal_mtp_kv_projection_test> ${STRATA_METALLIB})
set_tests_properties(metal_mtp_kv_projection_test_concurrent
                     PROPERTIES TIMEOUT 120)

add_executable(metal_mtp_kv_append_test
  tests/metal/metal_mtp_kv_append_test.mm)
target_include_directories(metal_mtp_kv_append_test PRIVATE
  ${STRATA_METAL_GENERATED_DIR})
target_link_libraries(metal_mtp_kv_append_test PRIVATE
  strata_metal_twins strata_metal_engine strata_metal_options)
set_source_files_properties(tests/metal/metal_mtp_kv_append_test.mm
                            PROPERTIES OBJECT_DEPENDS ${STRATA_METALLIB_SHA256_INC})
add_dependencies(metal_mtp_kv_append_test strata_metallib)
strata_metal_test(metal_mtp_kv_append_test metal
  metal_mtp_kv_append_test ${STRATA_METALLIB})
set_tests_properties(metal_mtp_kv_append_test PROPERTIES TIMEOUT 120)
strata_metal_test(metal_mtp_kv_append_test_concurrent metal
  ${CMAKE_COMMAND} -E env STRATA_METAL_DISPATCH=concurrent
  $<TARGET_FILE:metal_mtp_kv_append_test> ${STRATA_METALLIB})
set_tests_properties(metal_mtp_kv_append_test_concurrent
                     PROPERTIES TIMEOUT 120)

add_executable(metal_mtp_query_projection_test
  tests/metal/metal_mtp_query_projection_test.mm)
target_include_directories(metal_mtp_query_projection_test PRIVATE
  ${STRATA_METAL_GENERATED_DIR})
target_link_libraries(metal_mtp_query_projection_test PRIVATE
  strata_metal_twins strata_metal_engine strata_metal_options)
set_source_files_properties(tests/metal/metal_mtp_query_projection_test.mm
                            PROPERTIES OBJECT_DEPENDS ${STRATA_METALLIB_SHA256_INC})
add_dependencies(metal_mtp_query_projection_test strata_metallib)
strata_metal_test(metal_mtp_query_projection_test metal
  metal_mtp_query_projection_test ${STRATA_METALLIB})
set_tests_properties(metal_mtp_query_projection_test PROPERTIES TIMEOUT 180)
strata_metal_test(metal_mtp_query_projection_test_concurrent metal
  ${CMAKE_COMMAND} -E env STRATA_METAL_DISPATCH=concurrent
  $<TARGET_FILE:metal_mtp_query_projection_test> ${STRATA_METALLIB})
set_tests_properties(metal_mtp_query_projection_test_concurrent
                     PROPERTIES TIMEOUT 180)

add_executable(metal_mtp_query_norm_test
  tests/metal/metal_mtp_query_norm_test.mm)
target_include_directories(metal_mtp_query_norm_test PRIVATE
  ${STRATA_METAL_GENERATED_DIR})
target_link_libraries(metal_mtp_query_norm_test PRIVATE
  strata_metal_twins strata_metal_engine strata_metal_options)
set_source_files_properties(tests/metal/metal_mtp_query_norm_test.mm
                            PROPERTIES OBJECT_DEPENDS ${STRATA_METALLIB_SHA256_INC})
add_dependencies(metal_mtp_query_norm_test strata_metallib)
strata_metal_test(metal_mtp_query_norm_test metal
  metal_mtp_query_norm_test ${STRATA_METALLIB})
set_tests_properties(metal_mtp_query_norm_test PROPERTIES TIMEOUT 180)
strata_metal_test(metal_mtp_query_norm_test_concurrent metal
  ${CMAKE_COMMAND} -E env STRATA_METAL_DISPATCH=concurrent
  $<TARGET_FILE:metal_mtp_query_norm_test> ${STRATA_METALLIB})
set_tests_properties(metal_mtp_query_norm_test_concurrent
                     PROPERTIES TIMEOUT 180)

add_executable(metal_mtp_window_ids_test
  tests/metal/metal_mtp_window_ids_test.mm)
target_include_directories(metal_mtp_window_ids_test PRIVATE
  ${STRATA_METAL_GENERATED_DIR})
target_link_libraries(metal_mtp_window_ids_test PRIVATE
  strata_metal_twins strata_metal_engine strata_metal_options)
set_source_files_properties(tests/metal/metal_mtp_window_ids_test.mm
                            PROPERTIES OBJECT_DEPENDS ${STRATA_METALLIB_SHA256_INC})
add_dependencies(metal_mtp_window_ids_test strata_metallib)
strata_metal_test(metal_mtp_window_ids_test metal
  metal_mtp_window_ids_test ${STRATA_METALLIB})
set_tests_properties(metal_mtp_window_ids_test PROPERTIES TIMEOUT 180)
strata_metal_test(metal_mtp_window_ids_test_concurrent metal
  ${CMAKE_COMMAND} -E env STRATA_METAL_DISPATCH=concurrent
  $<TARGET_FILE:metal_mtp_window_ids_test> ${STRATA_METALLIB})
set_tests_properties(metal_mtp_window_ids_test_concurrent
                     PROPERTIES TIMEOUT 180)

add_executable(metal_weight_store_test tests/metal/metal_weight_store_test.cpp
               src/metal/engine/weights.cpp src/metal/engine/geometry.cpp
               src/metal/engine/window_plan.cpp src/metal/engine/mtp.cpp
               src/metal/engine/mtp_bind_layout.cpp)
target_link_libraries(metal_weight_store_test PRIVATE
                      strata_metal_platform strata_metal_options)
# PLE parameter static_asserts in the Metal branch of params.h are compiled by the metallib.
add_dependencies(metal_weight_store_test strata_metallib)
target_include_directories(metal_weight_store_test PRIVATE ${CMAKE_SOURCE_DIR}/include
                           ${CMAKE_SOURCE_DIR}/tests/core)
strata_metal_test(metal_weight_store_test host metal_weight_store_test)
strata_metal_test(model_metadata_test host ${Python3_EXECUTABLE}
                  ${CMAKE_SOURCE_DIR}/tools/metal/check_model_metadata.py
                  $<TARGET_FILE:metal_weight_store_test>
                  ${CMAKE_SOURCE_DIR}/tests/fixtures/model-headers)
strata_metal_test(model_types_test host metal_weight_store_test --test-model-types)
strata_metal_test(native_experts_reader_test host metal_weight_store_test --test-native-experts-reader)
strata_metal_test(pack_index_test host metal_weight_store_test --test-pack-index)
strata_metal_test(expert_address_test host metal_weight_store_test --test-expert-address)
strata_metal_test(ple_constants_test host ${Python3_EXECUTABLE}
                  ${CMAKE_SOURCE_DIR}/tools/metal/check_ple_constants.py
                  $<TARGET_FILE:metal_weight_store_test>
                  ${CMAKE_SOURCE_DIR}/tests/fixtures/model-headers)
strata_metal_test(dense_vs_gguf_test model ${Python3_EXECUTABLE}
                  ${CMAKE_SOURCE_DIR}/tools/metal/check_dense_vs_gguf.py)
set_tests_properties(dense_vs_gguf_test PROPERTIES TIMEOUT 1800)
add_executable(mtp_reader_model_test tests/metal/mtp_reader_model_test.cpp
               src/metal/engine/weights.cpp src/metal/engine/geometry.cpp
               src/metal/engine/window_plan.cpp src/metal/engine/mtp.cpp
               src/metal/engine/mtp_bind_layout.cpp)
target_compile_definitions(mtp_reader_model_test PRIVATE
                           STRATA_TEST_DATA_DIR="${CMAKE_SOURCE_DIR}/data")
target_link_libraries(mtp_reader_model_test PRIVATE
                      strata_metal_platform strata_metal_options)
strata_metal_test(mtp_reader_vocabulary_test host mtp_reader_model_test --vocabulary-self-test)
strata_metal_test(mtp_reader_test model mtp_reader_model_test --models)
set_tests_properties(mtp_reader_test PROPERTIES TIMEOUT 1800)
add_executable(metal_window_test tests/metal/metal_window_test.cpp
               src/metal/engine/window_plan.cpp src/metal/engine/window.cpp)
target_include_directories(metal_window_test PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(metal_window_test PRIVATE strata_metal_options)
target_compile_definitions(metal_window_test PRIVATE
                           STRATA_MODEL_HEADER_DIR="${CMAKE_SOURCE_DIR}/tests/fixtures/model-headers")
strata_metal_test(metal_window_test host metal_window_test)
add_executable(window_params_test tests/metal/window_params_test.cpp
               src/metal/engine/window_params.cpp)
target_include_directories(window_params_test PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(window_params_test PRIVATE strata_metal_options)
strata_metal_test(window_params_test host window_params_test)
add_executable(window_params_ring_test tests/metal/window_params_ring_test.mm)
target_link_libraries(window_params_ring_test PRIVATE
  strata_metal_engine strata_metal_options)
strata_metal_test(window_params_ring_test metal window_params_ring_test)
add_executable(metal_window_scratch_layout_test
               tests/metal/metal_window_scratch_layout_test.cpp
               src/metal/engine/window_scratch_layout.cpp)
target_include_directories(metal_window_scratch_layout_test PRIVATE
                           ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(metal_window_scratch_layout_test PRIVATE
                      strata_metal_options)
strata_metal_test(metal_window_scratch_layout_test host
                  metal_window_scratch_layout_test)
add_executable(metal_window_ple_rows_test
               tests/metal/metal_window_ple_rows_test.cpp
               src/metal/engine/window_ple_rows.cpp)
target_include_directories(metal_window_ple_rows_test PRIVATE
                           ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(metal_window_ple_rows_test PRIVATE
                      strata_metal_platform strata_metal_options)
strata_metal_test(metal_window_ple_rows_test host
                  metal_window_ple_rows_test)
add_executable(metal_tiered_layout_test tests/metal/metal_tiered_layout_test.cpp
               src/metal/engine/tiered.cpp src/metal/engine/geometry.cpp)
target_include_directories(metal_tiered_layout_test PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(metal_tiered_layout_test PRIVATE strata_metal_options)
strata_metal_test(metal_tiered_layout_test host metal_tiered_layout_test)
add_executable(prefill_chunk_test tests/metal/prefill_chunk_test.cpp
               src/metal/engine/prefill.cpp)
target_include_directories(prefill_chunk_test PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(prefill_chunk_test PRIVATE strata_metal_options)
strata_metal_test(prefill_chunk_test host prefill_chunk_test)
# Keep the fixture-only lease issuer out of the production engine. This isolated
# variant has the same sources and ABI, but enables the test friend only here.
add_library(strata_metal_storage_test_engine STATIC ${STRATA_METAL_ENGINE_SOURCES})
target_link_libraries(strata_metal_storage_test_engine
  PUBLIC strata_metal_host strata_spec
  PRIVATE strata_metal_options)
target_compile_definitions(strata_metal_storage_test_engine
  PRIVATE STRATA_METAL_TEST_LOCK_ACCESS=1)

strata_metal_test(metal_mtp_storage_fixture host ${Python3_EXECUTABLE}
                  ${CMAKE_SOURCE_DIR}/tests/metal/prepare_mtp_storage_fixture.py
                  --self-test $<TARGET_FILE:metal_weight_store_test>)
set_tests_properties(metal_mtp_storage_fixture PROPERTIES TIMEOUT 10)
add_executable(metal_mtp_bind_layout_test
               tests/metal/metal_mtp_bind_layout_test.cpp
               src/metal/engine/mtp_bind_layout.cpp)
target_include_directories(metal_mtp_bind_layout_test PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(metal_mtp_bind_layout_test PRIVATE strata_metal_options)
strata_metal_test(metal_mtp_bind_layout_test host metal_mtp_bind_layout_test)
add_executable(metal_mtp_storage_test tests/metal/metal_mtp_storage_test.mm)
target_link_libraries(metal_mtp_storage_test
  PRIVATE strata_metal_storage_test_engine strata_metal_options)
target_compile_definitions(metal_mtp_storage_test
  PRIVATE STRATA_METAL_TEST_LOCK_ACCESS=1)
add_dependencies(metal_mtp_storage_test strata_metallib)
add_dependencies(metal_mtp_storage_test strata)
strata_metal_test(metal_mtp_storage_test metal ${Python3_EXECUTABLE}
                  ${CMAKE_SOURCE_DIR}/tests/metal/prepare_mtp_storage_fixture.py
                  --run-storage-test ${CMAKE_BINARY_DIR}
                  $<TARGET_FILE:metal_weight_store_test>
                  $<TARGET_FILE:metal_mtp_storage_test> ${STRATA_METALLIB}
                  $<TARGET_FILE:strata>)
set_tests_properties(metal_mtp_storage_test PROPERTIES TIMEOUT 60)

add_executable(metal_window_storage_test tests/metal/metal_window_storage_test.mm)
target_link_libraries(metal_window_storage_test
  PRIVATE strata_metal_storage_test_engine strata_metal_oracles
          strata_metal_options)
target_compile_definitions(metal_window_storage_test
  PRIVATE STRATA_METAL_TEST_LOCK_ACCESS=1)
strata_metal_test(metal_window_storage_test metal metal_window_storage_test ${STRATA_METALLIB})
strata_metal_test(metal_never_wrap_test metal metal_window_storage_test ${STRATA_METALLIB} --never-wrap)
strata_metal_test(metal_tiered_copy_test metal metal_window_storage_test ${STRATA_METALLIB} --tiered-copy)
strata_metal_test(metal_model_storage_owner_test metal metal_window_storage_test ${STRATA_METALLIB} --model-owner)
strata_metal_test(metal_fullmode_handoff_test metal metal_window_storage_test ${STRATA_METALLIB} --fullmode-handoff)
set_tests_properties(metal_fullmode_handoff_test PROPERTIES TIMEOUT 180)
strata_metal_test(metal_fullmode_output_projection_test metal
                  metal_window_storage_test ${STRATA_METALLIB}
                  --fullmode-output-projection)
set_tests_properties(metal_fullmode_output_projection_test PROPERTIES TIMEOUT 180)
strata_metal_test(metal_fullmode_output_head_test metal
                  metal_window_storage_test ${STRATA_METALLIB}
                  --fullmode-output-head)
set_tests_properties(metal_fullmode_output_head_test PROPERTIES TIMEOUT 180)
strata_metal_test(metal_fullmode_output_sampling_test metal
                  metal_window_storage_test ${STRATA_METALLIB}
                  --fullmode-output-sampling)
set_tests_properties(metal_fullmode_output_sampling_test PROPERTIES TIMEOUT 180)
strata_metal_test(metal_fullmode_ple_handoff_test metal metal_window_storage_test ${STRATA_METALLIB} --fullmode-ple-handoff)
set_tests_properties(metal_fullmode_ple_handoff_test PROPERTIES TIMEOUT 180)
strata_metal_test(metal_fullmode_ple_q8_handoff_test metal metal_window_storage_test ${STRATA_METALLIB} --fullmode-ple-handoff-q8)
set_tests_properties(metal_fullmode_ple_q8_handoff_test PROPERTIES TIMEOUT 180)
strata_metal_test(metal_fullmode_ple_bf16_handoff_test metal metal_window_storage_test ${STRATA_METALLIB} --fullmode-ple-handoff-bf16)
set_tests_properties(metal_fullmode_ple_bf16_handoff_test PROPERTIES TIMEOUT 180)
add_executable(metal_slot_pool_test tests/metal/metal_slot_pool_test.mm)
target_link_libraries(metal_slot_pool_test PRIVATE strata_metal_engine strata_metal_options)
strata_metal_test(metal_slot_pool_test metal metal_slot_pool_test)
set_tests_properties(metal_slot_pool_test PROPERTIES TIMEOUT 30)
add_executable(serve_transcript_test tests/metal/serve_transcript_test.cpp tests/metal/fake_window.cpp
               src/metal/program/serve.cpp)
target_include_directories(serve_transcript_test PRIVATE ${CMAKE_SOURCE_DIR}/include
                           ${CMAKE_SOURCE_DIR}/tests/metal)
target_compile_definitions(serve_transcript_test PRIVATE
                           STRATA_SERVE_FIXTURE_DIR="${CMAKE_SOURCE_DIR}/tests/metal/fixtures/serve")
target_link_libraries(serve_transcript_test PRIVATE strata_metal_options Threads::Threads)
strata_metal_test(serve_transcript_test host serve_transcript_test)
set_tests_properties(serve_transcript_test PROPERTIES TIMEOUT 30)
add_executable(oneshot_output_test tests/metal/oneshot_output_test.cpp src/metal/program/oneshot.cpp
               src/metal/program/stats.cpp)
target_include_directories(oneshot_output_test PRIVATE ${CMAKE_SOURCE_DIR}/include)
strata_metal_test(oneshot_output_test host oneshot_output_test)
set(_metal_weight_store_test_dir ${CMAKE_BINARY_DIR}/metal/weight-store-cli)
file(MAKE_DIRECTORY ${_metal_weight_store_test_dir})
strata_metal_test(metal_weight_store_cli_test host ${CMAKE_COMMAND} -DSTRATA=$<TARGET_FILE:strata>
                  -DFIXTURE_TOOL=$<TARGET_FILE:metal_weight_store_test>
                  -DPYTHON=${Python3_EXECUTABLE}
                  -DFIXTURE_HELPER=${CMAKE_SOURCE_DIR}/tests/metal/prepare_mtp_storage_fixture.py
                  -DBUILD_ROOT=${CMAKE_BINARY_DIR}
                  -DTEST_ROOT=${_metal_weight_store_test_dir}/fixtures
                  -P ${CMAKE_SOURCE_DIR}/tests/metal/check_weight_store_cli.cmake)
set_tests_properties(metal_weight_store_cli_test PROPERTIES
                     WORKING_DIRECTORY ${_metal_weight_store_test_dir} TIMEOUT 20)
