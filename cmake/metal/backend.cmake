# The macOS-only product: the `strata` engine for Apple silicon on macOS 27 or newer, its `strata.metallib`, and
# the Mac tests (docs/mac/08-build-and-distribution.md, section 4). The root build always includes this backend.

# ---- requirements: checked here, once, each with the command that fixes it ----
if(NOT APPLE OR NOT CMAKE_SYSTEM_NAME STREQUAL "Darwin")
  message(FATAL_ERROR "Strata requires a macOS target on Apple silicon")
endif()
if(NOT CMAKE_OSX_ARCHITECTURES STREQUAL "arm64")
  message(FATAL_ERROR "Strata builds arm64 only: pass -DCMAKE_OSX_ARCHITECTURES=arm64 "
                      "(got '${CMAKE_OSX_ARCHITECTURES}')")
endif()
if(NOT CMAKE_OSX_DEPLOYMENT_TARGET OR CMAKE_OSX_DEPLOYMENT_TARGET VERSION_LESS 27.0)
  message(FATAL_ERROR "Strata requires macOS 27: pass -DCMAKE_OSX_DEPLOYMENT_TARGET=27.0 "
                      "(got '${CMAKE_OSX_DEPLOYMENT_TARGET}')")
endif()
find_program(STRATA_XCRUN xcrun REQUIRED)
execute_process(COMMAND ${STRATA_XCRUN} -sdk macosx --show-sdk-version
                OUTPUT_VARIABLE _strata_sdk OUTPUT_STRIP_TRAILING_WHITESPACE RESULT_VARIABLE _strata_rc ERROR_QUIET)
if(NOT _strata_rc EQUAL 0 OR _strata_sdk VERSION_LESS 27.0)
  message(FATAL_ERROR "Strata requires the macOS 27 SDK from Xcode 27 (found '${_strata_sdk}'): install "
                      "Xcode 27 and run sudo xcode-select -s /Applications/Xcode.app")
endif()
execute_process(COMMAND ${STRATA_XCRUN} -sdk macosx metal --version
                OUTPUT_VARIABLE _strata_metal_version RESULT_VARIABLE _strata_rc ERROR_QUIET)
if(NOT _strata_rc EQUAL 0 OR NOT _strata_metal_version MATCHES "Apple metal version")
  message(FATAL_ERROR "the Metal toolchain is not installed: run xcodebuild -downloadComponent MetalToolchain")
endif()
string(REGEX MATCH "Apple metal version [^\n]*" _strata_metal_version "${_strata_metal_version}")
message(STATUS "Metal: ${_strata_metal_version}, macOS SDK ${_strata_sdk}")

# METAL_REV numbers the Mac releases and only grows (section 11.1).  Only version.cpp sees it, so a bump
# recompiles one file.
file(STRINGS ${CMAKE_SOURCE_DIR}/METAL_REV STRATA_METAL_REV LIMIT_COUNT 1)
if(NOT STRATA_METAL_REV MATCHES "^[1-9][0-9]*$")
  message(FATAL_ERROR "METAL_REV must hold one positive integer (found '${STRATA_METAL_REV}')")
endif()
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/METAL_REV)

enable_language(OBJCXX)
set(CMAKE_OBJCXX_STANDARD 20)
set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)
set(CMAKE_OBJCXX_EXTENSIONS OFF)
option(STRATA_METAL_EMBED_SOURCES "keep the kernel sources in strata.metallib for the Xcode shader debugger" OFF)

# ---- compile options shared by every Mac target ----
# strata_warnings guards its flags with COMPILE_LANGUAGE:CXX, so the Objective-C++ files get theirs here.
add_library(strata_metal_options INTERFACE)
target_link_libraries(strata_metal_options INTERFACE strata_warnings)
target_compile_options(strata_metal_options INTERFACE
  $<$<COMPILE_LANGUAGE:OBJCXX>:-Wall>
  $<$<COMPILE_LANGUAGE:OBJCXX>:-Wextra>
  $<$<AND:$<COMPILE_LANGUAGE:OBJCXX>,$<BOOL:${STRATA_WERROR}>>:-Werror>
  $<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>
  # no absolute path of the checkout in the binaries or their debug information (section 11.4); -g feeds the
  # dSYM that the release keeps beside the stripped binary (section 11.6)
  -ffile-prefix-map=${CMAKE_SOURCE_DIR}=.
  -fdebug-compilation-dir=.
  -g)
target_compile_definitions(strata_metal_options INTERFACE STRATA_METAL=1)
target_link_options(strata_metal_options INTERFACE LINKER:-dead_strip)
# The shared speculative-decoding library is part of the Metal engine; apply the same path mapping.
target_link_libraries(strata_spec PRIVATE strata_metal_options)

# ---- the kernels: strata.metallib and the two generated headers ----
include(${CMAKE_CURRENT_LIST_DIR}/metallib.cmake)

# ---- shared host sources used by the macOS engine (docs/mac/06, section 1.4) ----
add_library(strata_metal_platform STATIC
  src/platform/memory.cpp
  src/platform/memory_probe.cpp
  src/platform/direct_file.cpp
  src/ngram/ple_reader.cpp
  src/kernels/ngram.cpp
)
target_link_libraries(strata_metal_platform PUBLIC strata_artifact Threads::Threads PRIVATE strata_metal_options)

# ---- the Mac host: Metal objects behind a C++ interface (docs/mac/06, section 1.5) ----
add_library(strata_metal_host STATIC
  src/metal/host/context.mm
  src/metal/host/encoder.mm
  src/metal/host/buffers.mm
  src/metal/host/events.mm
  src/metal/host/pipeline_cache.mm
  src/metal/host/capture.mm)
target_include_directories(strata_metal_host PRIVATE ${STRATA_METAL_GENERATED_DIR})
target_link_libraries(strata_metal_host
  PUBLIC strata_metal_platform
  PRIVATE strata_metal_options "-framework Metal" "-framework Foundation" "-framework IOKit"
          "-framework CoreFoundation")
add_dependencies(strata_metal_host strata_metallib)

set(STRATA_METAL_ENGINE_SOURCES
  src/metal/engine/weights.cpp
  src/metal/engine/geometry.cpp
  src/metal/engine/window_plan.cpp
  src/metal/engine/window_storage.cpp
  src/metal/engine/model_storage.cpp
  src/metal/engine/session_state.cpp
  src/metal/engine/session_state_layout.cpp
  src/metal/engine/window_scratch_layout.cpp
  src/metal/engine/window_scratch_storage.cpp
  src/metal/engine/window_ple_rows.cpp
  src/metal/engine/window_ple_stage.cpp
  src/metal/engine/plan.cpp
  src/metal/engine/tiered.cpp
  src/metal/engine/tiered_storage.cpp
  src/metal/engine/slot_pool.cpp
  src/metal/engine/experts.cpp
  src/metal/engine/expert_profile.cpp
  src/metal/engine/kv_layout.cpp
  src/metal/engine/kv.cpp
  src/metal/engine/mtp_kv.cpp
  src/metal/engine/mtp_kv_layout.cpp
  src/metal/engine/sampler_tail.cpp
  src/metal/engine/sampler_dispatch.mm
  src/metal/engine/mv_dispatch.mm
  src/metal/engine/moe_dispatch.cpp
  src/metal/engine/gdn_dispatch.mm
  src/metal/engine/qsa_dispatch.mm
  src/metal/engine/embed_dispatch.mm
  src/metal/engine/prefill_dispatch.mm
  src/metal/engine/prefill_gr_norm_dispatch.mm
  src/metal/engine/prefill_gr_read_dispatch.mm
  src/metal/engine/prefill_gr_write_norm_dispatch.mm
  src/metal/engine/prefill_gdn_gates_dispatch.mm
  src/metal/engine/prefill_gdn_conv_dispatch.mm
  src/metal/engine/prefill_gdn_l2_dispatch.mm
  src/metal/engine/prefill_gdn_rec_dispatch.mm
  src/metal/engine/prefill_gdn_out_norm_dispatch.mm
  src/metal/engine/prefill_gdn_dispatch.mm
  src/metal/engine/prefill_gdn_block_dispatch.mm
  src/metal/engine/prefill_attention_gdn_transition_dispatch.mm
  src/metal/engine/prefill_qsa_gate_dispatch.mm
  src/metal/engine/prefill_qsa_output_dispatch.mm
  src/metal/engine/prefill_qsa_prepare_dispatch.mm
  src/metal/engine/prefill_qsa_tail_dispatch.mm
  src/metal/engine/prefill_qsa_layer_dispatch.mm
  src/metal/engine/prefill_qsa_projection_dispatch.mm
  src/metal/engine/prefill_dequant_dispatch.mm
  src/metal/engine/prefill_native_projection.mm
src/metal/engine/prefill_moe_grouping_dispatch.mm
src/metal/engine/prefill_moe_post_dispatch.mm
src/metal/engine/prefill_moe_expert_dispatch.mm
src/metal/engine/prefill_moe_down_dispatch.mm
src/metal/engine/prefill_moe_pipeline.cpp
src/metal/engine/prefill_moe_full_pipeline.cpp
src/metal/engine/prefill_moe_shared_pipeline.cpp
  src/metal/engine/prefill_gemm_dispatch.mm
  src/metal/engine/prefill_gemm_selector.cpp
  src/metal/engine/prefill_gr_mix_dispatch.mm
  src/metal/engine/prefill_gr_read_mix_dispatch.mm
  src/metal/engine/prefill_gr_silu_dispatch.mm
  src/metal/engine/gr_dispatch.mm
  src/metal/engine/ple_dispatch.mm
  src/metal/engine/window_ple_dispatch.mm
  src/metal/engine/window_pending_dispatch.mm
  src/metal/engine/window_gdn_dispatch.mm
  src/metal/engine/window_dispatch.mm
  src/metal/engine/window.cpp
  src/metal/engine/window_params.cpp
  src/metal/engine/window_params_ring.cpp
  src/metal/engine/window_t1_input.cpp
  src/metal/engine/window_speculative_input.cpp
  src/metal/engine/window_head_mixer.mm
  src/metal/engine/window_output_head.mm
  src/metal/engine/window_output_sampling.mm
  src/metal/engine/window_output_projection.mm
  src/metal/engine/mtp.cpp
  src/metal/engine/mtp_storage.cpp
  src/metal/engine/mtp_bind_layout.cpp
  src/metal/engine/mtp_params_layout.cpp
  src/metal/engine/mtp_runtime_storage.cpp
  src/metal/engine/mtp_dispatch.cpp
  src/metal/engine/mtp_draft_tail_dispatch.cpp
  src/metal/engine/mtp_full_layer_dispatch.cpp
  src/metal/engine/prefill.cpp
  src/metal/engine/snapshot.cpp)
add_library(strata_metal_engine STATIC ${STRATA_METAL_ENGINE_SOURCES})
target_link_libraries(strata_metal_engine PUBLIC strata_metal_host strata_spec PRIVATE strata_metal_options)
set_source_files_properties(src/metal/engine/sampler_tail.cpp
                            PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")

# ---- command-line parsing is also exercised without starting a Metal device ----
add_library(strata_metal_cli_options STATIC src/metal/program/options.cpp)
target_include_directories(strata_metal_cli_options PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(strata_metal_cli_options PUBLIC strata_metal_options)

# ---- the macOS engine executable (docs/mac/06, section 2.1) ----
add_executable(strata
  src/metal/program/main.cpp
  src/metal/program/serve.cpp
  src/metal/program/oneshot.cpp
  src/metal/program/stats.cpp
  src/metal/program/watchdog.cpp
  src/metal/program/version.cpp)
target_link_libraries(strata PRIVATE strata_metal_engine strata_metal_cli_options strata_metal_options)
target_include_directories(strata PRIVATE ${STRATA_METAL_GENERATED_DIR})
set_source_files_properties(src/metal/program/version.cpp PROPERTIES
  COMPILE_DEFINITIONS STRATA_METAL_REV=${STRATA_METAL_REV}
  OBJECT_DEPENDS ${STRATA_METALLIB_SHA256_INC})
add_dependencies(strata strata_metallib)
# ld hashes the debug map into the UUID, and the code signature covers the UUID, so the stripped binary differs
# between checkouts unless the debug map does not: object paths relative to the build folder, object dates zero
# (section 11.4).  dsymutil then runs in the build folder.
file(REAL_PATH ${CMAKE_BINARY_DIR} _strata_real_build)
target_link_options(strata PRIVATE LINKER:-oso_prefix,${_strata_real_build}/)
set_target_properties(strata PROPERTIES CXX_LINKER_LAUNCHER "${CMAKE_COMMAND};-E;env;ZERO_AR_DATE=1")
# strata.metallib sits next to strata in the build tree as in the release, so the engine runs from the build tree
# without --metallib

# the Mac tests, and the host / metal / model labels for every test, upstream ones included (section 6)
include(${CMAKE_CURRENT_LIST_DIR}/tests.cmake)
