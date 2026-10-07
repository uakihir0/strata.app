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
foreach(_o GGML_METAL GGML_BLAS GGML_ACCELERATE GGML_OPENMP GGML_NATIVE GGML_CCACHE GGML_BACKEND_DL GGML_BUILD_TESTS
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

# ---- tests/metal ----
add_executable(dequant_oracle_test tests/metal/dequant_oracle_test.cpp)
target_link_libraries(dequant_oracle_test PRIVATE strata_artifact strata_ggml_oracle strata_metal_options)
strata_metal_test(dequant_oracle_test host dequant_oracle_test)

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

# the shipped engine: identity and the hash it embeds, on the binary itself (host: --version opens no device)
strata_metal_test(strata_version host ${CMAKE_COMMAND} -DSTRATA=$<TARGET_FILE:strata> -DMETALLIB=${STRATA_METALLIB}
                  -DVERSION=${PROJECT_VERSION} -DMETAL_REV=${STRATA_METAL_REV}
                  -P ${CMAKE_CURRENT_LIST_DIR}/check_engine.cmake)
