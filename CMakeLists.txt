cmake_minimum_required(VERSION 3.24)

if(DEFINED CMAKE_SYSTEM_NAME AND NOT CMAKE_SYSTEM_NAME STREQUAL "Darwin")
  message(FATAL_ERROR "Strata supports macOS targets only; use an Apple Silicon Mac with Xcode 27 or newer")
endif()

project(strata VERSION 0.1.34 LANGUAGES CXX)

if(NOT APPLE OR NOT CMAKE_SYSTEM_NAME STREQUAL "Darwin")
  message(FATAL_ERROR "Strata supports macOS targets only; use an Apple Silicon Mac with Xcode 27 or newer")
endif()

add_compile_definitions(STRATA_VERSION="${PROJECT_VERSION}")
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

option(STRATA_BUILD_TESTS "Build the development test suite" OFF)
option(STRATA_BUILD_CONVERSATION_TESTS "Build standalone conversation cache tests" OFF)
option(STRATA_WERROR "Treat compiler warnings as errors" OFF)
set(STRATA_GGML_DIR "" CACHE PATH "llama.cpp checkout used by Metal reference tests")

find_package(Threads REQUIRED)

add_library(strata_warnings INTERFACE)
target_compile_options(strata_warnings INTERFACE -Wall -Wextra)
if(STRATA_WERROR)
  target_compile_options(strata_warnings INTERFACE -Werror)
endif()

# Shared artifact readers and planning tools used by the Metal engine and its tests.
add_library(strata_artifact INTERFACE)
target_include_directories(strata_artifact INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(strata_artifact INTERFACE strata_warnings)

add_executable(strata-gguf src/artifact/gguf_reader.cpp)
target_link_libraries(strata-gguf PRIVATE strata_artifact)

add_executable(strata-dequant src/artifact/dequant.cpp)
target_link_libraries(strata-dequant PRIVATE strata_artifact)
target_compile_definitions(strata-dequant PRIVATE STRATA_DEQUANT_SELFTEST)

add_library(strata_spec STATIC
  src/spec/suffix_drafter.cpp
  src/spec/controller.cpp
  src/spec/draft_policy.cpp)
target_include_directories(strata_spec PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(strata_spec PRIVATE strata_warnings)

if(STRATA_BUILD_TESTS OR STRATA_BUILD_CONVERSATION_TESTS)
  enable_testing()
  add_executable(conversation_cache_test src/core/conversation_cache_test.cpp)
  target_include_directories(conversation_cache_test PRIVATE include)
  add_test(NAME conversation_cache_test COMMAND conversation_cache_test)
endif()

if(STRATA_BUILD_TESTS)
  enable_testing()
  add_executable(gguf_reader_test src/artifact/gguf_reader_test.cpp)
  target_link_libraries(gguf_reader_test PRIVATE strata_artifact)
  add_test(NAME gguf_reader_test COMMAND gguf_reader_test)

  add_executable(gguf_split_test tests/core/gguf_split_test.cpp)
  target_link_libraries(gguf_split_test PRIVATE strata_artifact)
  add_test(NAME gguf_split_test COMMAND gguf_split_test)

  add_executable(suffix_drafter_test src/spec/suffix_drafter_test.cpp)
  target_link_libraries(suffix_drafter_test PRIVATE strata_spec)
  add_test(NAME suffix_drafter_test COMMAND suffix_drafter_test)
  add_executable(controller_test src/spec/controller_test.cpp)
  target_link_libraries(controller_test PRIVATE strata_spec)
  add_test(NAME controller_test COMMAND controller_test)
  add_executable(draft_policy_test src/spec/draft_policy_test.cpp)
  target_link_libraries(draft_policy_test PRIVATE strata_spec)
  add_test(NAME draft_policy_test COMMAND draft_policy_test)

  add_executable(conv_cache_test src/program/conv_cache_test.cpp)
  target_include_directories(conv_cache_test PRIVATE include)
  add_test(NAME conv_cache_test COMMAND conv_cache_test)
  add_executable(coupled_draft_test src/core/coupled_draft_test.cpp)
  target_include_directories(coupled_draft_test PRIVATE include)
  add_test(NAME coupled_draft_test COMMAND coupled_draft_test)
  add_executable(bf16_bits_test src/kernels/bf16_bits_test.cpp)
  target_include_directories(bf16_bits_test PRIVATE include)
  add_test(NAME bf16_bits_test COMMAND bf16_bits_test)

  add_executable(direct_file_async_test
    tests/platform/direct_file_async.cpp
    src/platform/direct_file.cpp)
  target_include_directories(direct_file_async_test PRIVATE include)
  target_link_libraries(direct_file_async_test PRIVATE Threads::Threads)
  add_test(NAME direct_file_async_test COMMAND direct_file_async_test)
endif()

# This repository builds and tests the Metal product on macOS; no other runtime backend is configured.
include(cmake/metal/backend.cmake)
