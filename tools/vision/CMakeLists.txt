# Build the optional Metal image encoder against the same pinned llama.cpp as the engine tests.
cmake_minimum_required(VERSION 3.24)
project(strata_vision C CXX)

if(NOT APPLE OR NOT CMAKE_SYSTEM_NAME STREQUAL "Darwin")
  message(FATAL_ERROR "strata-vision is supported on macOS only")
endif()

set(_project_root "${CMAKE_CURRENT_SOURCE_DIR}/../..")
file(STRINGS "${_project_root}/third_party/ggml/VERSION.txt" _llama_pin LIMIT_COUNT 1)
string(REGEX MATCH "^[0-9a-f]+" _llama_pin "${_llama_pin}")
string(LENGTH "${_llama_pin}" _llama_pin_length)
if(NOT _llama_pin_length EQUAL 40)
  message(FATAL_ERROR "third_party/ggml/VERSION.txt must contain a full llama.cpp commit SHA")
endif()

include(FetchContent)
FetchContent_Declare(strata_vision_llama
  GIT_REPOSITORY https://github.com/ggml-org/llama.cpp.git
  GIT_TAG "${_llama_pin}"
  GIT_SHALLOW FALSE
  SOURCE_SUBDIR strata-populate-only)
FetchContent_MakeAvailable(strata_vision_llama)
set(_llama_dir "${strata_vision_llama_SOURCE_DIR}")

set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_APP OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_MTMD ON CACHE BOOL "" FORCE)
set(LLAMA_OPENSSL OFF CACHE BOOL "" FORCE)
set(LLAMA_CURL OFF CACHE BOOL "" FORCE)
set(GGML_METAL ON CACHE BOOL "" FORCE)
set(GGML_NATIVE ON CACHE BOOL "" FORCE)
foreach(_backend GGML_CUDA GGML_HIP GGML_VULKAN GGML_SYCL GGML_OPENCL GGML_BLAS GGML_ACCELERATE
                 GGML_OPENMP GGML_BACKEND_DL)
  set(${_backend} OFF CACHE BOOL "" FORCE)
endforeach()

add_subdirectory("${_llama_dir}" llama EXCLUDE_FROM_ALL)

add_executable(strata-vision strata_vision.cpp)
target_link_libraries(strata-vision PRIVATE mtmd llama)
target_compile_features(strata-vision PRIVATE cxx_std_17)
set_target_properties(strata-vision PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
