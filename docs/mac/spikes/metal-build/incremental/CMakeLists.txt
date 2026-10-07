# The metallib build of docs/mac/08-build-and-distribution.md, section 5, cut down to one kernel: AIR with a depfile, a
# stamp-driven link step whose byproducts are the metallib and the generated hash header, and an engine that embeds
# the hash.  See ../README.md for the steps and the expected results.
cmake_minimum_required(VERSION 3.24)
project(incr LANGUAGES CXX)
set(K ${CMAKE_SOURCE_DIR}/k.metal)
set(AIR ${CMAKE_BINARY_DIR}/k.air)
set(MLIB ${CMAKE_BINARY_DIR}/strata.metallib)
set(INC ${CMAKE_BINARY_DIR}/generated/metallib_sha256.inc)
set(STAMP ${CMAKE_BINARY_DIR}/strata_metallib.stamp)
add_custom_command(OUTPUT ${AIR}
  COMMAND /bin/sh ${CMAKE_SOURCE_DIR}/metal.sh -std=metal4.0 -mmacosx-version-min=27.0 -gline-tables-only
          -frecord-sources -ffile-prefix-map=${CMAKE_SOURCE_DIR}=. -fdebug-compilation-dir=. -I${CMAKE_SOURCE_DIR}/common
          -c ${K} -o ${AIR} -MMD -MF ${AIR}.d -MT ${AIR}
  DEPENDS ${K} DEPFILE ${AIR}.d VERBATIM)
add_custom_command(OUTPUT ${STAMP}
  BYPRODUCTS ${MLIB} ${INC}
  COMMAND xcrun -sdk macosx metallib ${AIR} -o ${MLIB}.tmp
  COMMAND xcrun -sdk macosx metal-dsymutil -flat -remove-source ${MLIB}.tmp -o ${CMAKE_BINARY_DIR}/strata.metallibsym
  COMMAND ${CMAKE_COMMAND} -E rename ${MLIB}.tmp ${MLIB}
  COMMAND ${CMAKE_COMMAND} -DMETALLIB=${MLIB} -DINC=${INC} -P ${CMAKE_SOURCE_DIR}/hash.cmake
  COMMAND ${CMAKE_COMMAND} -E touch ${STAMP}
  DEPENDS ${AIR} ${CMAKE_SOURCE_DIR}/hash.cmake VERBATIM)
add_custom_target(strata_metallib DEPENDS ${STAMP})
add_executable(strata main.cpp version.cpp)
target_include_directories(strata PRIVATE ${CMAKE_BINARY_DIR}/generated)
set_source_files_properties(version.cpp PROPERTIES OBJECT_DEPENDS ${INC})
add_dependencies(strata strata_metallib)
