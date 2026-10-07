# strata.metallib from src/metal/kernels (docs/mac/08-build-and-distribution.md, section 5).
#
#   kernels/common/*.h      shared Metal headers, no kernels
#   kernels/safe/*.metal    kernels compiled with safe math: the parity gates hold them to bitwise or ulp bounds
#   kernels/fast/*.metal    kernels compiled with fast math: only where docs/mac/04 allows it
#   kernels/manifest.txt    one line per kernel: <function> <safe|fast> <threads-x> <threads-y> <threads-z>
#
# Outputs, all in the build folder: strata.metallib (next to strata), strata.metallibsym (the stripped kernel
# sources and line tables, for the release's symbols archive), generated/metallib_sha256.inc (the hash that the
# engine embeds and checks at startup) and generated/kernel_manifest.inc (the manifest as a C++ table).
set(_k ${CMAKE_SOURCE_DIR}/src/metal/kernels)
set(STRATA_METAL_GENERATED_DIR ${CMAKE_BINARY_DIR}/generated)
set(STRATA_METALLIB ${CMAKE_BINARY_DIR}/strata.metallib)
set(STRATA_METALLIB_SHA256_INC ${STRATA_METAL_GENERATED_DIR}/metallib_sha256.inc)

# ---- the manifest, read at configure time ----
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${_k}/manifest.txt)
file(STRINGS ${_k}/manifest.txt _lines)
set(_names "")
set(_rows "")
foreach(_line IN LISTS _lines)
  if(_line MATCHES "^[ \t]*(#|$)")
    continue()
  endif()
  if(NOT _line MATCHES "^([a-z_][a-z0-9_]*) (safe|fast) ([1-9][0-9]*) ([1-9][0-9]*) ([1-9][0-9]*)$")
    message(FATAL_ERROR "src/metal/kernels/manifest.txt: bad line '${_line}' "
                        "(want '<function> <safe|fast> <threads-x> <threads-y> <threads-z>', one space apart)")
  endif()
  set(_name ${CMAKE_MATCH_1})
  set(_mode ${CMAKE_MATCH_2})
  set(_tx ${CMAKE_MATCH_3})
  set(_ty ${CMAKE_MATCH_4})
  set(_tz ${CMAKE_MATCH_5})
  if(_tx GREATER 1024 OR _ty GREATER 1024 OR _tz GREATER 1024)
    message(FATAL_ERROR "src/metal/kernels/manifest.txt: threadgroup axes must each be in 1..1024")
  endif()
  math(EXPR _threads "${_tx} * ${_ty} * ${_tz}")
  if(_threads GREATER 1024)
    message(FATAL_ERROR "src/metal/kernels/manifest.txt: threadgroup for ${_name} has ${_threads} threads, maximum is 1024")
  endif()
  if(_name IN_LIST _names)
    message(FATAL_ERROR "src/metal/kernels/manifest.txt: ${_name} is listed twice")
  endif()
  list(APPEND _names ${_name})
  string(APPEND _rows "{\"${_name}\", KernelMath::${_mode}, ${_threads}, ${_tx}, ${_ty}, ${_tz}},\n")
endforeach()
if(NOT _names)
  message(FATAL_ERROR "src/metal/kernels/manifest.txt lists no kernel")
endif()
file(CONFIGURE OUTPUT ${STRATA_METAL_GENERATED_DIR}/kernel_manifest.inc @ONLY CONTENT
"// generated from src/metal/kernels/manifest.txt by cmake/metal/metallib.cmake; do not edit
@_rows@")

# ---- one AIR file per kernel source ----
set(_common -std=metal4.0 -mmacosx-version-min=${CMAKE_OSX_DEPLOYMENT_TARGET} -Wall
            -gline-tables-only -frecord-sources -ffile-prefix-map=${CMAKE_SOURCE_DIR}=. -fdebug-compilation-dir=.
            -I${_k}/common)
if(STRATA_WERROR)
  list(APPEND _common -Werror)
endif()
set(_flags_safe -fmetal-math-mode=safe -fmetal-math-fp32-functions=precise -ffp-contract=off)
set(_flags_fast -fmetal-math-mode=fast)
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/metal/safe ${CMAKE_BINARY_DIR}/metal/fast)
set(_airs "")
foreach(_mode safe fast)
  file(GLOB _srcs CONFIGURE_DEPENDS ${_k}/${_mode}/*.metal)   # sorted, so the link order is fixed
  foreach(_src IN LISTS _srcs)
    get_filename_component(_n ${_src} NAME_WE)
    set(_air ${CMAKE_BINARY_DIR}/metal/${_mode}/${_n}.air)
    add_custom_command(OUTPUT ${_air}
      COMMAND /bin/sh ${CMAKE_CURRENT_LIST_DIR}/metal.sh ${_common} ${_flags_${_mode}}
              -c ${_src} -o ${_air} -MMD -MF ${_air}.d -MT ${_air}
      DEPENDS ${_src} ${CMAKE_CURRENT_LIST_DIR}/metal.sh
      DEPFILE ${_air}.d
      COMMENT "Compiling Metal (${_mode} math) ${_n}.metal"
      VERBATIM)
    list(APPEND _airs ${_air})
  endforeach()
endforeach()

# ---- link, strip, check, hash ----
# The stamp is the only declared output; the metallib, the symbols file and the hash header are byproducts, so a
# kernel edit that leaves the metallib byte-identical rebuilds nothing downstream (copy_if_different).
set(_stamp ${CMAKE_BINARY_DIR}/metal/strata_metallib.stamp)
string(REPLACE ";" "|" _air_arg "${_airs}")   # one -D value for the script; | never occurs in a path we make
set(_tmp ${STRATA_METALLIB}.tmp)
if(STRATA_METAL_EMBED_SOURCES)
  set(_strip_cmd ${CMAKE_COMMAND} -E remove -f ${CMAKE_BINARY_DIR}/strata.metallibsym)
else()
  set(_strip_cmd ${STRATA_XCRUN} -sdk macosx metal-dsymutil -flat -remove-source ${_tmp}
                 -o ${CMAKE_BINARY_DIR}/strata.metallibsym)
endif()
add_custom_command(OUTPUT ${_stamp}
  BYPRODUCTS ${STRATA_METALLIB} ${CMAKE_BINARY_DIR}/strata.metallibsym ${STRATA_METALLIB_SHA256_INC}
  COMMAND ${STRATA_XCRUN} -sdk macosx metallib ${_airs} -o ${_tmp}
  COMMAND ${_strip_cmd}
  COMMAND ${CMAKE_COMMAND} -DXCRUN=${STRATA_XCRUN} -DMANIFEST=${_k}/manifest.txt -DAIRS=${_air_arg}
          -DTMP=${_tmp} -DMETALLIB=${STRATA_METALLIB} -DINC=${STRATA_METALLIB_SHA256_INC}
          -P ${CMAKE_CURRENT_LIST_DIR}/check_manifest.cmake
  COMMAND ${CMAKE_COMMAND} -E touch ${_stamp}
  DEPENDS ${_airs} ${_k}/manifest.txt ${CMAKE_CURRENT_LIST_DIR}/check_manifest.cmake
  COMMENT "Linking strata.metallib"
  VERBATIM)
add_custom_target(strata_metallib ALL DEPENDS ${_stamp})
