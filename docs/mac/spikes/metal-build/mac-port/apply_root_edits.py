"""Apply edits A-G of docs/mac/08-build-and-distribution.md section 3 to an upstream CMakeLists.txt at 1678de33.

    python3 apply_root_edits.py <checkout>/CMakeLists.txt

It checks every anchor line before it changes anything, keeps the UTF-8 BOM and the LF line endings, and refuses a
file that already has a `# MAC:` line.  The project applies the edits once; later compatibility reviews compare and preserve them while selectively porting changes.
"""
import sys

ANCHORS = {   # line number (1-based, at 1678de33) -> the exact upstream text there
    36: 'option(STRATA_ENABLE_HIP "Build the HIP targets for wave32 AMD GPUs (gfx1100, gfx1201; gfx1030 unvalidated)" OFF)',
    48: 'endif()',
    778: 'add_library(strata_kernels_cpu STATIC src/kernels/cpu/expert.cpp src/kernels/cpu/pool.cpp',
    793: 'target_link_libraries(strata_kernels_cpu PUBLIC Threads::Threads)',
    801: 'set(STRATA_GGML_DIR "" CACHE PATH "llama.cpp checkout to take ggml from (empty: fetch the pinned commit)")',
    996: '  add_executable(expert_multi_test src/kernels/cpu/expert_multi_test.cpp)',
    1013: '  add_test(NAME pool_stress COMMAND pool_stress 20)',
    1015: '  if(UNIX AND NOT APPLE)',
    1029: 'endif()',
}

A = ['option(STRATA_ENABLE_METAL "Build the Metal engine for Apple silicon, macOS 27+ (docs/mac/08)" OFF)  # MAC:']
B = ['if(STRATA_ENABLE_CUDA AND CMAKE_SYSTEM_NAME STREQUAL "Darwin")  # MAC:',
     '  message(FATAL_ERROR "The CUDA backend is not supported on macOS; use STRATA_ENABLE_METAL=ON on Apple silicon")',
     'endif()  # MAC:',
     'if(STRATA_ENABLE_METAL AND (STRATA_ENABLE_CUDA OR STRATA_ENABLE_HIP))  # MAC:',
     '  message(FATAL_ERROR "STRATA_ENABLE_METAL is a separate backend; turn STRATA_ENABLE_CUDA and STRATA_ENABLE_HIP off")',
     'endif()  # MAC:']
C = ['# MAC: the CPU expert kernels are x86-64 only (AVX2 / AVX-512); an arm64 build has no CPU expert path (docs/mac/08)',
     'set(_strata_x86 OFF)  # MAC:',
     'if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64)$" AND NOT CMAKE_OSX_ARCHITECTURES MATCHES "arm64")  # MAC:',
     '  set(_strata_x86 ON)  # MAC:',
     'endif()  # MAC:',
     'find_package(Threads REQUIRED)  # MAC: needed without the x86 block too',
     'if(_strata_x86)  # MAC:']
C_END = ['endif()  # MAC: _strata_x86']
D = ['if(NOT _strata_x86)  # MAC: the ggml-cpu experts live in strata_kernels_cpu',
     '  set(STRATA_NATIVE_EXPERTS OFF CACHE BOOL "" FORCE)  # MAC:',
     'endif()  # MAC:']
E = ['  if(_strata_x86)  # MAC:']
E_END = ['  endif()  # MAC: _strata_x86']
F = '  if(UNIX)  # MAC: direct_file.cpp has an F_NOCACHE path on macOS (docs/mac/06 section 7.4)'
G = ['if(STRATA_ENABLE_METAL)  # MAC:',
     '  include(cmake/metal/backend.cmake)  # MAC:',
     'endif()  # MAC:']


def main(path):
    raw = open(path, 'rb').read()
    bom = raw.startswith(b'\xef\xbb\xbf')
    text = raw[3:].decode('utf-8') if bom else raw.decode('utf-8')
    if '\r\n' in text:
        sys.exit('CRLF line endings: expected LF')
    if '# MAC:' in text:
        sys.exit('already edited')
    lines = text.split('\n')
    for n, want in ANCHORS.items():
        if lines[n - 1] != want:
            sys.exit(f'line {n} is {lines[n - 1]!r}, expected {want!r}: re-derive the anchors for this commit')
    # bottom to top, so that the line numbers above each edit stay valid
    lines[1029:1029] = G
    lines[1014] = F
    lines[1013:1013] = E_END
    lines[995:995] = E
    lines[801:801] = D
    lines[793:793] = C_END
    lines[777:777] = C
    lines[48:48] = B
    lines[36:36] = A
    out = '\n'.join(lines).encode('utf-8')
    open(path, 'wb').write((b'\xef\xbb\xbf' if bom else b'') + out)
    print(f'{path}: {sum("# MAC:" in x for x in lines)} lines with # MAC:')


if __name__ == '__main__':
    main(sys.argv[1])
