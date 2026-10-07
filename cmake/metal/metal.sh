#!/bin/sh
# xcrun metal with the toolchain and SDK paths mapped away; both are resolved on every call because the toolchain's
# cryptex mount point has a random suffix that can change (docs/mac/08-build-and-distribution.md, section 5.3)
tc=$(cd "$(dirname "$(xcrun -sdk macosx -f metal)")/../.." && pwd -P) || exit 1
sdk=$(xcrun -sdk macosx --show-sdk-path) || exit 1
exec xcrun -sdk macosx metal "-ffile-prefix-map=$tc=/metal-toolchain" "-ffile-prefix-map=$sdk=/macos-sdk" "$@"
