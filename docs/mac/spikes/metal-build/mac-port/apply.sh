#!/bin/sh
# Turn an upstream checkout at UPSTREAM_SYNC into the verified Mac skeleton of docs/mac/08-build-and-distribution.md:
# the root edits A-G, the direct_file.cpp patch and the .gitignore / .gitattributes lines (root-edits.patch), then
# the new files (tree/).
#
#   ./apply.sh <upstream checkout>
#
# Refuses a checkout at another commit or with local changes.  The result builds with
# `cmake --preset mac-dev && cmake --build --preset mac-dev && ctest --preset mac-host`.
set -eu
here=$(cd "$(dirname "$0")" && pwd)
dst=${1:?usage: apply.sh <upstream checkout>}
want=$(cat "$here/tree/UPSTREAM_SYNC")
have=$(git -C "$dst" rev-parse HEAD)
[ "$have" = "$want" ] || { echo "apply.sh: $dst is at $have, the skeleton is for $want" >&2; exit 1; }
[ -z "$(git -C "$dst" status --porcelain)" ] || { echo "apply.sh: $dst has local changes" >&2; exit 1; }
git -C "$dst" apply --whitespace=error-all "$here/root-edits.patch"
(cd "$here/tree" && find . -type f) | sed 's|^\./||' | sort | rsync -a --files-from=- "$here/tree/" "$dst/"
echo "apply.sh: $dst now has the Mac skeleton ($(grep -c '# MAC:' "$dst/CMakeLists.txt") lines with # MAC: in CMakeLists.txt)"
