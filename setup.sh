#!/bin/sh
# Strata for macOS: fail closed until the native installer is implemented.
cd "$(dirname "$0")" || exit 1

if [ "$(uname -s)" != "Darwin" ]; then
  echo "Strata supports macOS only. Run setup on an Apple silicon Mac."
  exit 1
fi
if [ "$(/usr/sbin/sysctl -n hw.optional.arm64 2>/dev/null)" != "1" ]; then
  echo "Strata needs a Mac with Apple silicon (M1 or newer)."
  exit 1
fi
macos_version=$(/usr/bin/sw_vers -productVersion 2>/dev/null)
macos_major=${macos_version%%.*}
case "$macos_major" in
  ''|*[!0-9]*) echo "Could not determine the macOS version."; exit 1 ;;
esac
if [ "$macos_major" -lt 27 ]; then
  echo "Strata needs macOS 27 or newer; this Mac has macOS $macos_version."
  exit 1
fi

echo "Strata's macOS setup is not implemented yet. This checkout does not create a virtual environment or install files."
exit 2
