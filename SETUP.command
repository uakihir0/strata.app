#!/bin/sh
# Double-click in Finder: choose the model, context and options again (the same as ./setup.sh --setup).
# Until the Mac installer is implemented, setup.sh stops with exit code 2 and says so (docs/mac/07, section 5.1).
cd "$(dirname "$0")" || exit 1
exec ./setup.sh --setup "$@"
