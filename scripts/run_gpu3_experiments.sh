#!/usr/bin/env bash
# Run inside ycho-holosoma. Preparation only: nothing starts until this file is invoked.
set -e -o pipefail
if [[ ! -f /.dockerenv ]]; then
    echo "Run this script inside the Holosoma Docker container." >&2
    exit 1
fi
queue_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$queue_root"
source scripts/source_isaacsim_setup.sh
exec python scripts/run_bilateral_queue.py "$@" --gpu 3
