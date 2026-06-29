#!/usr/bin/env bash
# Build the GOT-patching seed hook for libcg.so.
# Run from anywhere inside the project:
#   bash cg/build_seed_hook.sh
set -e
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$DIR/libcg_seed_hook.so"
g++ -shared -fPIC -O2 -std=c++17 \
    -o "$OUT" \
    "$DIR/seed_hook.cpp" \
    -ldl
echo "Built: $OUT"
