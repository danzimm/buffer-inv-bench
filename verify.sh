#!/usr/bin/env bash
# Compare the GPU machine code of the working copy against the last commit.
set -euo pipefail
cd "$(dirname "$0")"
ARCH=${ARCH:-gfx950}   # gfx942 on MI300
LLVM=${ROCM_PATH:-/opt/rocm}/llvm/bin

disasm() {  # disasm <source.hip> <output.dis>
  hipcc -O3 -std=c++17 --offload-arch=$ARCH --offload-device-only -c -o /tmp/verify.o "$1"
  "$LLVM/clang-offload-bundler" --unbundle --type=o --input=/tmp/verify.o \
      --targets=hipv4-amdgcn-amd-amdhsa--$ARCH --output=/tmp/verify.co
  "$LLVM/llvm-objdump" -d --no-show-raw-insn /tmp/verify.co |
      sed -E 's/^\s*[0-9a-f]+:\s*//; s/\/\/.*$//' | grep -v 'file format' > "$2"
}

git show HEAD:bench.hip > /tmp/bench-head.hip
disasm /tmp/bench-head.hip /tmp/head.dis
disasm bench.hip /tmp/work.dis
if diff -q /tmp/head.dis /tmp/work.dis > /dev/null; then
  echo "identical machine code to HEAD"
else
  echo "machine code differs from HEAD:"; diff /tmp/head.dis /tmp/work.dis | head -40
fi
