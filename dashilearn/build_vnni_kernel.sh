#!/usr/bin/env bash
set -euo pipefail

cc="${CC:-cc}"
${cc} ${CFLAGS:-} -O3 -fPIC -shared -o vnni_kernel.so vnni_kernel.c
