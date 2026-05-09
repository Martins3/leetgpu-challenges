#!/usr/bin/env bash
set -E -e -u -o pipefail

cd /home/martins3/data/leetgpu-challenges
export LD_LIBRARY_PATH="$(dirname "$(gcc -print-file-name=libstdc++.so.6)")"
export LOCAL_CUDA_LIBCUDA_PATH="/lib64/libcuda.so.1"
# .venv/bin/python local_cuda/local_test.py challenges/easy/1_vector_add
.venv/bin/python local_cuda/local_test.py challenges/easy/19_reverse_array
