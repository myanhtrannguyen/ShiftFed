#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

mpirun -np 4 python3 source/parallel/parallel_fedavg_mpi.py \
  --rounds 2 \
  --local-steps 5 \
  --batch-size 16 \
  --model lenet5 \
  --synthetic \
  --log-dir outputs/fl_mpi_local_smoke
