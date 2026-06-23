# Federated Learning MPI for Domain Shift Digits

This experiment trains one global digit classifier across three isolated domains:
MNIST, SVHN, and USPS. Each domain is owned by one MPI client process. Rank 0 is
the server, broadcasts the global model, gathers client weights, applies FedAvg,
and evaluates on the test splits of all three datasets after every round.

The default training architecture is LeNet-5, a compact CNN for digit
recognition. Inputs are converted to grayscale `1x28x28`; SVHN is resized from
color `32x32`, and USPS is resized from `16x16`.

LeNet-5 layers:

1. Conv1: 6 filters, `5x5`, ReLU, then `2x2` max pooling.
2. Conv2: 16 filters, `5x5`, ReLU, then `2x2` max pooling.
3. FC1: 120 nodes, ReLU.
4. FC2: 84 nodes, ReLU.
5. Output: 10 nodes.

With `28x28` inputs, Conv1 uses padding 2 so the classifier receives
`16x5x5` features, keeping the parameter count close to classic LeNet-5
and the MPI weight payload small.

## Parallel Model

- Parallel level: data-level parallelism, specifically Federated Data Parallelism.
- Decomposition: domain-based data decomposition.
- Mapping: static master-worker, 1 server and 3 clients.
- Communication: star topology.
- Rank 0: server, model initialization, broadcast, gather/reduce, evaluation.
- Rank 1: MNIST client.
- Rank 2: SVHN client.
- Rank 3: USPS client.

The synchronous version uses blocking `bcast` and `gather`. The asynchronous
version uses `isend`/`irecv`, so a fast client can send an update without waiting
for slower domains.

## Install

On a fresh Ubuntu/WSL machine, install `pip` first:

```bash
sudo apt-get update
sudo apt-get install -y python3-pip
```

Then install Python packages:

```bash
python -m pip install -r source/parallel/requirements_fl_mpi.txt
```

You also need an MPI runtime such as OpenMPI or MPICH. On Ubuntu:

```bash
sudo apt-get install -y openmpi-bin libopenmpi-dev
```

If `sudo` asks for a password, run those install commands manually in an
interactive terminal first. The local MPI launcher itself can be tested without
PyTorch using:

```bash
mpirun -np 4 python3 -c "import socket, os; print(socket.gethostname(), os.getpid())"
```

## Run Synchronous MPI FL

## One-Machine Smoke Test

Run this first before deploying to multiple machines. It starts 4 MPI processes
on one computer: rank 0 is the server, ranks 1-3 are the MNIST/SVHN/USPS clients.
The `--synthetic` flag avoids dataset downloads and only checks the training and
communication pipeline.

```bash
bash source/parallel/run_local_smoke.sh
```

Equivalent explicit command:

```bash
mpirun -np 4 python3 source/parallel/parallel_fedavg_mpi.py \
  --rounds 2 \
  --local-steps 5 \
  --batch-size 16 \
  --model lenet5 \
  --synthetic \
  --log-dir outputs/fl_mpi_local_smoke
```

If this passes, replace `--synthetic` with `--download` or point `--data-dir` to
prepared MNIST/SVHN/USPS data.

## Run Synchronous MPI FL

```bash
mpirun -np 4 python source/parallel/parallel_fedavg_mpi.py \
  --rounds 10 \
  --local-steps 100 \
  --batch-size 64 \
  --model lenet5 \
  --download
```

Use `--synthetic` for a fast smoke test that does not download datasets:

```bash
mpirun -np 4 python source/parallel/parallel_fedavg_mpi.py --rounds 2 --local-steps 5 --synthetic
```

## Run Asynchronous MPI FL

```bash
mpirun -np 4 python source/parallel/parallel_fedavg_mpi.py \
  --rounds 10 \
  --local-steps 100 \
  --model lenet5 \
  --async-mode \
  --download
```

## Multi-Machine Hostfile

After the local test works, copy [hostfile.example](hostfile.example) and edit
hostnames/IPs. A typical run is:

```bash
mpirun --hostfile source/parallel/hostfile.example \
  -np 4 \
  python3 source/parallel/parallel_fedavg_mpi.py \
  --rounds 10 \
  --local-steps 100 \
  --model lenet5 \
  --download
```

For multi-machine MPI, make sure all machines have:

- the same source code path;
- the same Python environment and packages;
- SSH access without an interactive password prompt;
- network/firewall rules that allow MPI worker connections.

## Run Sequential Baseline

```bash
python source/parallel/sequential_fl.py \
  --rounds 10 \
  --local-steps 100 \
  --model lenet5 \
  --download
```

The sequential baseline loops over MNIST, SVHN, and USPS in one process, then
averages weights. It is intended for accuracy sanity checks and speedup
calculation.

## Benchmark

```bash
python source/parallel/benchmark.py --rounds 10 20 50 --models lenet5 mlp --download
```

For a quick no-download run:

```bash
python source/parallel/benchmark.py --rounds 2 --models mlp --local-steps 5 --synthetic
```

The benchmark writes `outputs/fl_mpi/benchmark_summary.csv`.

## Plot Results

```bash
python source/parallel/plot_results.py --log-dir outputs/fl_mpi
```

Generated figures include:

- accuracy/loss per communication round;
- compute time vs communication time;
- MPI speedup compared with sequential FL when benchmark summary exists.

## Dataset Notes

All images are converted to grayscale `1x28x28`, so the same CNN/MLP can train
across MNIST, SVHN, and USPS despite domain shift. Because dataset sizes are
imbalanced, local training is controlled by fixed mini-batch steps per round
(`--local-steps`) instead of full local epochs.
