# Parallel Federated Learning: Evaluation Guide

This document outlines the step-by-step practical process to evaluate the MPI-based Federated Learning system. It covers the initial training verification and all performance analysis experiments (Benchmark, Granularity, and Speedup).

**Prerequisites:**
Ensure you are in the `src` directory and have your MPI cluster (e.g., `localhost` and `slave` nodes) configured.

```bash
cd source/parallel/src
```

---

## 1. Initial Training & Testing (Accuracy Verification)
Before analyzing performance, verify that the parallel Federated Learning algorithm correctly solves the classification problem and that the accuracy converges over time.

Run a short test with 4 processes (1 Server + 3 Clients) in Asynchronous mode:

```bash
mpirun -host localhost,slave2 -np 4 --map-by node python3.10 parallel_fedavg_mpi.py --rounds 10 --local-steps 100 --async-mode
```

**Expected Output:**
You should see terminal logs indicating the global accuracy (`acc`) increasing round by round. This proves the models are successfully learning and communicating.

---

## 2. Benchmark: Finding Input Size (N) for 2-3 Minutes Runtime
The goal is to determine the input data size $N$ (represented by the number of communication `--rounds`) that causes the entire program to run for exactly 120 - 180 seconds using all available physical cores.

Assuming your cluster has **12 physical cores**, run the benchmark plotting script:

```bash
python3.10 benchmark_plot.py --np 12 --rounds 5 10 20 30 40 50
```

**What it does:**
- Iterates through the specified $N$ values.
- Plots `benchmark_plot.png` with two lines: **Total Wall Time** (with communication) and **Compute Only Time** (without communication).
- Highlights a horizontal green band representing the 120-180s target.
- Prints the optimal $N$ to the terminal.

---

## 3. Granularity and Load Balancing Check
Using the optimal $N$ discovered in Step 2 (e.g., $N = 30$), we analyze the workload distribution among the clients to check for load balancing issues (stragglers).

```bash
# Example using N=30
python3.10 granularity_plot.py --np 12 --rounds 30 --local-steps 100
```

**What it does:**
- Generates a stacked bar chart (`granularity_plot.png`) showing Compute Time vs. Communication/Idle Time for each client rank.
- Calculates the maximum idle time difference between any two processes.
- **Evaluation:** If the difference exceeds **25%**, the system is unbalanced.

**Fixing the Imbalance (Adjusting Granularity):**
If the system is unbalanced, you can either:
1. Make the granularity **finer** by reducing local steps (e.g., `--local-steps 10`).
2. **(Recommended)** Enable the built-in dynamic load balancer:
   ```bash
   python3.10 granularity_plot.py --np 12 --rounds 30 --local-steps 100 --load-balance
   ```

---

## 4. Speedup Analysis
To measure the parallel speedup (Strong Scaling), we run the program for $N$ communication rounds (e.g., $N=30$, identical to normal training) and fix the total computation workload per round across the cluster to $2 \times \text{local\_steps}$. We then vary the number of worker processes from $1, 2, 4, 8, \dots, 2X$ (where $X$ is the total number of physical cores).

Assuming $X = 12$ physical cores:

```bash
python3.10 speedup_plot.py --x-cores 12 --n 30
```

**What it does:**
- Runs $N=30$ rounds (`--rounds 30`) and automatically divides the fixed total local computation steps per round evenly among the current number of active workers (`local_steps = (2 * base_local_steps) // workers`).
- Measures both Wall Time (with communication) and Compute Time (without communication) as the number of processes scales up.
- Generates `speedup_plot.png` containing two subplots:
  1. **Execution Time vs. Workers** (showing time decreasing).
  2. **Speedup vs. Workers** (showing the speedup curve alongside the ideal $y=x$ baseline).
- **Evaluation:** The "Compute Only" curve will generally stay closer to the ideal speedup, while the "Wall Time" curve will fall below it due to network communication overhead.
