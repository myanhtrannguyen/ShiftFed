"""Automate FL timing benchmarks and write a summary CSV."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run sequential and MPI FL benchmarks.")
    parser.add_argument("--rounds", nargs="+", type=int, default=[10, 20, 50])
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["lenet5", "cnn", "mlp"],
        default=["lenet5", "mlp"],
    )
    parser.add_argument("--local-steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--log-dir", default="./outputs/fl_mpi")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--mpirun", default="mpirun")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--async-mode", action="store_true")
    return parser.parse_args()


def run_command(command: list[str]) -> float:
    start = time.perf_counter()
    print("+ " + " ".join(command))
    subprocess.run(command, check=True)
    return time.perf_counter() - start


def main() -> None:
    args = parse_args()
    here = Path(__file__).resolve().parent
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    summary_path = log_dir / "benchmark_summary.csv"
    rows = []

    for model in args.models:
        for rounds in args.rounds:
            common = [
                "--rounds",
                str(rounds),
                "--local-steps",
                str(args.local_steps),
                "--batch-size",
                str(args.batch_size),
                "--model",
                model,
                "--data-dir",
                args.data_dir,
                "--log-dir",
                args.log_dir,
            ]
            if args.synthetic:
                common.append("--synthetic")
            if args.download:
                common.append("--download")

            seq_time = run_command([sys.executable, str(here / "sequential_fl.py"), *common])
            mpi_command = [
                args.mpirun,
                "-np",
                "4",
                sys.executable,
                str(here / "parallel_fedavg_mpi.py"),
                *common,
            ]
            if args.async_mode:
                mpi_command.append("--async-mode")
            mpi_time = run_command(mpi_command)
            rows.append(
                {
                    "model": model,
                    "rounds": rounds,
                    "local_steps": args.local_steps,
                    "sequential_wall_time": seq_time,
                    "mpi_wall_time": mpi_time,
                    "speedup": seq_time / mpi_time if mpi_time > 0 else 0.0,
                    "async_mode": args.async_mode,
                }
            )

    with summary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Benchmark summary written to {summary_path}")


if __name__ == "__main__":
    main()
