import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--np", type=int, default=12, help="Number of MPI processes")
    parser.add_argument("--rounds", type=int, nargs="+", default=[5, 10, 20, 30, 40], help="List of N (rounds) to test")
    parser.add_argument("--local-steps", type=int, default=100)
    parser.add_argument("--mpirun", default="mpirun")
    parser.add_argument("--hostfile", default="", help="Path to hostfile for MPI cluster")
    parser.add_argument("--oversubscribe", action="store_true", help="Allow running more processes than physical cores")
    parser.add_argument("--out-png", default="benchmark_plot.png")
    parser.add_argument("--async-mode", action="store_true")
    parser.add_argument("--load-balance", action="store_true", help="Enable dynamic load balancing")
    parser.add_argument("--plot-only", action="store_true", help="Skip MPI run and only plot from existing CSVs")
    return parser.parse_args()

def run_experiment(args, n_rounds):
    # Use relative paths to avoid breaking symlinks across different MPI nodes
    log_dir = Path("outputs") / f"benchmark_N_{n_rounds}"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        args.mpirun,
        "-np", str(args.np)
    ]
    if args.hostfile:
        cmd.extend(["--hostfile", args.hostfile])
    if args.oversubscribe:
        cmd.append("--oversubscribe")
        
    cmd.extend([
        sys.executable, "parallel_fedavg_mpi.py",
        "--rounds", str(n_rounds),
        "--local-steps", str(args.local_steps),
        "--log-dir", str(log_dir),
        "--download"
    ])
    if args.async_mode:
        cmd.append("--async-mode")
    if args.load_balance:
        cmd.append("--load-balance")
        
    if args.plot_only:
        print(f"Skipping MPI run for N={n_rounds}, reading from CSV...")
        total_wall_time = 0.0
    else:
        print(f"Running N={n_rounds} with command: {' '.join(cmd)}")
        start_time = time.perf_counter()
        subprocess.run(cmd, check=True)
        total_wall_time = time.perf_counter() - start_time
    
    # Read the CSV to calculate Compute Time (without communication)
    csv_files = list(log_dir.glob("*.csv"))
    if not csv_files:
        print("Warning: No CSV output found.")
        return total_wall_time, 0.0

    csv_path = csv_files[0]
    total_compute_time = 0.0
    
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        
        # Calculate compute time without communication
        # For each round, find the max compute_time among all clients
        rounds_data = {}
        for row in rows:
            r = int(row["round"])
            c_time = float(row.get("compute_time", 0.0))
            if r not in rounds_data:
                rounds_data[r] = []
            rounds_data[r].append(c_time)
            
            if args.plot_only and "total_time" in row:
                total_wall_time = max(total_wall_time, float(row["total_time"]))
                
        for r, times in rounds_data.items():
            total_compute_time += max(times) if times else 0.0
            
    return total_wall_time, total_compute_time

def main():
    args = parse_args()
    
    n_values = sorted(args.rounds)
    time_with_comm = []
    time_without_comm = []
    
    actual_n_values = []
    
    try:
        for n in n_values:
            total_time, compute_time = run_experiment(args, n)
            time_with_comm.append(total_time)
            time_without_comm.append(compute_time)
            actual_n_values.append(n)
            print(f"N={n} -> Total Time (with comm): {total_time:.2f}s, Compute Time (without comm): {compute_time:.2f}s")
    except KeyboardInterrupt:
        print("\n[!] Bị ngắt bởi người dùng (Ctrl+C). Đang tiến hành vẽ biểu đồ với các dữ liệu đã thu thập được...")
        if not time_with_comm:
            print("Chưa có dữ liệu nào được thu thập. Hủy vẽ biểu đồ.")
            sys.exit(0)
            
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(actual_n_values, time_with_comm, marker='o', linewidth=2, label='Total Time (With Communication)')
    plt.plot(actual_n_values, time_without_comm, marker='s', linewidth=2, linestyle='--', label='Compute Time (Without Communication)')
    
    # Add a horizontal band for 2-3 minutes (120 - 180 seconds)
    plt.axhspan(120, 180, color='green', alpha=0.2, label='Target 2-3 minutes (120s - 180s)')
    
    plt.title(f"Execution Time vs Data Size N (Rounds)\nNumber of Processes: {args.np}", fontsize=14, fontweight='bold')
    plt.xlabel('Data Size N (Number of Rounds)', fontsize=12)
    plt.ylabel('Time (Seconds)', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(fontsize=10)
    plt.tight_layout()
    
    plt.savefig(args.out_png, dpi=300)
    print(f"\nĐã lưu biểu đồ vào file: {args.out_png}")
    
    # Find optimal N
    for i, t in enumerate(time_with_comm):
        if 120 <= t <= 180:
            print(f"-> N = {n_values[i]} phù hợp với yêu cầu thời gian 2-3 phút (Thực tế: {t:.2f}s).")
            return
            
    print("-> Không có N nào rơi vào đúng khoảng 120-180s trong tập thử nghiệm. Bạn có thể nhìn biểu đồ để nội suy giá trị N phù hợp.")

if __name__ == "__main__":
    main()
