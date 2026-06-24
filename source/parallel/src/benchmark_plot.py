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
    parser.add_argument("--out-png", default="benchmark_plot.png")
    parser.add_argument("--async-mode", action="store_true")
    return parser.parse_args()

def run_experiment(args, n_rounds):
    here = Path(__file__).resolve().parent
    # We use a unique log dir to easily find the CSV
    log_dir = here / "outputs" / f"benchmark_N_{n_rounds}"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        args.mpirun,
        "-np", str(args.np),
        sys.executable, str(here / "parallel_fedavg_mpi.py"),
        "--rounds", str(n_rounds),
        "--local-steps", str(args.local_steps),
        "--log-dir", str(log_dir)
    ]
    if args.async_mode:
        cmd.append("--async-mode")
        
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
            
        for r, times in rounds_data.items():
            total_compute_time += max(times) if times else 0.0
            
    return total_wall_time, total_compute_time

def main():
    args = parse_args()
    
    n_values = sorted(args.rounds)
    time_with_comm = []
    time_without_comm = []
    
    for n in n_values:
        total_time, compute_time = run_experiment(args, n)
        time_with_comm.append(total_time)
        time_without_comm.append(compute_time)
        print(f"N={n} -> Total Time (with comm): {total_time:.2f}s, Compute Time (without comm): {compute_time:.2f}s")
        
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(n_values, time_with_comm, marker='o', linewidth=2, label='Thời gian chạy (Có truyền thông - Total Wall Time)')
    plt.plot(n_values, time_without_comm, marker='s', linewidth=2, linestyle='--', label='Thời gian chạy (Không truyền thông - Compute Only)')
    
    # Add a horizontal band for 2-3 minutes (120 - 180 seconds)
    plt.axhspan(120, 180, color='green', alpha=0.2, label='Mục tiêu 2-3 phút (120s - 180s)')
    
    plt.title(f"Thời gian chạy theo Kích thước dữ liệu N (Rounds)\nSố lượng tiến trình: {args.np}", fontsize=14, fontweight='bold')
    plt.xlabel('Kích thước dữ liệu đầu vào N (Số Vòng - Rounds)', fontsize=12)
    plt.ylabel('Thời gian (Giây)', fontsize=12)
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
