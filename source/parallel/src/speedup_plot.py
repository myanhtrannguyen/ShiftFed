import argparse
import subprocess
import sys
import time
from pathlib import Path
import matplotlib.pyplot as plt
import csv

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--x-cores", type=int, default=4, help="X = physical cores of the machine")
    parser.add_argument("--n", type=int, default=5000, help="N data size (base steps)")
    parser.add_argument("--mpirun", default="mpirun")
    parser.add_argument("--out-png", default="speedup_plot.png")
    return parser.parse_args()

def run_experiment(np, total_steps, mpirun):
    here = Path(__file__).resolve().parent
    log_dir = here / "outputs" / "speedup_check"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    workers = np - 1
    local_steps = total_steps // workers
    
    cmd = [
        mpirun,
        "-np", str(np),
        sys.executable, str(here / "parallel_fedavg_mpi.py"),
        "--rounds", "3",  # Keep rounds small, we just want to measure speedup of local computation
        "--local-steps", str(local_steps),
        "--log-dir", str(log_dir)
    ]
    
    start_time = time.perf_counter()
    subprocess.run(cmd, check=True)
    total_time = time.perf_counter() - start_time
    
    csv_files = sorted(log_dir.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    compute_time = 0.0
    if csv_files:
        with open(csv_files[0], "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rounds_data = {}
            for row in reader:
                r = int(row["round"])
                c_time = float(row.get("compute_time", 0.0))
                if r not in rounds_data:
                    rounds_data[r] = []
                rounds_data[r].append(c_time)
            for r, times in rounds_data.items():
                compute_time += max(times) if times else 0.0
                
    return total_time, compute_time

def main():
    args = parse_args()
    
    # Số tiến trình công nhân biến đổi từ 1, 2, 4, 8, ..., 2X
    X = args.x_cores
    max_workers = 2 * X
    
    worker_counts = []
    w = 1
    while w <= max_workers:
        worker_counts.append(w)
        w *= 2
        
    total_data = 2 * args.n
    print(f"Tổng kích thước dữ liệu mô phỏng = 2*N = {total_data}")
    print(f"Số lượng nhân vật lý (X) = {X}")
    print(f"Chạy nghiệm với số tiến trình công nhân: {worker_counts}")
    
    times_with_comm = []
    times_without_comm = []
    
    for w in worker_counts:
        np = w + 1 # Tổng số tiến trình thực tế của MPI = 1 Server + w Workers
        print(f"\n[*] Đang chạy với {w} workers (Tổng tiến trình MPI={np})...")
        t_total, t_compute = run_experiment(np, total_data, args.mpirun)
        times_with_comm.append(t_total)
        times_without_comm.append(t_compute)
        print(f"    -> Có truyền thông (Total Time): {t_total:.2f}s")
        print(f"    -> Không truyền thông (Compute Only): {t_compute:.2f}s")
        
    # Tính toán độ tăng tốc (Speedup)
    # Speedup = Time(1 worker) / Time(w workers)
    t1_with = times_with_comm[0]
    t1_without = times_without_comm[0]
    
    speedup_with = [t1_with / t for t in times_with_comm]
    speedup_without = [t1_without / t for t in times_without_comm]
    
    # Vẽ biểu đồ
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Biểu đồ 1: Thời gian chạy
    ax1.plot(worker_counts, times_with_comm, 'o-', color='salmon', linewidth=2, label='Có truyền thông (Wall Time)')
    ax1.plot(worker_counts, times_without_comm, 's--', color='skyblue', linewidth=2, label='Không truyền thông (Compute Only)')
    ax1.set_xlabel('Số lượng tiến trình (Workers)', fontsize=12)
    ax1.set_ylabel('Thời gian chạy (Giây)', fontsize=12)
    ax1.set_title('Thời gian chạy theo số lượng tiến trình', fontsize=14, fontweight='bold')
    ax1.set_xticks(worker_counts)
    ax1.grid(True, linestyle=':', alpha=0.7)
    ax1.legend(fontsize=10)
    
    # Biểu đồ 2: Độ tăng tốc (Speedup)
    ax2.plot(worker_counts, speedup_with, 'o-', color='salmon', linewidth=2, label='Speedup (Có truyền thông)')
    ax2.plot(worker_counts, speedup_without, 's--', color='skyblue', linewidth=2, label='Speedup (Không truyền thông)')
    ax2.plot(worker_counts, worker_counts, 'k:', linewidth=2, label='Tăng tốc lý tưởng (Ideal Speedup = p)')
    ax2.set_xlabel('Số lượng tiến trình (Workers)', fontsize=12)
    ax2.set_ylabel('Độ tăng tốc', fontsize=12)
    ax2.set_title('Biểu đồ Độ tăng tốc (Speedup)', fontsize=14, fontweight='bold')
    ax2.set_xticks(worker_counts)
    ax2.grid(True, linestyle=':', alpha=0.7)
    ax2.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(args.out_png, dpi=300)
    print(f"\n[+] Đã lưu biểu đồ thành công vào file: {args.out_png}")

if __name__ == "__main__":
    main()
