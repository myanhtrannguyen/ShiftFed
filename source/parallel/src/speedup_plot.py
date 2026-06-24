import argparse
import subprocess
import sys
import time
from pathlib import Path
import matplotlib.pyplot as plt
import csv
import json
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--x-cores", type=int, default=4, help="X = physical cores of the machine")
    parser.add_argument("--n", type=int, default=5000, help="N data size (base steps)")
    parser.add_argument("--mpirun", default="mpirun")
    parser.add_argument("--hostfile", default="", help="Path to hostfile for MPI cluster")
    parser.add_argument("--oversubscribe", action="store_true", help="Allow running more processes than physical cores")
    parser.add_argument("--out-png", default="speedup_plot.png")
    parser.add_argument("--async-mode", action="store_true")
    parser.add_argument("--load-balance", action="store_true", help="Enable dynamic load balancing")
    parser.add_argument("--plot-only", action="store_true", help="Only plot from previously saved speedup data")
    return parser.parse_args()

def run_experiment(np, total_steps, args):
    # Use relative paths to avoid breaking symlinks across different MPI nodes
    log_dir = Path("outputs") / "speedup_check"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    workers = np - 1
    local_steps = total_steps // workers
    
    cmd = [
        args.mpirun,
        "-np", str(np)
    ]
    if args.hostfile:
        cmd.extend(["--hostfile", args.hostfile])
    if args.oversubscribe:
        cmd.append("--oversubscribe")
        
    cmd.extend([
        sys.executable, "parallel_fedavg_mpi.py",
        "--rounds", "3",  # Keep rounds small, we just want to measure speedup of local computation
        "--local-steps", str(local_steps),
        "--log-dir", str(log_dir),
        "--download"
    ])
    if args.async_mode:
        cmd.append("--async-mode")
    if args.load_balance:
        cmd.append("--load-balance")
    
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
        
    # Đảm bảo 2X luôn có mặt ở cuối cùng theo đúng đề bài
    if worker_counts[-1] != max_workers:
        worker_counts.append(max_workers)
        
    total_data = 2 * args.n
    print(f"Tổng kích thước dữ liệu mô phỏng = 2*N = {total_data}")
    print(f"Số lượng nhân vật lý (X) = {X}")
    print(f"Chạy nghiệm với số tiến trình công nhân: {worker_counts}")
    
    results_file = Path("outputs") / "speedup_check" / "speedup_results.json"
    results_file.parent.mkdir(parents=True, exist_ok=True)
    
    times_with_comm = []
    times_without_comm = []
    actual_worker_counts = []
    
    if args.plot_only:
        print("[*] Chế độ vẽ lại (Plot Only) được bật. Đang đọc dữ liệu đã lưu...")
        if results_file.exists():
            with open(results_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                times_with_comm = data.get("times_with_comm", [])
                times_without_comm = data.get("times_without_comm", [])
                actual_worker_counts = data.get("actual_worker_counts", [])
            print(f"Đã load {len(actual_worker_counts)} mốc dữ liệu thành công.")
        else:
            print(f"[!] Không tìm thấy file dữ liệu: {results_file}")
            sys.exit(1)
    else:
        try:
            for w in worker_counts:
                np_val = w + 1 # Tổng số tiến trình thực tế của MPI = 1 Server + w Workers
                print(f"\n[*] Đang chạy với {w} workers (Tổng tiến trình MPI={np_val})...")
                t_total, t_compute = run_experiment(np_val, total_data, args)
                times_with_comm.append(t_total)
                times_without_comm.append(t_compute)
                actual_worker_counts.append(w)
                print(f"    -> Có truyền thông (Total Time): {t_total:.2f}s")
                print(f"    -> Không truyền thông (Compute Only): {t_compute:.2f}s")
                
                # TỰ ĐỘNG LƯU SAU MỖI BƯỚC CHẠY ĐỂ CHỐNG MẤT DỮ LIỆU
                with open(results_file, "w", encoding="utf-8") as f:
                    json.dump({
                        "times_with_comm": times_with_comm,
                        "times_without_comm": times_without_comm,
                        "actual_worker_counts": actual_worker_counts
                    }, f, indent=4)
                print("    -> Đã tự động lưu sao lưu kết quả.")
                
        except (KeyboardInterrupt, subprocess.CalledProcessError) as e:
            print(f"\n[!] Bị ngắt đột ngột (Lỗi máy ảo sập hoặc Ctrl+C).")
            print("[!] Đang tiến hành vẽ biểu đồ với các mốc đã chạy thành công...")
            
    if not times_with_comm:
        print("Chưa có mốc nào hoàn thành. Hủy vẽ biểu đồ.")
        sys.exit(0)
            
    # Tính toán độ tăng tốc (Speedup)
    # Speedup = Time(1 worker) / Time(w workers)
    t1_with = times_with_comm[0]
    t1_without = times_without_comm[0]
    
    speedup_with = [t1_with / t for t in times_with_comm]
    speedup_without = [t1_without / t for t in times_without_comm]
    
    # Vẽ biểu đồ
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Biểu đồ 1: Thời gian chạy
    ax1.plot(actual_worker_counts, times_with_comm, 'o-', color='salmon', linewidth=2, label='With Communication (Wall Time)')
    ax1.plot(actual_worker_counts, times_without_comm, 's--', color='skyblue', linewidth=2, label='Without Communication (Compute Only)')
    ax1.set_xlabel('Number of Processes (Workers)', fontsize=12)
    ax1.set_ylabel('Execution Time (Seconds)', fontsize=12)
    ax1.set_title('Execution Time vs Number of Processes', fontsize=14, fontweight='bold')
    ax1.set_xticks(actual_worker_counts)
    ax1.grid(True, linestyle=':', alpha=0.7)
    ax1.legend(fontsize=10)
    
    # Biểu đồ 2: Độ tăng tốc (Speedup)
    ax2.plot(actual_worker_counts, speedup_with, 'o-', color='salmon', linewidth=2, label='Speedup (With Comm)')
    ax2.plot(actual_worker_counts, speedup_without, 's--', color='skyblue', linewidth=2, label='Speedup (Compute Only)')
    ax2.plot(actual_worker_counts, actual_worker_counts, 'k:', linewidth=2, label='Ideal Speedup (S = p)')
    ax2.set_xlabel('Number of Processes (Workers)', fontsize=12)
    ax2.set_ylabel('Speedup Factor', fontsize=12)
    ax2.set_title('Speedup vs Number of Processes', fontsize=14, fontweight='bold')
    ax2.set_xticks(actual_worker_counts)
    ax2.grid(True, linestyle=':', alpha=0.7)
    ax2.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(args.out_png, dpi=300)
    print(f"\n[+] Đã lưu biểu đồ thành công vào file: {args.out_png}")

if __name__ == "__main__":
    main()
