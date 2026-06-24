import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--np", type=int, default=12, help="Number of MPI processes")
    parser.add_argument("--rounds", type=int, default=30, help="Fixed N (rounds) for the program")
    parser.add_argument("--local-steps", type=int, default=100, help="Local steps (granularity)")
    parser.add_argument("--mpirun", default="mpirun")
    parser.add_argument("--out-png", default="granularity_plot.png")
    parser.add_argument("--async-mode", action="store_true")
    parser.add_argument("--load-balance", action="store_true", help="Enable dynamic load balancing")
    return parser.parse_args()

def run_experiment(args):
    here = Path(__file__).resolve().parent
    log_dir = here / "outputs" / "granularity_check"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        args.mpirun,
        "-np", str(args.np),
        sys.executable, str(here / "parallel_fedavg_mpi.py"),
        "--rounds", str(args.rounds),
        "--local-steps", str(args.local_steps),
        "--log-dir", str(log_dir)
    ]
    if args.async_mode:
        cmd.append("--async-mode")
    if args.load_balance:
        cmd.append("--load-balance")
        
    print(f"Running command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    
    csv_files = sorted(log_dir.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not csv_files:
        print("Warning: No CSV output found.")
        return {}

    csv_path = csv_files[0]
    
    rank_stats = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            r = int(row["rank"])
            if r not in rank_stats:
                rank_stats[r] = {"compute_time": 0.0, "communication_time": 0.0}
            rank_stats[r]["compute_time"] += float(row.get("compute_time", 0.0))
            rank_stats[r]["communication_time"] += float(row.get("communication_time", 0.0))
            
    return rank_stats

def main():
    args = parse_args()
    
    rank_stats = run_experiment(args)
    if not rank_stats:
        return
        
    ranks = sorted(rank_stats.keys())
    compute_times = [rank_stats[r]["compute_time"] for r in ranks]
    comm_times = [max(0.0, rank_stats[r]["communication_time"]) for r in ranks] # Ensure no negative
    
    # Check load balancing condition
    idle_times = comm_times
    max_idle = max(idle_times)
    min_idle = min(idle_times)
    is_balanced = True
    diff_percent = 0.0
    
    if max_idle > 0:
        diff_percent = ((max_idle - min_idle) / max_idle) * 100
        if diff_percent > 25.0:
            is_balanced = False

    # Plotting
    fig, ax = plt.subplots(figsize=(12, 7))
    
    x = np.arange(len(ranks))
    width = 0.6
    
    p1 = ax.bar(x, compute_times, width, label='Thời gian tính toán (Compute Time)', color='skyblue', edgecolor='black')
    p2 = ax.bar(x, comm_times, width, bottom=compute_times, label='Thời gian truyền thông/Rảnh rỗi (Comm/Idle Time)', color='salmon', edgecolor='black')
    
    ax.set_ylabel('Thời gian (Giây)', fontsize=12)
    ax.set_xlabel('Tiến trình (Rank)', fontsize=12)
    mode_str = "CÓ Load Balance" if args.load_balance else "KHÔNG Load Balance"
    ax.set_title(f'Biểu đồ Thời gian chạy của từng Tiến trình ({mode_str})\nĐộ lệch thời gian rảnh lớn nhất: {diff_percent:.2f}%', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f"Client {r}" for r in ranks])
    ax.legend(fontsize=11)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add values on bars
    for i in range(len(ranks)):
        ax.text(i, compute_times[i]/2, f"{compute_times[i]:.1f}s", ha='center', va='center', color='black', fontsize=9)
        if comm_times[i] > 1.0: # Only text if space is large enough
            ax.text(i, compute_times[i] + comm_times[i]/2, f"{comm_times[i]:.1f}s", ha='center', va='center', color='black', fontsize=9)
            
    plt.tight_layout()
    plt.savefig(args.out_png, dpi=300)
    print(f"\n[+] Đã lưu biểu đồ vào: {args.out_png}")
    
    print("\n" + "="*50)
    print("PHÂN TÍCH CÂN BẰNG TẢI (LOAD BALANCING ANALYSIS)")
    print("="*50)
    print(f"Độ lệch thời gian rảnh rỗi lớn nhất: {diff_percent:.2f}%")
    
    if is_balanced:
        print("Kết luận: HỆ THỐNG ĐÃ CÂN BẰNG TẢI TỐT.")
        print("Mức độ chênh lệch < 25%, độ mịn (granularity) hiện tại là phù hợp.")
    else:
        print("Kết luận: HỆ THỐNG MẤT CÂN BẰNG TẢI!")
        print("Mức độ chênh lệch > 25%, có sự lãng phí tài nguyên lớn ở một số tiến trình.")
        print("\nĐỀ XUẤT ĐIỀU CHỈNH ĐỘ MỊN (GRANULARITY):")
        print("1. Chỉnh độ mịn tinh hơn (Finer granularity):")
        print(f"   Giảm --local-steps xuống thấp hơn (VD: --local-steps {max(10, args.local_steps//2)}) để các máy giao tiếp thường xuyên hơn.")
        print("2. Dùng Load Balancing tự động:")
        print("   Hãy thêm cờ --load-balance khi chạy lệnh để Server tự động tính toán độ mịn (số step) riêng cho từng tiến trình dựa trên tốc độ thực tế của chúng.")
        
if __name__ == "__main__":
    main()
