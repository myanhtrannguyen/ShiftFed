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
    parser.add_argument("--hostfile", default="", help="Path to hostfile for MPI cluster")
    parser.add_argument("--oversubscribe", action="store_true", help="Allow running more processes than physical cores")
    parser.add_argument("--out-png", default="granularity_plot.png")
    parser.add_argument("--async-mode", action="store_true")
    parser.add_argument("--load-balance", action="store_true", help="Enable dynamic load balancing")
    return parser.parse_args()

def run_experiment(args):
    # Use relative paths to avoid breaking symlinks across different MPI nodes
    log_dir = Path("outputs") / f"granularity_steps_{args.local_steps}"
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
        "--rounds", str(args.rounds),
        "--local-steps", str(args.local_steps),
        "--log-dir", str(log_dir),
        "--download"
    ])
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
    
    p1 = ax.bar(x, compute_times, width, label='Compute Time', color='skyblue', edgecolor='black')
    p2 = ax.bar(x, comm_times, width, bottom=compute_times, label='Communication/Idle Time', color='salmon', edgecolor='black')
    
    ax.set_ylabel('Time (Seconds)', fontsize=12)
    ax.set_xlabel('Client Rank', fontsize=12)
    mode_str = "WITH Load Balancing" if args.load_balance else "NO Load Balancing"
    ax.set_title(f'Execution Time by Client ({mode_str})\nMax Idle Time Difference: {diff_percent:.2f}%', fontsize=14, fontweight='bold')
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
    print(f"\n[+] Saved plot to: {args.out_png}")
    
    print("\n" + "="*50)
    print("LOAD BALANCING ANALYSIS")
    print("="*50)
    print(f"Max idle time difference: {diff_percent:.2f}%")
    
    if is_balanced:
        print("Conclusion: SYSTEM IS WELL BALANCED.")
        print("Difference < 25%, current granularity is appropriate.")
    else:
        print("Conclusion: SYSTEM IS IMBALANCED!")
        print("Difference > 25%, significant resource waste on some clients.")
        print("\nSUGGESTED GRANULARITY ADJUSTMENTS:")
        print("1. Finer granularity:")
        print(f"   Decrease --local-steps (e.g. --local-steps {max(10, args.local_steps//2)}) to communicate more frequently.")
        print("2. Automatic Load Balancing:")
        print("   Use the --load-balance flag to let the Server dynamically assign steps based on actual client speed.")
        
if __name__ == "__main__":
    main()
