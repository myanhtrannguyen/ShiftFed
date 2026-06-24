import matplotlib.pyplot as plt

worker_counts = [1, 2, 4, 8, 16]
times_with_comm = [31.66, 35.71, 67.09, 210.25, 506.15]
times_without_comm = [2.17, 2.17, 2.47, 7.34, 20.28]

t1_with = times_with_comm[0]
t1_without = times_without_comm[0]

speedup_with = [t1_with / t for t in times_with_comm]
speedup_without = [t1_without / t for t in times_without_comm]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Biểu đồ 1: Thời gian chạy
ax1.plot(worker_counts, times_with_comm, 'o-', color='salmon', linewidth=2, label='With Communication (Wall Time)')
ax1.plot(worker_counts, times_without_comm, 's--', color='skyblue', linewidth=2, label='Without Communication (Compute Only)')
ax1.set_xlabel('Number of Processes (Workers)', fontsize=12)
ax1.set_ylabel('Execution Time (Seconds)', fontsize=12)
ax1.set_title('Execution Time vs Number of Processes', fontsize=14, fontweight='bold')
ax1.set_xticks(worker_counts)
ax1.grid(True, linestyle=':', alpha=0.7)
ax1.legend(fontsize=10)

# Biểu đồ 2: Độ tăng tốc (Speedup)
ax2.plot(worker_counts, speedup_with, 'o-', color='salmon', linewidth=2, label='Speedup (With Comm)')
ax2.plot(worker_counts, speedup_without, 's--', color='skyblue', linewidth=2, label='Speedup (Compute Only)')
ax2.plot(worker_counts, worker_counts, 'k:', linewidth=2, label='Ideal Speedup (S = p)')
ax2.set_xlabel('Number of Processes (Workers)', fontsize=12)
ax2.set_ylabel('Speedup Factor', fontsize=12)
ax2.set_title('Speedup vs Number of Processes', fontsize=14, fontweight='bold')
ax2.set_xticks(worker_counts)
ax2.grid(True, linestyle=':', alpha=0.7)
ax2.legend(fontsize=10)

plt.tight_layout()
plt.savefig('speedup_plot_manual.png', dpi=300)
print("Đã vẽ xong biểu đồ và lưu thành speedup_plot_manual.png")
