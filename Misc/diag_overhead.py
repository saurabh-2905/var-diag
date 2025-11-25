import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 20,          # default text size
    "axes.titlesize": 28,     # title size
    "axes.labelsize": 28,     # x/y labels
    "xtick.labelsize": 22,    # tick labels
    "ytick.labelsize": 22,
    "legend.fontsize": 22,    # legend text
})

# Data from the table
events = [1000, 2000, 3000, 4000, 5000, 10000, 15000, 20000, 25000]
dust = [218, 390, 618, 680, 838, 1861, 3245, 4198, 5797]
varl = [0.5, 2, 3, 4, 6, 11, 25, 44, 66]

# Plot setup
plt.figure(figsize=(8, 5))
plt.plot(events, dust, 'o-', label='Dustminer', linewidth=2, markersize=6)
plt.plot(events, varl, 's--', label='VarDiag', linewidth=2, markersize=6)

# Labels and title
plt.xlabel('Reference Data (No. of Events)', fontsize=20)
plt.ylabel('Time (seconds)', fontsize=20)
plt.title('Computation Overhead', fontsize=20)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

# Optionally use logarithmic scale if the difference is large
plt.yscale('log')

# Show plot
plt.tight_layout()
plt.show()
