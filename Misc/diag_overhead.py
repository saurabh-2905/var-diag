import matplotlib.pyplot as plt

# APP = 'mamba' 
APP = 'contiki'

plt.rcParams.update({
    "font.size": 18,          # default text size
    "axes.titlesize": 24,     # title size
    "axes.labelsize": 24,     # x/y labels
    "xtick.labelsize": 20,    # tick labels
    "ytick.labelsize": 20,
    "legend.fontsize": 20,    # legend text
})

# Data from the table
# if APP == 'mamba':
#     events = [1000, 2000, 3000, 4000, 5000, 10000, 15000, 20000, 25000]
#     dust = [6557, 6568, 6762, 6689, 6645, 9175, 13576, 13496, 15162]
#     varl = [9882, 10681, 11829, 12440, 13613, 18923, 25020, 35400, 46010]
# elif APP == 'contiki':
#     events = [1000, 2000, 3000, 4000, 5000, 10000, 15000]
#     dust = [1262, 5298, 9128, 9229, 13442, 26345, 33180]
#     varl = [2061, 2432, 2750, 3390, 3930, 6732, 10104]

events_m = [1000, 2000, 3000, 4000, 5000, 10000, 15000, 20000, 25000]
dust_m = [6557, 6568, 6762, 6689, 6645, 9175, 13576, 13496, 15162]
varl_m = [9882, 10681, 11829, 12440, 13613, 18923, 25020, 35400, 46010]
events_c = [1000, 2000, 3000, 4000, 5000, 10000, 15000]
dust_c = [1262, 5298, 9128, 9229, 13442, 26345, 33180]
varl_c = [2061, 2432, 2750, 3390, 3930, 6732, 10104]

# Plot setup
plt.figure(figsize=(8, 7))
plt.plot(events_m, dust_m, 'o-', label='Dustminer-habitat', linewidth=3, markersize=8, color="#A6290D")
plt.plot(events_m, varl_m, 's--', label='VarDiag-habitat', linewidth=3, markersize=8, color="#A6290D")
plt.plot(events_c, dust_c, 'o-', label='Dustminer-contiki', linewidth=3, markersize=8, color="#4F7302")
plt.plot(events_c, varl_c, 's--', label='VarDiag-contiki', linewidth=3, markersize=8, color="#4F7302")

# Labels and title
plt.xlabel('No. of Events (Reference Data)', fontsize=20)
plt.ylabel('Time (ms)', fontsize=20)
plt.title('End-to-end Computation Overhead', fontsize=20)
plt.legend(loc='upper center', fontsize=16, bbox_to_anchor=(0.5, -0.15), fancybox=True, ncol=2)
plt.grid(True, linestyle='--', alpha=0.6)

# Optionally use logarithmic scale if the difference is large
# plt.yscale('log')

# Show plot
plt.tight_layout()
plt.show()
