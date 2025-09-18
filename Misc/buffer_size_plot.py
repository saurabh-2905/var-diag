import matplotlib.pyplot as plt

FONTSIZE = 16

# Data from the table
events = [100, 200, 500, 700, 1000, 1500, 2000]
buffer_size = [1.56, 3.1, 7.98, 11.98, 15.98, 23.98, 31.98]
time_to_write = [1.82, 1.94, 2.01, 2.09, 2.06, 2.26, 2.49]

# Create the figure and axis objects
fig, ax1 = plt.subplots()

# Plot buffer size on the first y-axis
color = 'tab:blue'
# ax1.set_xlabel('Number of Events', fontsize=FONTSIZE)
ax1.set_xlabel('Buffer Size (no. of events)', fontsize=FONTSIZE)
# ax1.set_ylabel('Buffer size (kB)', color=color, fontsize=FONTSIZE)
ax1.set_ylabel('RAM Req. (kB)', color=color, fontsize=FONTSIZE)
ln1 = ax1.plot(events, buffer_size, color=color, marker='o', markersize=8, label="RAM Req.",)
ax1.tick_params(axis='y', labelcolor=color, labelsize=FONTSIZE)
ax1.tick_params(axis='x', labelsize=FONTSIZE)
# ax1.legend(loc=(0.005, 0.8))

# Create a second y-axis for the time to write
ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('Time to write (sec)', color=color, fontsize=FONTSIZE)
ln2 = ax2.plot(events, time_to_write, color=color, marker='s', markersize=8, label="Time to Write")
ax2.tick_params(axis='y', labelcolor=color, labelsize=FONTSIZE)
# ax2.legend(loc=(0.005, 0.87))

### combine legends for both axes
lns = ln1+ln2
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc='upper left')
# Add title and show grid
# plt.title('Buffer Size and Time to Write vs Number of Events')
# plt.legend(loc='upper left')
# plt.rcParams.update({'font.size': 22})
ax1.grid(True)

# Show the plot
plt.tight_layout()
plt.show()
