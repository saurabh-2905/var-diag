import matplotlib.pyplot as plt

# Data from the table
events = [100, 200, 500, 700, 1000, 1500, 2000]
buffer_size = [1.56, 3.1, 7.98, 11.98, 15.98, 23.98, 31.98]
time_to_write = [1.82, 1.94, 2.01, 2.09, 2.06, 2.26, 2.49]

# Create the figure and axis objects
fig, ax1 = plt.subplots()

# Plot buffer size on the first y-axis
color = 'tab:blue'
ax1.set_xlabel('Number of Events')
ax1.set_ylabel('Buffer size (kB)', color=color)
ax1.plot(events, buffer_size, color=color, marker='o', label="Buffer Size")
ax1.tick_params(axis='y', labelcolor=color)

# Create a second y-axis for the time to write
ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('Time to write (sec)', color=color)
ax2.plot(events, time_to_write, color=color, marker='s', label="Time to Write")
ax2.tick_params(axis='y', labelcolor=color)

# Add title and show grid
# plt.title('Buffer Size and Time to Write vs Number of Events')
ax1.grid(True)

# Show the plot
plt.show()
