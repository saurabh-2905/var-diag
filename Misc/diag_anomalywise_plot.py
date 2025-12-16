import matplotlib.pyplot as plt
import numpy as np

# Font settings
font = { 'size': 17 }
plt.rc('font', **font)

# Fault types (including repeated "Bitflip" renamed for clarity)
fault_types = [
    "Failed Tx.",
    "Bursty \nSensor",
    "Bitflip \n(Smart-\nBuilding)",
    "Faulty \nSensor \nData",
    "Bitflip \n(Habitat)",
    # "Unhandled \nInterrupt",
    "Node \nOut of \nSync"
]

# Two methods
models = ["Dustminer", "VarDiag"]

# F1-scores
data = [
    [0.86, 0.9],   # Failed Tx.
    [0.25, 1.0],   # Bursty Sensor
    [1.0, 1.0],   # Bitflip-1
    [0.0, 1.0],   # Faulty Sensor Data
    [0.59, 0.92],  # Bitflip-2
    # [0.0, 1.0],   # Unhandled External Interrupt
    [1.0, 1.0]    # Node Out of Sync
]

# Colors (you can keep your original palette)
colours = ['#A6290D', '#4F7302']

# X-axis positions
x = np.arange(len(fault_types))

# Width of each bar
width = 0.32   # Wider because only 2 bars per group

# Create plot
fig, ax = plt.subplots(figsize=(12, 6))

# ----- Proper spacing for 2 models -----
# model 0 → x - width/2
# model 1 → x + width/2
offsets = [-width/2, width/2]

for i in range(len(models)):
    model_scores = [row[i] for row in data]
    ax.bar(x + offsets[i], model_scores, width,
           label=models[i],
           color=colours[i],
           edgecolor='black')

# Labels
ax.set_ylabel('Recall')
ax.set_xlabel('Anomalies')
ax.set_xticks(x)
ax.set_xticklabels(fault_types, rotation=0)
ax.set_ylim(0, 1.1)

# Legend
ax.legend(title='Detection Methods', loc='upper left', bbox_to_anchor=(1, 1))

# Gridlines
ax.grid(True, axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()



