# import matplotlib.pyplot as plt
# import numpy as np

# # Font settings
# font = {
#     'size': 14
# }
# plt.rc('font', **font)

# # Data from the table
# anomalies = [
#     "Failed Tx.\n(Temp-Sensor)", "Bursty sensor\n(Temp-Sensor)", "Bitflip\n(Temp-Sensor)", 
#     "Faulty sensor\n(MaMBA)", "Bitflip\n(MaMBA)", "Node out of sync\n(Contiki-MAC)"
# ]
# st_dr = [0, 100, 100, 56.25, 3.7, 33.33]   # Detection Rates for ST
# ei_dr = [78.26, 100, 100, 100, 100, 91.67]  # Detection Rates for EI

# # X-axis values
# x = np.arange(len(anomalies))  # Positions of bars on X-axis

# # Bar width
# bar_width = 0.35

# # Plotting
# plt.figure(figsize=(10, 6))
# plt.bar(x - bar_width/2, st_dr, bar_width, label="ST", color='darkred', edgecolor='black', alpha=0.8) # ,edgecolor='black'
# plt.bar(x + bar_width/2, ei_dr, bar_width, label="EI", color='darkblue', edgecolor='black', alpha=0.8) # , edgecolor='black'

# # Adding labels and title
# plt.xlabel("Anomalies", fontsize=16)
# plt.ylabel("Detection Rate (Anomalywise)", fontsize=16)
# plt.xticks(ticks=x, labels=anomalies, rotation=45, ha="right")
# plt.legend()

# # Adding horizontal grid lines
# plt.grid(axis='y', linestyle='--', alpha=0.7)

# # Displaying values on top of each bar
# for i, (st_val, ei_val) in enumerate(zip(st_dr, ei_dr)):
#     plt.text(i - bar_width/2, st_val + 2, f"{st_val:.1f}", ha='center', color='red', fontsize=12)
#     plt.text(i + bar_width/2, ei_val + 2, f"{ei_val:.1f}", ha='center', color='blue', fontsize=12)

# # Display the plot
# plt.tight_layout()
# plt.show()


##################################################################################

#### modified 30/06/2025

import matplotlib.pyplot as plt
import numpy as np

# Font settings
font = {
    'size': 17
}
plt.rc('font', **font)

# Fault types (including repeated "Bitflip" renamed for clarity)
fault_types = [
    "Failed Tx.",
    "Bursty \nSensor",
    "Bitflip \n(Temp-\nSensor)",
    "Faulty \nSensor \nData",
    "Bitflip \n(MaMBA)",
    "Node \nOut of \nSync"
]

# Models
models = ["ST-2", "ST-30", "EI", "LSTM", "GRU", "LSTM+CNN"]

# F1-scores
data = [
    [0.0,   1.0, 0.78, 0.86, 1.0, 0.65],   # Failed Tx.
    [1.0,   1.0, 1.0,  0.75, 0.77, 0.52],  # Bursty Sensor
    [1.0,   1.0, 1.0,  0.98, 0.98, 0.61],  # Bitflip-1
    [0.93,  1.0, 1.0,  1.0,  0.85, 1.0],   # Faulty Sensor Data
    [0.037, 0.96, 1.0, 1.0,  0.96, 1.0],   # Bitflip-2
    [0.33,  1.0, 0.91, 1.0,  1.0,  0.91]   # Node Out of Sync
]

colours = [ '#304994', '#4FCEE9', '#F3691D', '#F2CF2A', '#A560A6', '#4D8434' ]
# X-axis positions
x = np.arange(len(fault_types))
width = 0.13  # width of the bars

# Create plot
fig, ax = plt.subplots(figsize=(12, 6))

# Plot bars for each model
for i in range(len(models)):
    model_scores = [row[i] for row in data]
    ax.bar(x + (i - 2.5) * width, model_scores, width, label=models[i], color=colours[i], edgecolor='black', alpha=1)

# Labels, title, etc.
ax.set_ylabel('Recall')
ax.set_xlabel('Anomalies')
# ax.set_title('F1-Score per Fault Type by Detection Method')
ax.set_xticks(x)
ax.set_xticklabels(fault_types, rotation=0)
ax.set_ylim(0, 1.1)
ax.legend(title='Detection Methods', loc='upper left', bbox_to_anchor=(1, 1))
ax.grid(True, axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()


##################################################################################

# import matplotlib.pyplot as plt
# import numpy as np

# # Font settings
# font = {
#     'size': 18
# }
# plt.rc('font', **font)

# # Fault types (including repeated "Bitflip" renamed for clarity)
# fault_types = [
#     "Failed Tx.",
#     "Bursty Sensor",
#     "Bitflip (Temp-Sensor)",
#     "Faulty Sensor Data",
#     "Bitflip (MaMBA)",
#     "Node Out of Sync"
# ]

# # Models
# models = ["ST-2", "ST-30", "EI", "LSTM", "GRU", "LSTM+CNN"]

# # F1-scores
# data = [
#     [0.0,   1.0, 0.78, 0.86, 1.0, 0.65],   # Failed Tx.
#     [1.0,   1.0, 1.0,  0.75, 0.77, 0.52],  # Bursty Sensor
#     [1.0,   1.0, 1.0,  0.98, 0.98, 0.61],  # Bitflip (Temp-Sensor)
#     [0.93,  1.0, 1.0,  1.0,  0.85, 1.0],   # Faulty Sensor Data
#     [0.037, 0.96, 1.0, 1.0,  0.96, 1.0],   # Bitflip (MaMBA)
#     [0.33,  1.0, 0.91, 1.0,  1.0,  0.91]   # Node Out of Sync
# ]

# # Transpose data for easier access by model
# data_T = list(zip(*data))  # Now: one row per model

# # Colors for each model
# colours = ['#304994', '#4FCEE9', '#F3691D', '#F2CF2A', '#A560A6', '#4D8434']

# # Y-axis positions (one per fault type)
# y = np.arange(len(fault_types))
# height = 0.13  # Height of each horizontal bar

# # Create plot
# fig, ax = plt.subplots(figsize=(12, 7))

# # Plot bars for each model
# for i in range(len(models)):
#     model_scores = data_T[i]
#     ax.barh(y + (i - 2.5) * height, model_scores, height,
#             label=models[i], color=colours[i], edgecolor='black', alpha=1)

# # Labels and formatting
# ax.set_xlabel('F1-Score')
# ax.set_ylabel('Fault Type')
# ax.set_yticks(y)
# ax.set_yticklabels(fault_types)
# ax.set_xlim(0, 1.1)
# ax.legend(title='Detection Methods', loc='upper center', bbox_to_anchor=(0.5, 1.15),
#           ncol=3, fontsize=14)
# ax.grid(True, axis='x', linestyle='--', alpha=0.5)

# plt.tight_layout()
# plt.show()