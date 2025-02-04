import matplotlib.pyplot as plt
import numpy as np

# Font settings
font = {
    'size': 14
}
plt.rc('font', **font)

# Data from the table
anomalies = [
    "Failed Tx.\n(Temp-Sensor)", "Bursty sensor\n(Temp-Sensor)", "Bitflip\n(Temp-Sensor)", 
    "Faulty sensor\n(MaMBA)", "Bitflip\n(MaMBA)", "Node out of sync\n(Contiki-MAC)"
]
st_dr = [0, 100, 100, 56.25, 3.7, 33.33]   # Detection Rates for ST
ei_dr = [78.26, 100, 100, 100, 100, 91.67]  # Detection Rates for EI

# X-axis values
x = np.arange(len(anomalies))  # Positions of bars on X-axis

# Bar width
bar_width = 0.35

# Plotting
plt.figure(figsize=(10, 6))
plt.bar(x - bar_width/2, st_dr, bar_width, label="ST", color='darkred', edgecolor='black', alpha=0.8) # ,edgecolor='black'
plt.bar(x + bar_width/2, ei_dr, bar_width, label="EI", color='darkblue', edgecolor='black', alpha=0.8) # , edgecolor='black'

# Adding labels and title
plt.xlabel("Anomalies", fontsize=16)
plt.ylabel("Detection Rate (Anomalywise)", fontsize=16)
plt.xticks(ticks=x, labels=anomalies, rotation=45, ha="right")
plt.legend()

# Adding horizontal grid lines
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Displaying values on top of each bar
for i, (st_val, ei_val) in enumerate(zip(st_dr, ei_dr)):
    plt.text(i - bar_width/2, st_val + 2, f"{st_val:.1f}", ha='center', color='red', fontsize=12)
    plt.text(i + bar_width/2, ei_val + 2, f"{ei_val:.1f}", ha='center', color='blue', fontsize=12)

# Display the plot
plt.tight_layout()
plt.show()
