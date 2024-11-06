import matplotlib.pyplot as plt

# Data from the table
anomalies = [
    "Failed Tx.\n(Temp-Sensor)", "bursty sensor", "bitflip", 
    "faulty sensor", "bitflip", "node out of sync"
]
st_dr = [0, 100, 100, 56.25, 3.7, 33.33]   # Detection Rates for ST
ei_dr = [78.26, 100, 100, 100, 100, 91.67]  # Detection Rates for EI

# X-axis values
x = range(1, len(anomalies) + 1)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(x, st_dr, label="ST", marker='o', color='blue')
plt.plot(x, ei_dr, label="EI", marker='s', color='green')

# Adding labels and title
plt.xlabel("Anomalies")
plt.ylabel("Detection Rate (DR)")
plt.title("Detection Rate Comparison for ST and EI")
plt.xticks(ticks=x, labels=anomalies, rotation=0, ha="center")
plt.legend()

# Adding horizontal grid lines
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Displaying values at each point
for i, (x_val, st_val, ei_val) in enumerate(zip(x, st_dr, ei_dr)):
    plt.text(x_val, st_val + 2, f"{st_val}%", ha='center', color='blue', fontsize=9)
    plt.text(x_val, ei_val + 2, f"{ei_val}%", ha='center', color='green', fontsize=9)

# Display the plot
plt.tight_layout()
plt.show()
