
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

# Display the plot
plt.tight_layout()
plt.show()