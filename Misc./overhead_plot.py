import numpy as np
import matplotlib.pyplot as plt
import math  # For rounding up the values

# Data for the current application (with three trials) - Temp sense
cpu_without_log_app1 = [602387375, 602371701, 603169147]
cpu_with_log_app1 = [639124218, 640963293, 638680750]
power_without_log_app1 = [20, 20, 20]
power_with_log_app1 = [22, 22, 22]
ram_without_log_app1 = [27018, 27027, 27029]
ram_with_log_app1 = [42739, 41982, 42352]

# Generate synthetic data for two other applications (for demonstration purposes) - Mamba
cpu_without_log_app2 = [60500000, 60490000, 60510000]
cpu_with_log_app2 = [64500000, 64490000, 64510000]
power_without_log_app2 = [1, 1, 1]
power_with_log_app2 = [3, 3, 3]
ram_without_log_app2 = [2800, 2800, 2800]
ram_with_log_app2 = [4300, 4320, 4310]

# Contiki-MAC
cpu_without_log_app3 = [561409971, 561448535, 561517995]
cpu_with_log_app3 = [561398716, 561374182, 561402948]
power_without_log_app3 = [9, 9, 10]
power_with_log_app3 = [12, 11, 11]
ram_without_log_app3 = [27227, 27243, 27244]
ram_with_log_app3 = [53889, 53882, 54036]

# Combine the data for all three applications
cpu_without_log = [np.mean(cpu_without_log_app1), np.mean(cpu_without_log_app2), np.mean(cpu_without_log_app3)]
cpu_with_log = [np.mean(cpu_with_log_app1), np.mean(cpu_with_log_app2), np.mean(cpu_with_log_app3)]

power_without_log = [np.mean(power_without_log_app1), np.mean(power_without_log_app2), np.mean(power_without_log_app3)]
power_with_log = [np.mean(power_with_log_app1), np.mean(power_with_log_app2), np.mean(power_with_log_app3)]

ram_without_log = [np.mean(ram_without_log_app1), np.mean(ram_without_log_app2), np.mean(ram_without_log_app3)]
ram_with_log = [np.mean(ram_with_log_app1), np.mean(ram_with_log_app2), np.mean(ram_with_log_app3)]

source_without_log = [10.24, 16.9, 35.2]
source_with_log = [15.23, 27.4, 55]

# X labels for applications
labels = ['Temp-Sensor', 'MaMBA', 'Contiki-MAC']
x = np.arange(len(labels))
bar_width = 0.3  # Reduced bar width to make bars thinner

def add_bar_labels(ax, bars, round_up=True):
    """Add rounded labels on top of the bars."""
    for bar in bars:
        yval = bar.get_height()  # Round up the value
        if round_up:
            yval_round = round(bar.get_height(), 3)  # Round up the value
        else:
            yval_round = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval, f'{yval_round}', ha='center', va='bottom')

def add_scientific_labels(ax, bars):
    """Add labels on top of the bars in scientific notation."""
    for bar in bars:
        yval = bar.get_height()
        # Convert to scientific notation
        sci_label = f'{yval:.2e}'  # Format as scientific notation (e.g., 6.02e+08)
        ax.text(bar.get_x() + bar.get_width()/2, yval, sci_label, ha='center', va='bottom')


# Create subplots
fig, (ax1) = plt.subplots(1, 1, figsize=(8, 6))  # Adjusted figure size to make graphs less wide

# Plot CPU count comparison (removed error bars)
bars1 = ax1.bar(x - bar_width/2, cpu_without_log, bar_width, label='Without Log', color='skyblue')
bars2 = ax1.bar(x + bar_width/2, cpu_with_log, bar_width, label='With Log', color='coral')
ax1.set_ylabel('CPU ticks')
ax1.set_title('CPU ticks Comparison Across Applications')
ax1.set_xticks(x)
ax1.set_xticklabels(labels)
ax1.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=2)

# Add bar labels
add_scientific_labels(ax1, bars1)
add_scientific_labels(ax1, bars2)

# Adjust layout and show plot
plt.tight_layout()
plt.show()

# Create subplots for Power consumption
fig, (ax2) = plt.subplots(1, 1, figsize=(8, 6))

# Plot Power consumption comparison (removed error bars)
bars1 = ax2.bar(x - bar_width/2, power_without_log, bar_width, label='Without Log', color='lightgreen')
bars2 = ax2.bar(x + bar_width/2, power_with_log, bar_width, label='With Log', color='orange')
ax2.set_ylabel('mAH')
ax2.set_title('Power Consumption Comparison Across Applications')
ax2.set_xticks(x)
ax2.set_xticklabels(labels)
ax2.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=2)

# Add bar labels
add_bar_labels(ax2, bars1)
add_bar_labels(ax2, bars2)

# Adjust layout and show plot
plt.tight_layout()
plt.show()

# Create subplots for RAM usage
fig, (ax3) = plt.subplots(1, 1, figsize=(8, 6))

# Plot RAM usage comparison (removed error bars)
bars1 = ax3.bar(x - bar_width/2, ram_without_log, bar_width, label='Without Log', color='lightpink')
bars2 = ax3.bar(x + bar_width/2, ram_with_log, bar_width, label='With Log', color='lightseagreen')
ax3.set_ylabel('Bytes')
ax3.set_title('RAM Usage (heap) Comparison Across Applications')
ax3.set_xticks(x)
ax3.set_xticklabels(labels)
ax3.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=2)

# Add bar labels
add_bar_labels(ax3, bars1)
add_bar_labels(ax3, bars2)

# Adjust layout and show plot
plt.tight_layout()
plt.show()


# Create subplots for RAM usage
fig, (ax4) = plt.subplots(1, 1, figsize=(8, 6))

# Plot RAM usage comparison (removed error bars)
bars1 = ax4.bar(x - bar_width/2, source_without_log, bar_width, label='Without Log', color='lightcoral')
bars2 = ax4.bar(x + bar_width/2, source_with_log, bar_width, label='With Log', color='cornflowerblue')
ax4.set_ylabel('Kilo-Bytes (KB)')
ax4.set_title('Source Code Size Comparison Across Applications')
ax4.set_xticks(x)
ax4.set_xticklabels(labels)
ax4.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=2)

# Add bar labels
add_bar_labels(ax4, bars1)
add_bar_labels(ax4, bars2)

# Adjust layout and show plot
plt.tight_layout()
plt.show()