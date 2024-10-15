import numpy as np
import matplotlib.pyplot as plt

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
cpu_without_log_app3 = [561409971,	561448535,	561517995]
cpu_with_log_app3 = [561398716,	561374182,	561402948]
power_without_log_app3 = [9, 9, 10]
power_with_log_app3 = [12, 11, 11]
ram_without_log_app3 = [27227,	27243,	27244]
ram_with_log_app3 = [53889,	53882,	54036]

# Combine the data for all three applications
cpu_without_log = [np.mean(cpu_without_log_app1), np.mean(cpu_without_log_app2), np.mean(cpu_without_log_app3)]
cpu_with_log = [np.mean(cpu_with_log_app1), np.mean(cpu_with_log_app2), np.mean(cpu_with_log_app3)]
cpu_without_log_std = [np.std(cpu_without_log_app1), np.std(cpu_without_log_app2), np.std(cpu_without_log_app3)]
cpu_with_log_std = [np.std(cpu_with_log_app1), np.std(cpu_with_log_app2), np.std(cpu_with_log_app3)]

power_without_log = [np.mean(power_without_log_app1), np.mean(power_without_log_app2), np.mean(power_without_log_app3)]
power_with_log = [np.mean(power_with_log_app1), np.mean(power_with_log_app2), np.mean(power_with_log_app3)]
power_without_log_std = [np.std(power_without_log_app1), np.std(power_without_log_app2), np.std(power_without_log_app3)]
power_with_log_std = [np.std(power_with_log_app1), np.std(power_with_log_app2), np.std(power_with_log_app3)]

ram_without_log = [np.mean(ram_without_log_app1), np.mean(ram_without_log_app2), np.mean(ram_without_log_app3)]
ram_with_log = [np.mean(ram_with_log_app1), np.mean(ram_with_log_app2), np.mean(ram_with_log_app3)]
ram_without_log_std = [np.std(ram_without_log_app1), np.std(ram_without_log_app2), np.std(ram_without_log_app3)]
ram_with_log_std = [np.std(ram_with_log_app1), np.std(ram_with_log_app2), np.std(ram_with_log_app3)]

# X labels for applications
labels = ['Temp-Sensor', 'MaMBA', 'Contiki-MAC']
x = np.arange(len(labels))
bar_width = 0.3  # Reduced bar width to make bars thinner

# Create subplots
fig, (ax1) = plt.subplots(1, 1, figsize=(6, 6))  # Adjusted figure size to make graphs less wide

# Plot CPU count comparison
ax1.bar(x - bar_width/2, cpu_without_log, bar_width, yerr=cpu_without_log_std, capsize=5, label='Without Log', color='skyblue')
ax1.bar(x + bar_width/2, cpu_with_log, bar_width, yerr=cpu_with_log_std, capsize=5, label='With Log', color='coral')
ax1.set_ylabel('CPU Count')
ax1.set_title('CPU Count Comparison Across Applications')
ax1.set_xticks(x)
ax1.set_xticklabels(labels)
ax1.legend(loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=2)

# Adjust layout and show plot
plt.tight_layout()
plt.show()

# Create subplots
fig, (ax2) = plt.subplots(1, 1, figsize=(6, 6))  # Adjusted figure size to make graphs less wide

# Plot Power consumption comparison
ax2.bar(x - bar_width/2, power_without_log, bar_width, yerr=power_without_log_std, capsize=5, label='Without Log', color='lightgreen')
ax2.bar(x + bar_width/2, power_with_log, bar_width, yerr=power_with_log_std, capsize=5, label='With Log', color='orange')
ax2.set_ylabel('Power (mAH)')
ax2.set_title('Power Consumption Comparison Across Applications')
ax2.set_xticks(x)
ax2.set_xticklabels(labels)
ax2.legend(loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=2)

# Adjust layout and show plot
plt.tight_layout()
plt.show()

# Create subplots
fig, (ax3) = plt.subplots(1, 1, figsize=(6, 6))  # Adjusted figure size to make graphs less wide

# Plot RAM usage comparison
ax3.bar(x - bar_width/2, ram_without_log, bar_width, yerr=ram_without_log_std, capsize=5, label='Without Log', color='lightpink')
ax3.bar(x + bar_width/2, ram_with_log, bar_width, yerr=ram_with_log_std, capsize=5, label='With Log', color='lightseagreen')
ax3.set_ylabel('RAM Usage (bytes)')
ax3.set_title('RAM Usage Comparison Across Applications')
ax3.set_xticks(x)
ax3.set_xticklabels(labels)
ax3.legend(loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=2)

# Adjust layout and show plot
plt.tight_layout()
plt.show()
