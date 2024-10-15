import numpy as np
import matplotlib.pyplot as plt

# Data for the current application (with three trials)
cpu_without_log_app1 = [602387375, 602371701, 603169147]
cpu_with_log_app1 = [639124218, 640963293, 638680750]
power_without_log_app1 = [20, 20, 20]
power_with_log_app1 = [22, 22, 22]
ram_without_log_app1 = [27018, 27027, 27029]
ram_with_log_app1 = [42739, 41982, 42352]

# Generate synthetic data for two other applications (for demonstration purposes)
cpu_without_log_app2 = [605000000, 604900000, 605100000]
cpu_with_log_app2 = [645000000, 644900000, 645100000]
power_without_log_app2 = [21, 21, 21]
power_with_log_app2 = [23, 23, 23]
ram_without_log_app2 = [28000, 28005, 28002]
ram_with_log_app2 = [43000, 43200, 43100]

cpu_without_log_app3 = [601000000, 601200000, 601500000]
cpu_with_log_app3 = [637000000, 636900000, 637200000]
power_without_log_app3 = [19, 19, 19]
power_with_log_app3 = [21, 21, 21]
ram_without_log_app3 = [26000, 26010, 26005]
ram_with_log_app3 = [42000, 42100, 42200]

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
labels = ['Application 1', 'Application 2', 'Application 3']
x = np.arange(len(labels))
bar_width = 0.3  # Reduced bar width to make bars thinner

# Create subplots
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(6, 10))  # Adjusted figure size to make graphs less wide

# Plot CPU count comparison
ax1.bar(x - bar_width/2, cpu_without_log, bar_width, yerr=cpu_without_log_std, capsize=5, label='Without Log', color='skyblue')
ax1.bar(x + bar_width/2, cpu_with_log, bar_width, yerr=cpu_with_log_std, capsize=5, label='With Log', color='coral')
ax1.set_ylabel('CPU Count')
ax1.set_title('CPU Count Comparison Across Applications')
ax1.set_xticks(x)
ax1.set_xticklabels(labels)
ax1.legend(loc='lower center', bbox_to_anchor=(0.5, -0.4), ncol=2)

# Plot Power consumption comparison
ax2.bar(x - bar_width/2, power_without_log, bar_width, yerr=power_without_log_std, capsize=5, label='Without Log', color='lightgreen')
ax2.bar(x + bar_width/2, power_with_log, bar_width, yerr=power_with_log_std, capsize=5, label='With Log', color='orange')
ax2.set_ylabel('Power (mAH)')
ax2.set_title('Power Consumption Comparison Across Applications')
ax2.set_xticks(x)
ax2.set_xticklabels(labels)
ax2.legend(loc='lower center', bbox_to_anchor=(0.5, -0.4), ncol=2)

# Plot RAM usage comparison
ax3.bar(x - bar_width/2, ram_without_log, bar_width, yerr=ram_without_log_std, capsize=5, label='Without Log', color='lightpink')
ax3.bar(x + bar_width/2, ram_with_log, bar_width, yerr=ram_with_log_std, capsize=5, label='With Log', color='lightseagreen')
ax3.set_ylabel('RAM Usage (bytes)')
ax3.set_title('RAM Usage Comparison Across Applications')
ax3.set_xticks(x)
ax3.set_xticklabels(labels)
ax3.legend(loc='lower center', bbox_to_anchor=(0.5, -0.4), ncol=2)

# Adjust layout and show plot
plt.tight_layout()
plt.show()
