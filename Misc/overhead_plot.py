import numpy as np
import matplotlib.pyplot as plt
import math  # For rounding up the values

font = {
        # 'weight' : 'bold',
        'size'   : 18
        }

plt.rc('font', **font)

# Data for the current application (with three trials) - Temp sense
# cpu_without_log_app1 = [602387375, 602371701, 603169147]
# cpu_with_log_app1 = [639124218, 640963293, 638680750]
power_without_log_app1 = [20, 20, 20]
power_with_log_app1 = [22, 22, 22]
power_with_det_app1 = [23, 23, 23]
ram_without_log_app1 = [27018, 27027, 27029]
ram_with_log_app1 = [42739, 41982, 42352]
ram_wit_det_app1 = [44051.68, 45840, 44470]

# Generate synthetic data for two other applications (for demonstration purposes) - Mamba
# cpu_without_log_app2 = [60500000, 60490000, 60510000]
# cpu_with_log_app2 = [64500000, 64490000, 64510000]
power_without_log_app2 = [18.59, 18.76,	18.84]
power_with_log_app2 = [19.78, 19.64, 19.75]
power_with_det_app2 = [19.78, 19.71, 19.88]
ram_without_log_app2 = [41225, 41226, 41240]
ram_with_log_app2 = [52384, 52420, 52314]
ram_with_det_app2 = [62665.71, 62814.72, 62848.43]

# Contiki-MAC
# cpu_without_log_app3 = [561409971, 561448535, 561517995]
# cpu_with_log_app3 = [561398716, 561374182, 561402948]
power_without_log_app3 = [9, 9, 10]
power_with_log_app3 = [12, 11, 11]
power_with_det_app3 = [13, 12, 12]
ram_without_log_app3 = [27227, 27243, 27244]
ram_with_log_app3 = [53889, 53882, 54036]
ram_with_det_app3 = [65420.4, 65594, 66154.2]

# Combine the data for all three applications
# cpu_without_log = [np.mean(cpu_without_log_app1), np.mean(cpu_without_log_app2), np.mean(cpu_without_log_app3)]
# cpu_with_log = [np.mean(cpu_with_log_app1), np.mean(cpu_with_log_app2), np.mean(cpu_with_log_app3)]

power_without_log = [np.mean(power_without_log_app1), np.mean(power_without_log_app2), np.mean(power_without_log_app3)]
power_with_log = [np.mean(power_with_log_app1), np.mean(power_with_log_app2), np.mean(power_with_log_app3)]
power_with_det = [np.mean(power_with_det_app1), np.mean(power_with_det_app2), np.mean(power_with_det_app3)]

ram_without_log = [np.mean(ram_without_log_app1), np.mean(ram_without_log_app2), np.mean(ram_without_log_app3)]
ram_with_log = [np.mean(ram_with_log_app1), np.mean(ram_with_log_app2), np.mean(ram_with_log_app3)]
ram_with_det = [np.mean(ram_wit_det_app1), np.mean(ram_with_det_app2), np.mean(ram_with_det_app3)]

source_without_log = [53.752, 41.401, 36]
source_with_log = [74.71, 68.96, 76,23]
source_with_det = [82.329, 78.471, 86.29]

# X labels for applications
# labels = ['Temp-Sensor', 'MaMBA', 'Contiki-MAC']
# labels = ['Power\nConsumption', 'RAM\nusage', 'Source Code\nSize']

# x = np.arange(len(labels))
# bar_width = 0.3  # Reduced bar width to make bars thinner

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
# fig, (ax1) = plt.subplots(1, 1, figsize=(8, 6))  # Adjusted figure size to make graphs less wide

# # Plot CPU count comparison (removed error bars)
# bars1 = ax1.bar(x - bar_width/2, cpu_without_log, bar_width, label='Without Log', color='skyblue')
# bars2 = ax1.bar(x + bar_width/2, cpu_with_log, bar_width, label='With Log', color='coral')
# ax1.set_ylabel('CPU ticks')
# # ax1.set_title('CPU ticks Comparison Across Applications')
# ax1.set_xticks(x)
# ax1.set_xticklabels(labels)
# ax1.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=2)

# Add bar labels
# add_scientific_labels(ax1, bars1)
# add_scientific_labels(ax1, bars2)

# # Adjust layout and show plot
# plt.tight_layout()
# plt.show()

# Create subplots for Power consumption
# fig, (ax2) = plt.subplots(1, 1, figsize=(8, 6))

# # Plot Power consumption comparison (removed error bars)
# bars1 = ax2.bar(x - bar_width/2, power_without_log, bar_width, label='Without Log', color='lightgreen')
# bars2 = ax2.bar(x + bar_width/2, power_with_log, bar_width, label='With Log', color='orange')
# ax2.set_ylabel('mAH')
# # ax2.set_title('Power Consumption Comparison Across Applications')
# ax2.set_xticks(x)
# ax2.set_xticklabels(labels)
# ax2.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=2)

# # Add bar labels
# add_bar_labels(ax2, bars1)
# add_bar_labels(ax2, bars2)

# # Adjust layout and show plot
# plt.tight_layout()
# plt.show()

# # Create subplots for RAM usage
# fig, (ax3) = plt.subplots(1, 1, figsize=(8, 6))

# # Plot RAM usage comparison (removed error bars)
# bars1 = ax3.bar(x - bar_width/2, ram_without_log, bar_width, label='Without Log', color='lightpink')
# bars2 = ax3.bar(x + bar_width/2, ram_with_log, bar_width, label='With Log', color='lightseagreen')
# ax3.set_ylabel('Bytes')
# # ax3.set_title('RAM Usage (heap) Comparison Across Applications')
# ax3.set_xticks(x)
# ax3.set_xticklabels(labels)
# ax3.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=2)

# # Add bar labels
# add_bar_labels(ax3, bars1)
# add_bar_labels(ax3, bars2)

# # Adjust layout and show plot
# plt.tight_layout()
# plt.show()


# # Create subplots for RAM usage
# fig, (ax4) = plt.subplots(1, 1, figsize=(8, 6))

# # Plot RAM usage comparison (removed error bars)
# bars1 = ax4.bar(x - bar_width/2, source_without_log, bar_width, label='Without Log', color='lightcoral')
# bars2 = ax4.bar(x + bar_width/2, source_with_log, bar_width, label='With Log', color='cornflowerblue')
# ax4.set_ylabel('Kilo-Bytes (KB)')
# # ax4.set_title('Source Code Size Comparison Across Applications')
# ax4.set_xticks(x)
# ax4.set_xticklabels(labels)
# ax4.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=2)

# # Add bar labels
# add_bar_labels(ax4, bars1)
# add_bar_labels(ax4, bars2)

# # Adjust layout and show plot
# plt.tight_layout()
# plt.show()


########################### Overhead Calculation ########################################
# Calculate overhead for each metric
# cpu_overhead = [(w - wo) / wo * 100 for w, wo in zip(cpu_with_log, cpu_without_log)]
power_overhead = [(w - wo) / wo * 100 for w, wo in zip(power_with_log, power_without_log)]
ram_overhead = [(w - wo) / wo * 100 for w, wo in zip(ram_with_log, ram_without_log)]
source_overhead = [(w - wo) / wo * 100 for w, wo in zip(source_with_log, source_without_log)]

# Metrics and Applications
# metrics = ['CPU Count', 'Power Consumption', 'RAM Usage', 'Source Code Size']
metrics = ['Power Consumption', 'RAM Usage', 'Source Code Size']
app_labels = ['Temp-Sensor', 'MaMBA', 'Contiki-MAC']

# Combine overhead data
# overhead_data = [cpu_overhead, power_overhead, ram_overhead, source_overhead]
overhead_data = [power_overhead, ram_overhead, source_overhead]

# Plotting the overhead percentages
x = np.arange(len(metrics))  # the label locations
width = 0.25  # the width of the bars

fig, ax = plt.subplots(figsize=(10, 6))

# Plot bars for each application
rects1 = ax.bar(x - width, [overhead[0] for overhead in overhead_data], width, label='Temp-Sensor')
rects2 = ax.bar(x, [overhead[1] for overhead in overhead_data], width, label='MaMBA')
rects3 = ax.bar(x + width, [overhead[2] for overhead in overhead_data], width, label='Contiki-MAC')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Overhead (%)')
# ax.set_title('VarLogger Overhead')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()

# Add bar labels
def add_bar_labels(bars):
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.1f}', ha='center', va='bottom')

add_bar_labels(rects1)
add_bar_labels(rects2)
add_bar_labels(rects3)

# Adding horizontal grid lines
# plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.title('Logging Mode ')
plt.tight_layout()
plt.show()

###### Detection + logging overhead ######
power_overhead_det = [(w - wo) / wo * 100 for w, wo in zip(power_with_det, power_without_log)]
ram_overhead_det = [(w - wo) / wo * 100 for w, wo in zip(ram_with_det, ram_without_log)]
source_overhead_det = [(w - wo) / wo * 100 for w, wo in zip(source_with_det, source_without_log)]

overhead_data_det = [power_overhead_det, ram_overhead_det, source_overhead_det]
# Plotting the overhead percentages
x = np.arange(len(metrics))  # the label locations
width = 0.25  # the width of the bars

fig, ax = plt.subplots(figsize=(10, 6))

# Plot bars for each application
rects1 = ax.bar(x - width, [overhead[0] for overhead in overhead_data_det], width, label='Temp-Sensor')
rects2 = ax.bar(x, [overhead[1] for overhead in overhead_data_det], width, label='MaMBA')
rects3 = ax.bar(x + width, [overhead[2] for overhead in overhead_data_det], width, label='Contiki-MAC')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Overhead (%)')
# ax.set_title('VarLogger Overhead')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()

# Add bar labels
def add_bar_labels(bars):
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.1f}', ha='center', va='bottom')

add_bar_labels(rects1)
add_bar_labels(rects2)
add_bar_labels(rects3)

# Adding horizontal grid lines
# plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.title('Detection Mode')
plt.tight_layout()
plt.show()



####################################################################################


# ### Application wise plots: log vs det+log plots ###

# # Add bar labels
# def add_bar_labels(bars):
#     for bar in bars:
#         yval = bar.get_height()
#         ax.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.1f}', ha='center', va='bottom')

# ## Plotting separate figures for each application
# for i, app_label in enumerate(app_labels):
#     x = np.arange(len(metrics))  # Metric indices
#     width = 0.35  # Bar width
    
#     fig, ax = plt.subplots(figsize=(10, 6))

#     # plt.figure(figsize=(8, 5))
    
#     # Bar plots
#     rect1 = ax.bar(x - width / 2, [overhead_data[j][i] for j in range(len(metrics))], width, label='Logging', color='darkred')
#     rect2 = ax.bar(x + width / 2, [overhead_data_det[j][i] for j in range(len(metrics))], width, label='Detection+Logging', color='darkblue')
    
#     add_bar_labels(rect1)
#     add_bar_labels(rect2)
#     # Titles and labels
#     plt.title(app_label)
#     plt.xticks(x, metrics)
#     plt.ylabel('Overhead (%)')
#     plt.xlabel('Metrics')
#     # plt.grid(axis='y', linestyle='--', alpha=0.7)
#     plt.legend()
    
#     # Adjust layout and display
#     plt.tight_layout()
#     plt.show()



#########################################################################################


## plot all info in a single plot ##

# Setup
x = np.arange(len(metrics))  # Metric indices
width = 0.2  # Width of each bar
colors = ['darkred', 'darkblue', 'darkgreen']

fig, ax = plt.subplots(figsize=(12, 8))

# Plot bars for each application
for i, app_label in enumerate(app_labels):
    rect1 = ax.bar(
        x + (i - 1) * width,
        [overhead[i] for overhead in overhead_data],
        width,
        label=f'{app_label} - Logging',
        color=colors[i]
    )
    rect2 = ax.bar(
        x + (i - 1) * width,
        [overhead_det[i] - overhead[i] for overhead, overhead_det in zip(overhead_data, overhead_data_det)],
        width,
        bottom=[overhead[i] for overhead in overhead_data],
        label=f'{app_label} - Detection',
        color=colors[i],
        alpha=0.6
    )

    # Add labels to the bars
    def add_bar_labels(bars, data):
        for bar, value in zip(bars, data):
            if value <= 5:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + bar.get_y() + 9,
                    f'{value:.1f}',
                    ha='center',
                    va='bottom',
                    fontsize=18
                )
            else:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + bar.get_y(),
                    f'{value:.1f}',
                    ha='center',
                    va='bottom',
                    fontsize=18
                )

    add_bar_labels(rect1, [overhead[i] for overhead in overhead_data])
    add_bar_labels(rect2, [overhead_det[i] - overhead[i] for overhead, overhead_det in zip(overhead_data, overhead_data_det)])

# Titles and labels
ax.set_title('Overhead Comparison for Logging and Detection Across Applications', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=18)
ax.set_ylabel('Overhead (%)', fontsize=18)
ax.set_xlabel('Metrics', fontsize=18)

# Add legend
ax.legend(fontsize='small')

# Grid and layout
ax.yaxis.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()