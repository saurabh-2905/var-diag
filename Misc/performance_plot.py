import numpy as np
import matplotlib.pyplot as plt

# Applications
applications = ['Temp-Sensor', 'MaMBA', 'Contiki-MAC']

# Performance data for "State Transition" model
state_transition_f1 = [0.91, 0.36, 0.33]
state_transition_recall = [0.83, 0.50, 0.33]
state_transition_precision = [1.00, 0.28, 0.33]

# Performance data for "Exe. Interval" model (diff_val=5)
exe_interval_f1 = [0.94, 0.85, 0.95]
exe_interval_recall = [0.90, 1.00, 0.90]
exe_interval_precision = [1.00, 0.75, 1.00]

# Plot for State Transition Model
x = np.arange(len(applications))  # the label locations
width = 0.25  # the width of the bars

fig, ax = plt.subplots(figsize=(10, 6))

# State Transition bars
rects1 = ax.bar(x - width, state_transition_f1, width, label='F1-Score (State Transition)', color='lightblue')
rects2 = ax.bar(x, state_transition_recall, width, label='Recall (State Transition)', color='lightgreen')
rects3 = ax.bar(x + width, state_transition_precision, width, label='Precision (State Transition)', color='lightcoral')

# Adding labels and title
ax.set_ylabel('Performance Metrics')
ax.set_title('Performance of State Transition Model')
ax.set_xticks(x)
ax.set_xticklabels(applications)
ax.legend()

# Bar labels
def add_labels(rects):
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2, height, f'{height:.2f}', ha='center', va='bottom')

add_labels(rects1)
add_labels(rects2)
add_labels(rects3)

plt.tight_layout()
plt.show()

# Plot for Exe. Interval Model
fig, ax = plt.subplots(figsize=(10, 6))

# Exe Interval bars
rects4 = ax.bar(x - width, exe_interval_f1, width, label='F1-Score (Exe. Interval)', color='cornflowerblue')
rects5 = ax.bar(x, exe_interval_recall, width, label='Recall (Exe. Interval)', color='seagreen')
rects6 = ax.bar(x + width, exe_interval_precision, width, label='Precision (Exe. Interval)', color='salmon')

# Adding labels and title
ax.set_ylabel('Performance Metrics')
ax.set_title('Performance of Exe. Interval Model (diff_val=5)')
ax.set_xticks(x)
ax.set_xticklabels(applications)
ax.legend()

# Bar labels
add_labels(rects4)
add_labels(rects5)
add_labels(rects6)

plt.tight_layout()
plt.show()
