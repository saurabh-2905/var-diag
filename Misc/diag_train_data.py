import matplotlib.pyplot as plt
import numpy as np

# Data
use_cases = ['Temp-Sensor', 'MaMBA', 'Contiki-MAC']
total_samples = np.array([2136, 28644, 13863])
unique_samples = np.array([436, 22121, 8862])
duplicate_samples = total_samples - unique_samples

x = np.arange(len(use_cases))
width = 0.5

# Create figure
fig, ax = plt.subplots(figsize=(7, 5))

# Plot stacked bars
bars_unique = ax.bar(x, unique_samples, width, label='Unique Samples', color='tab:orange')
bars_dup = ax.bar(x, duplicate_samples, width, bottom=unique_samples,
                  label='Duplicate Samples', color='tab:blue', alpha=0.6)

# Labels and title
ax.set_xlabel('Use Case', fontsize=12)
ax.set_ylabel('Number of Samples', fontsize=12)
ax.set_title('Total Reference Samples per Use Case', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(use_cases, fontsize=11)
ax.grid(True, axis='y', linestyle='--', alpha=0.6)

# Add percentage text inside bars
for i in range(len(x)):
    total = total_samples[i]
    unique_pct = (unique_samples[i] / total) * 100
    dup_pct = (duplicate_samples[i] / total) * 100

    # Annotate unique and duplicate proportions
    ax.text(x[i], unique_samples[i]/2, f'{unique_pct:.1f}%', 
            ha='center', va='center', color='black', fontsize=9, fontweight='bold')
    ax.text(x[i], unique_samples[i] + duplicate_samples[i]/2, f'{dup_pct:.1f}%',
            ha='center', va='center', color='black', fontsize=9, fontweight='bold')

    # Total label above bar
    ax.text(x[i], total + max(total_samples)*0.02, 
            f'Total: {total:,}', ha='center', fontsize=9)

# Adjust plot area first
# plt.subplots_adjust(right=0.85)  # leave space on the right for legend

# ✅ Add legend outside (visible)
plt.legend(
    loc='lower center', bbox_to_anchor=(0.5, -0.25), ncol=2,
    fontsize=10,
)
plt.tight_layout()
plt.show()
