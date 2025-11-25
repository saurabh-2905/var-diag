import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


plt.rcParams.update({
    "font.size": 20,          # default text size
    "axes.titlesize": 28,     # title size
    "axes.labelsize": 28,     # x/y labels
    "xtick.labelsize": 22,    # tick labels
    "ytick.labelsize": 22,
    "legend.fontsize": 22,    # legend text
})


# ---------------------------------------------
# Define Data
# ---------------------------------------------
data = {
    "Application": [
        "Temp-Sensor","Temp-Sensor","Temp-Sensor","Temp-Sensor","Temp-Sensor","Temp-Sensor",
        "MaMBA","MaMBA","MaMBA","MaMBA","MaMBA","MaMBA",
        "Contiki-MAC","Contiki-MAC","Contiki-MAC","Contiki-MAC","Contiki-MAC","Contiki-MAC"
    ],
    "Model": [
        "ST-2","ST-30","EI","LSTM","GRU","LSTM+CNN",
        "ST-2","ST-30","EI","LSTM","GRU","LSTM+CNN",
        "ST-2","ST-30","EI","LSTM","GRU","LSTM+CNN"
    ],
    "F1": [
        0.72,0.97,0.94,0.53,0.68,0.47,
        0.35,0.51,0.85,1.0,0.31,0.83,
        0.33,0.68,0.95,0.59,0.5,0.62
    ],
    "pdPrecision": [
        0.92,0.24,0.29,0.26,0.17,0.25,
        0.45,0.36,0.26,0.05,0.24,0.04,
        0.88,0.41,0.19,0.06,0.06,0.12
    ]
}

df = pd.DataFrame(data)

applications = df["Application"].unique()
models = df["Model"].unique()

# X positions for applications
x = np.arange(len(applications))
width = 0.1  # bar width

# ---------------------------------------------
# Helper function to annotate bars
# ---------------------------------------------
def add_value_labels(ax):
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f"{height:.2f}",
                        (p.get_x() + p.get_width()/2, height),
                        ha='center', va='bottom', fontsize=16)


# ---------------------------------------------
# Plot 1 — F1-score
# ---------------------------------------------
plt.figure(figsize=(14,6))
ax = plt.gca()

for i, model in enumerate(models):
    model_data = df[df["Model"] == model].set_index("Application").loc[applications]
    bars = ax.bar(x + i*width, model_data["F1"], width, label=model)

# X-axis labels
plt.xticks(x + width*len(models)/2, applications, rotation=0)

plt.ylabel("F1-score")
# plt.title("F1-score Comparison Across Applications")
plt.grid(axis="y", linestyle="--", alpha=0.6)

plt.legend(title="Method", bbox_to_anchor=(1.05, 1), loc='upper left')

# # Add text labels
# add_value_labels(ax)

plt.tight_layout()
plt.show()


# ---------------------------------------------
# Plot 2 — pdPrecision
# ---------------------------------------------
plt.figure(figsize=(14,6))
ax = plt.gca()

for i, model in enumerate(models):
    model_data = df[df["Model"] == model].set_index("Application").loc[applications]
    bars = ax.bar(x + i*width, model_data["pdPrecision"], width, label=model)

# X-axis labels
plt.xticks(x + width*len(models)/2, applications, rotation=0)

plt.ylabel("pdPrecision")
# plt.title("pdPrecision Comparison Across Applications")
plt.grid(axis="y", linestyle="--", alpha=0.6)

plt.legend(title="Method", bbox_to_anchor=(1.05, 1), loc='upper left')

# # Add value labels
# add_value_labels(ax)

plt.tight_layout()
plt.show()
