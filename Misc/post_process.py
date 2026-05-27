import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 18,          # default text size
    "axes.titlesize": 18,     # title size
    "axes.labelsize": 18,     # x/y labels
    "xtick.labelsize": 18,    # tick labels
    "ytick.labelsize": 18,
    "legend.fontsize": 18,    # legend text
})

# Data from the LaTeX table
diff_val = [0, 1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

fp_data = {
    "ST":        [32, 24, 21, 20, 15, 15, 15, 14, 12, 11, 9, 8, 5, 5, 5],
    "ST30":      [72, 47, 40, 38, 35, 29, 23, 16, 12, 10, 8, 5, 3, 3, 1],
    "EI":        [7, 7, 7, 6, 6, 5, 5, 3, 2, 2, 1, 0, 0, 0, 0],
    "LSTM":      [2162, 218, 10, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "GRU":       [374, 248, 228, 149, 74, 56, 22, 13, 9, 5, 3, 2, 2, 1, 1],
    "LSTM+CNN":  [505, 15, 15, 15, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
}

pdprecision_data = {
    "ST":        [1.00, 0.92, 0.78, 0.78, 0.45, 0.45, 0.45, 0.43, 0.37, 0.35, 0.38, 0.38, 0.23, 0.24, 0.24],
    "ST30":      [0.76, 0.72, 0.71, 0.42, 0.41, 0.36, 0.34, 0.32, 0.22, 0.24, 0.25, 0.18, 0.14, 0.13, 0.13],
    "EI":        [0.27, 0.27, 0.27, 0.27, 0.27, 0.26, 0.26, 0.25, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.21],
    "LSTM":      [0.80, 0.38, 0.10, 0.07, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
    "GRU":       [0.77, 0.73, 0.69, 0.46, 0.28, 0.24, 0.16, 0.13, 0.12, 0.08, 0.06, 0.06, 0.06, 0.06, 0.05],
    "LSTM+CNN":  [0.48, 0.06, 0.06, 0.06, 0.04, 0.04, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
}

# Fixed color mapping so the same model has the same color in both planes
model_colors = {
    "ST": "tab:blue",
    "ST30": "tab:orange",
    "EI": "tab:green",
    "LSTM": "tab:red",
    "GRU": "tab:purple",
    "LSTM+CNN": "tab:brown",
}

# Create two vertically separated subplots with shared x-axis
fig, (ax_fp, ax_pd) = plt.subplots(
    2, 1, figsize=(13, 9), sharex=True, gridspec_kw={"hspace": 0.15}
)

# Top plane: FP
for model, values in fp_data.items():
    ax_fp.plot(
        diff_val,
        values,
        marker="o",
        linewidth=2,
        markersize=5,
        color=model_colors[model],
        label=model,
    )

ax_fp.set_yscale('symlog', linthresh=1)
ax_fp.set_ylabel("FP")
# ax_fp.set_title("Impact of diff_val on FP and pdPrecision (MaMBA)", fontsize=14)
ax_fp.grid(True, linestyle="--", alpha=0.5)
# ax_fp.legend(loc="upper right", ncol=2)

# Bottom plane: pdPrecision
for model, values in pdprecision_data.items():
    ax_pd.plot(
        diff_val,
        values,
        marker="o",
        linewidth=2,
        markersize=5,
        color=model_colors[model],
        label=model,
    )

# Red horizontal line at pdPrecision = 0.1
# ax_pd.axhline(y=0.1, color="black", linestyle="--", linewidth=2, label="min acceptable")

ax_pd.set_xlabel("diff_val" )
ax_pd.set_ylabel("pdPrecision")
ax_pd.grid(True, linestyle="--", alpha=0.5)

# Optional axis limits
ax_pd.set_ylim(0, 1.05)

# Collect handles and labels from one axis (since both use same models)
handles, labels = ax_fp.get_legend_handles_labels()

# Add the red threshold line manually to legend
handles.append(ax_pd.axhline(y=0.1, color="black", linestyle="--", linewidth=2))
labels.append("pdPrecision cutoff (0.1)")

# Global legend at bottom
fig.legend(
    handles,
    labels,
    loc="center right",
    bbox_to_anchor=(1.0, 0.5),  # outside right
    fontsize=14
)

plt.subplots_adjust(right=0.76)  # increase bottom margin
plt.tight_layout(rect=[0, 0.2, 1, 1])
plt.savefig("fp_vs_pdprecision.png", dpi=300, bbox_inches="tight")
plt.show()