import matplotlib.pyplot as plt
import numpy as np

# Font settings
font = { 'size': 14 }
plt.rc('font', **font)

#### diff colours for different configurations, different marks for top-10 vs full

# Left-side configurations
top_10 = {
    "ml=10, ms=0.1": {
        "Temp-Sensor": (0.47, 0.96, 0.31),
        "MaMBA": (0.04, 0.06, 0.03),
        "Contiki-MAC": (0.07, 0.16, 0.05)
    },
    "ml=15, ms=0.1": {
        "Temp-Sensor": (0.47, 0.96, 0.31),
        "MaMBA": (0.08, 0.13, 0.05),
        "Contiki-MAC": (0.07, 0.16, 0.05)
    },
    "ml=15, ms=0.05": {
        "Temp-Sensor": (0.4, 0.96, 0.25),
        "MaMBA": (0.24, 0.52, 0.16),
        "Contiki-MAC": (0.19, 1.0, 0.16)
    },
    "ml=15, ms=0.01": {
        "Temp-Sensor": (0.4, 0.96, 0.25),
        "MaMBA": (0.17, 0.4, 0.11),
        "Contiki-MAC": (0.19, 1.0, 0.11)
    },
    "ml=20, ms=0.01": {
        "Temp-Sensor": (0.4, 0.96, 0.25),
        "MaMBA": (0.2, 0.47, 0.13),
        "Contiki-MAC": (0.19, 1.0, 0.1)
    }
}

# Right-side configurations
top_full = {
    "ml=10, ms=0.1": {
        "Temp-Sensor": (0.39, 0.97, 0.24),
        "MaMBA": (0.02, 0.06, 0.01),
        "Contiki-MAC": (0.05, 0.16, 0.03)
    },
    "ml=15, ms=0.1": {
        "Temp-Sensor": (0.39, 0.97, 0.24),
        "MaMBA": (0.05, 0.13, 0.03),
        "Contiki-MAC": (0.05, 0.16, 0.03)
    },
    "ml=15, ms=0.05": {
        "Temp-Sensor": (0.32, 0.97, 0.19),
        "MaMBA": (0.12, 0.61, 0.06),
        "Contiki-MAC": (0.12, 1.0, 0.05)
    },
    "ml=15, ms=0.01": {
        "Temp-Sensor": (0.32, 0.97, 0.19),
        "MaMBA": (0.08, 0.5, 0.04),
        "Contiki-MAC": (0.1, 1.0, 0.05)
    },
    "ml=20, ms=0.01": {
        "Temp-Sensor": (0.32, 0.97, 0.19),
        "MaMBA": (0.09, 0.56, 0.05),
        "Contiki-MAC": (0.1, 1.0, 0.05)
    }
}

colours = ["#A6290D","#4F7302","#1F77B4", "#FF7F0E", "#2CA02C"]

# Applications and metrics
apps = ["Temp-Sensor", "MaMBA", "Contiki-MAC"]
metrics = ["F1", "Recall", "Precision"]
metric_idx = {"F1": 0, "Recall": 1, "Precision": 2}
markers = {"F1": "o", "Recall": "s", "Precision": "^"}

configs = list(top_10.keys())
x = np.arange(len(configs))

def plot_application(app):
    plt.figure(figsize=(7, 8))

    for m in metrics:
        y_top10 = [top_10[cfg][app][metric_idx[m]] for cfg in configs]
        y_topfull = [top_full[cfg][app][metric_idx[m]] for cfg in configs]

        # top_10
        plt.scatter(
            x,
            y_top10,
            marker=markers[m],
            s=100,
            # linestyle="--",
            label=f"{m} (top-10)",
            color=colours[0]
        )

    for m in metrics:
        y_top10 = [top_10[cfg][app][metric_idx[m]] for cfg in configs]
        y_topfull = [top_full[cfg][app][metric_idx[m]] for cfg in configs]

        # top_full
        plt.scatter(
            x,
            y_topfull,
            marker=markers[m],
            s=100,
            # linestyle="-",
            label=f"{m} (full)",
            color=colours[1]
        )

    plt.xticks(x, configs, rotation=30, ha="right")
    plt.ylabel("Score", fontsize=font['size'] + 2)
    plt.xlabel("Configuration: max_len (ml), min_sup (ms)", fontsize=font['size'] + 2)
    plt.title(app)
    plt.grid(True, linestyle="--", alpha=0.5)

    # Clean legend (no duplicates)
    handles, labels = plt.gca().get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    plt.legend(unique.values(), unique.keys(), ncol=2, loc='lower center', bbox_to_anchor=(0.5, 1.07))

    plt.tight_layout()
    plt.show()


# Generate plots
for app in apps:
    plot_application(app)
