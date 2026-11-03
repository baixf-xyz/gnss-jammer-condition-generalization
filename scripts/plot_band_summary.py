#!/usr/bin/env python3

import csv
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt


IN = Path(
    "exp/paper_figures/band_multiseed_summary.csv"
)

OUT_DIR = Path(
    "exp/paper_figures"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# Load
# ============================================================

data = defaultdict(
    lambda: defaultdict(list)
)

with IN.open() as f:

    reader = csv.DictReader(f)

    for r in reader:

        rep = r["representation"]
        band = int(r["band"])
        acc = 100.0 * float(r["accuracy"])

        data[rep][band].append(acc)


representations = [
    "Magnitude",
    "Complex",
    "Mag+DeltaPhase",
]

display_names = [
    "Magnitude",
    "Complex",
    "Mag + ΔPhase",
]


# ============================================================
# Mean / std
# ============================================================

band1_mean = []
band1_std = []

band5_mean = []
band5_std = []

for rep in representations:

    b1 = np.array(
        data[rep][1],
        dtype=float
    )

    b5 = np.array(
        data[rep][5],
        dtype=float
    )

    band1_mean.append(
        b1.mean()
    )

    band1_std.append(
        b1.std(ddof=1)
    )

    band5_mean.append(
        b5.mean()
    )

    band5_std.append(
        b5.std(ddof=1)
    )


# ============================================================
# Plot
# ============================================================

x = np.arange(
    len(representations)
)

width = 0.36

fig, ax = plt.subplots(
    figsize=(8.5, 5.5)
)


bars1 = ax.bar(
    x - width / 2,
    band1_mean,
    width,
    yerr=band1_std,
    capsize=5,
    label="Band 1",
)

bars5 = ax.bar(
    x + width / 2,
    band5_mean,
    width,
    yerr=band5_std,
    capsize=5,
    label="Band 5",
)


ax.set_ylabel(
    "Accuracy (%)"
)

ax.set_xlabel(
    "Input representation"
)

ax.set_title(
    "Band-wise Generalization under Condition-Disjoint Evaluation"
)

ax.set_xticks(x)
ax.set_xticklabels(
    display_names
)

ax.set_ylim(
    0,
    110
)

ax.legend()


# ============================================================
# Value labels
# ============================================================

for bars, means in [
    (bars1, band1_mean),
    (bars5, band5_mean),
]:

    for bar, value in zip(
        bars,
        means
    ):

        ax.text(
            bar.get_x()
            + bar.get_width() / 2,
            bar.get_height() + 2.0,
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )


fig.tight_layout()


png = (
    OUT_DIR
    / "band_wise_mean_std.png"
)

pdf = (
    OUT_DIR
    / "band_wise_mean_std.pdf"
)


fig.savefig(
    png,
    dpi=300,
    bbox_inches="tight",
)

fig.savefig(
    pdf,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# Print paper table
# ============================================================

print()
print(
    f"{'Representation':18s} "
    f"{'Band 1':>22s} "
    f"{'Band 5':>22s} "
    f"{'Gap':>12s}"
)

print("-" * 80)

for i, rep in enumerate(
    display_names
):

    gap = (
        band1_mean[i]
        - band5_mean[i]
    )

    print(
        f"{rep:18s} "
        f"{band1_mean[i]:7.3f} +/- {band1_std[i]:6.3f}% "
        f"{band5_mean[i]:7.3f} +/- {band5_std[i]:6.3f}% "
        f"{gap:9.3f} pp"
    )


print()
print("Saved:")
print(" ", png)
print(" ", pdf)
