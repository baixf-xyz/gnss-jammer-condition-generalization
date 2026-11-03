#!/usr/bin/env python3

import csv
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Inputs
# ============================================================

MANIFEST = Path(
    "data/gnss_jammer_3class_cond/test.list"
)

RUNS = [
    ("Magnitude", 42,
     Path("exp/rf_resnet34_3class_cond/phase_0/predictions.csv")),

    ("Magnitude", 43,
     Path("exp/rf_resnet34_3class_cond_seed43/eval_test_selected/predictions.csv")),

    ("Magnitude", 44,
     Path("exp/rf_resnet34_3class_cond_seed44/eval_test_selected/predictions.csv")),

    ("Complex", 42,
     Path("exp/rf_resnet34_complex_3class_cond/phase_0/predictions.csv")),

    ("Complex", 43,
     Path("exp/rf_resnet34_complex_3class_cond_seed43/eval_test_selected/predictions.csv")),

    ("Complex", 44,
     Path("exp/rf_resnet34_complex_3class_cond_seed44/eval_test_selected/predictions.csv")),

    ("Mag+DeltaPhase", 42,
     Path("exp/rf_resnet34_mag_phasediff_3class_cond/phase_0/predictions.csv")),

    ("Mag+DeltaPhase", 43,
     Path("exp/rf_resnet34_mag_phasediff_3class_cond_seed43/eval_test_selected/predictions.csv")),

    ("Mag+DeltaPhase", 44,
     Path("exp/rf_resnet34_mag_phasediff_3class_cond_seed44/eval_test_selected/predictions.csv")),
]

OUT_DIR = Path("exp/paper_figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Load manifest metadata
# ============================================================

meta = {}

with MANIFEST.open() as f:
    for line in f:
        x = json.loads(line)
        meta[x["key"]] = x


def condition(x):
    return (
        x["spk"],
        x["power"],
        x["bandwidth"],
        x["vga"],
        x["band"],
    )


# ============================================================
# Evaluate condition accuracy
# ============================================================

stats = {}

for rep, seed, pred_path in RUNS:

    if not pred_path.exists():
        raise FileNotFoundError(pred_path)

    by_cond = defaultdict(
        lambda: {
            "correct": 0,
            "total": 0,
        }
    )

    with pred_path.open() as f:
        reader = csv.DictReader(f)

        for r in reader:

            x = meta[r["key"]]
            c = condition(x)

            by_cond[c]["total"] += 1

            if r["true_label"] == r["pred_label"]:
                by_cond[c]["correct"] += 1

    stats[(rep, seed)] = by_cond


# ============================================================
# Fixed condition order
# ============================================================

conditions = [
    ("Meac", 40, -1, 100, 1),
    ("Meac", 40, -1, 100, 5),

    ("Prn", 47, 20, 100, 1),
    ("Prn", 47, 20, 100, 5),

    ("Spoof", 25, -1, 120, 1),
    ("Spoof", 25, -1, 120, 5),
    ("Spoof", 25, -1, 140, 1),
]


condition_labels = [
    "Meac | P40 | VGA100 | B1",
    "Meac | P40 | VGA100 | B5",

    "Prn | P47 | BW20 | VGA100 | B1",
    "Prn | P47 | BW20 | VGA100 | B5",

    "Spoof | P25 | VGA120 | B1",
    "Spoof | P25 | VGA120 | B5",
    "Spoof | P25 | VGA140 | B1",
]


columns = [
    ("Magnitude", 42),
    ("Magnitude", 43),
    ("Magnitude", 44),

    ("Complex", 42),
    ("Complex", 43),
    ("Complex", 44),

    ("Mag+DeltaPhase", 42),
    ("Mag+DeltaPhase", 43),
    ("Mag+DeltaPhase", 44),
]


column_labels = [
    "Mag\n42",
    "Mag\n43",
    "Mag\n44",

    "Complex\n42",
    "Complex\n43",
    "Complex\n44",

    "Mag+ΔP\n42",
    "Mag+ΔP\n43",
    "Mag+ΔP\n44",
]


# ============================================================
# Matrix
# ============================================================

M = np.zeros(
    (
        len(conditions),
        len(columns),
    ),
    dtype=float,
)

for i, c in enumerate(conditions):

    for j, key in enumerate(columns):

        s = stats[key][c]

        M[i, j] = (
            100.0
            * s["correct"]
            / s["total"]
        )


# ============================================================
# Save numeric CSV
# ============================================================

csv_path = OUT_DIR / "condition_heatmap_values.csv"

with csv_path.open("w", newline="") as f:

    writer = csv.writer(f)

    writer.writerow(
        ["condition"]
        + [
            f"{rep}_seed{seed}"
            for rep, seed in columns
        ]
    )

    for label, row in zip(
        condition_labels,
        M,
    ):
        writer.writerow(
            [label]
            + [
                f"{x:.6f}"
                for x in row
            ]
        )


# ============================================================
# Heatmap
# ============================================================

fig, ax = plt.subplots(
    figsize=(12.5, 6.5)
)

im = ax.imshow(
    M,
    aspect="auto",
    vmin=0,
    vmax=100,
)

ax.set_xticks(
    np.arange(len(column_labels))
)
ax.set_xticklabels(
    column_labels,
    fontsize=10,
)

ax.set_yticks(
    np.arange(len(condition_labels))
)
ax.set_yticklabels(
    condition_labels,
    fontsize=10,
)

ax.set_xlabel(
    "Representation and random seed",
    fontsize=11,
)

ax.set_ylabel(
    "Unseen test condition",
    fontsize=11,
)

ax.set_title(
    "Condition-wise Accuracy under Condition-Disjoint Evaluation",
    fontsize=13,
)


# ------------------------------------------------------------
# Cell values
# ------------------------------------------------------------

for i in range(M.shape[0]):

    for j in range(M.shape[1]):

        value = M[i, j]

        text_color = (
            "white"
            if value < 45 or value > 80
            else "black"
        )

        ax.text(
            j,
            i,
            f"{value:.1f}",
            ha="center",
            va="center",
            fontsize=9,
            color=text_color,
        )


# ------------------------------------------------------------
# Visual separators between representations
# ------------------------------------------------------------

ax.axvline(
    2.5,
    linewidth=1.2,
)

ax.axvline(
    5.5,
    linewidth=1.2,
)


# ------------------------------------------------------------
# Colorbar
# ------------------------------------------------------------

cbar = fig.colorbar(
    im,
    ax=ax,
)

cbar.set_label(
    "Accuracy (%)"
)


fig.tight_layout()

png_path = (
    OUT_DIR
    / "condition_wise_heatmap.png"
)

pdf_path = (
    OUT_DIR
    / "condition_wise_heatmap.pdf"
)

fig.savefig(
    png_path,
    dpi=300,
    bbox_inches="tight",
)

fig.savefig(
    pdf_path,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# Condition mean/std by representation
# ============================================================

summary_csv = (
    OUT_DIR
    / "condition_mean_std_by_representation.csv"
)

with summary_csv.open(
    "w",
    newline="",
) as f:

    writer = csv.writer(f)

    writer.writerow(
        [
            "condition",
            "representation",
            "mean_accuracy",
            "std_accuracy",
        ]
    )

    for i, label in enumerate(
        condition_labels
    ):

        for rep, start_col in [
            ("Magnitude", 0),
            ("Complex", 3),
            ("Mag+DeltaPhase", 6),
        ]:

            vals = M[
                i,
                start_col:start_col + 3
            ]

            writer.writerow(
                [
                    label,
                    rep,
                    f"{np.mean(vals):.6f}",
                    f"{np.std(vals, ddof=1):.6f}",
                ]
            )


print()
print("Saved:")
print(" ", png_path)
print(" ", pdf_path)
print(" ", csv_path)
print(" ", summary_csv)

print()
print("Matrix:")
print(M)
