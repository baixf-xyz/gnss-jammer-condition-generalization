#!/usr/bin/env python3

import csv
import statistics
from pathlib import Path
from collections import defaultdict


MULTI = Path(
    "exp/multiseed_3class_results.csv"
)

BAND = Path(
    "exp/paper_figures/band_multiseed_summary.csv"
)


# ============================================================
# Overall multi-seed
# ============================================================

rows = []

with MULTI.open() as f:
    for r in csv.DictReader(f):

        r["seed"] = int(r["seed"])

        for k in [
            "val_accuracy",
            "val_macro_f1",
            "test_accuracy",
            "test_macro_f1",
        ]:
            r[k] = float(r[k])

        rows.append(r)


groups = defaultdict(list)

for r in rows:
    groups[r["representation"]].append(r)


name_map = {
    "magnitude": "Magnitude",
    "complex": "Complex",
    "magdp": "Mag + DeltaPhase",
}


print()
print("=" * 90)
print("TABLE 1 - OVERALL CONDITION-DISJOINT TEST")
print("=" * 90)

print(
    f"{'Representation':20s} "
    f"{'Accuracy':>24s} "
    f"{'Macro-F1':>24s}"
)

print("-" * 72)

for key in [
    "magnitude",
    "complex",
    "magdp",
]:

    xs = groups[key]

    acc = [
        100 * x["test_accuracy"]
        for x in xs
    ]

    f1 = [
        100 * x["test_macro_f1"]
        for x in xs
    ]

    print(
        f"{name_map[key]:20s} "
        f"{statistics.mean(acc):7.3f} +/- "
        f"{statistics.stdev(acc):6.3f}% "
        f"{statistics.mean(f1):7.3f} +/- "
        f"{statistics.stdev(f1):6.3f}%"
    )


# ============================================================
# Validation
# ============================================================

print()
print("=" * 90)
print("VALIDATION MACRO-F1 BY SEED")
print("=" * 90)

for key in [
    "magnitude",
    "complex",
    "magdp",
]:

    xs = sorted(
        groups[key],
        key=lambda x: x["seed"]
    )

    vals = [
        100 * x["val_macro_f1"]
        for x in xs
    ]

    print(
        f"{name_map[key]:20s}: "
        + "  ".join(
            f"{x:.3f}%"
            for x in vals
        )
    )


# ============================================================
# Band results
# ============================================================

band_rows = []

with BAND.open() as f:
    for r in csv.DictReader(f):

        r["seed"] = int(r["seed"])
        r["band"] = int(r["band"])
        r["accuracy"] = float(
            r["accuracy"]
        )

        band_rows.append(r)


print()
print("=" * 90)
print("TABLE 2 - BAND-WISE TEST ACCURACY")
print("=" * 90)

print(
    f"{'Representation':20s} "
    f"{'Band 1':>24s} "
    f"{'Band 5':>24s} "
    f"{'Gap':>12s}"
)

print("-" * 86)

for rep in [
    "Magnitude",
    "Complex",
    "Mag+DeltaPhase",
]:

    b1 = [
        100 * r["accuracy"]
        for r in band_rows
        if (
            r["representation"] == rep
            and r["band"] == 1
        )
    ]

    b5 = [
        100 * r["accuracy"]
        for r in band_rows
        if (
            r["representation"] == rep
            and r["band"] == 5
        )
    ]

    m1 = statistics.mean(b1)
    s1 = statistics.stdev(b1)

    m5 = statistics.mean(b5)
    s5 = statistics.stdev(b5)

    gap = m1 - m5

    display = (
        "Mag + DeltaPhase"
        if rep == "Mag+DeltaPhase"
        else rep
    )

    print(
        f"{display:20s} "
        f"{m1:7.3f} +/- {s1:6.3f}% "
        f"{m5:7.3f} +/- {s5:6.3f}% "
        f"{gap:9.3f} pp"
    )


# ============================================================
# Per-seed test
# ============================================================

print()
print("=" * 90)
print("PER-SEED TEST MACRO-F1")
print("=" * 90)

for key in [
    "magnitude",
    "complex",
    "magdp",
]:

    xs = sorted(
        groups[key],
        key=lambda x: x["seed"]
    )

    print()
    print(name_map[key])

    for x in xs:
        print(
            f"  seed {x['seed']}: "
            f"Acc={100*x['test_accuracy']:.3f}%  "
            f"F1={100*x['test_macro_f1']:.3f}%  "
            f"ckpt={x['selected_checkpoint']}"
        )


print()
print("=" * 90)
print("PAPER FIGURES")
print("=" * 90)

for p in [
    "exp/paper_figures/condition_wise_heatmap.pdf",
    "exp/paper_figures/band_wise_mean_std.pdf",
]:

    print(p)

print()
print("DONE")
