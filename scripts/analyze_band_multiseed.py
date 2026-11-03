#!/usr/bin/env python3

import csv
import json
import statistics
from pathlib import Path
from collections import defaultdict

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

OUT = Path(
    "exp/paper_figures/band_multiseed_summary.csv"
)

meta = {}

with MANIFEST.open() as f:
    for line in f:
        x = json.loads(line)
        meta[x["key"]] = x


results = []

for rep, seed, pred_path in RUNS:

    stats = defaultdict(
        lambda: {
            "correct": 0,
            "total": 0,
        }
    )

    with pred_path.open() as f:
        reader = csv.DictReader(f)

        for r in reader:

            x = meta[r["key"]]
            band = int(x["band"])

            stats[band]["total"] += 1

            if r["true_label"] == r["pred_label"]:
                stats[band]["correct"] += 1

    for band in sorted(stats):

        correct = stats[band]["correct"]
        total = stats[band]["total"]

        acc = correct / total

        results.append({
            "representation": rep,
            "seed": seed,
            "band": band,
            "correct": correct,
            "total": total,
            "accuracy": acc,
        })


OUT.parent.mkdir(
    parents=True,
    exist_ok=True
)

with OUT.open(
    "w",
    newline=""
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "representation",
            "seed",
            "band",
            "correct",
            "total",
            "accuracy",
        ],
    )

    writer.writeheader()
    writer.writerows(results)


print()
print("=" * 80)
print("BAND ACCURACY BY SEED")
print("=" * 80)

print(
    f"{'Representation':18s} "
    f"{'Seed':>5s} "
    f"{'Band1':>10s} "
    f"{'Band5':>10s}"
)

print("-" * 55)

for rep in [
    "Magnitude",
    "Complex",
    "Mag+DeltaPhase",
]:

    for seed in [42, 43, 44]:

        xs = {
            r["band"]: r["accuracy"]
            for r in results
            if (
                r["representation"] == rep
                and r["seed"] == seed
            )
        }

        print(
            f"{rep:18s} "
            f"{seed:5d} "
            f"{100*xs[1]:9.3f}% "
            f"{100*xs[5]:9.3f}%"
        )


print()
print("=" * 80)
print("MEAN +/- STD ACROSS SEEDS")
print("=" * 80)

for rep in [
    "Magnitude",
    "Complex",
    "Mag+DeltaPhase",
]:

    print()
    print(rep)

    for band in [1, 5]:

        vals = [
            100 * r["accuracy"]
            for r in results
            if (
                r["representation"] == rep
                and r["band"] == band
            )
        ]

        print(
            f"  band {band}: "
            f"{statistics.mean(vals):.3f} "
            f"+/- "
            f"{statistics.stdev(vals):.3f}%"
        )


print()
print("Saved:")
print(OUT)
