#!/usr/bin/env python3

import csv
import json
from pathlib import Path
from collections import defaultdict, Counter

MANIFEST = Path(
    "data/gnss_jammer_3class_cond/test.list"
)

RUNS = {
    ("Magnitude", 42):
        Path("exp/rf_resnet34_3class_cond/phase_0/predictions.csv"),

    ("Magnitude", 43):
        Path("exp/rf_resnet34_3class_cond_seed43/eval_test_selected/predictions.csv"),

    ("Magnitude", 44):
        Path("exp/rf_resnet34_3class_cond_seed44/eval_test_selected/predictions.csv"),

    ("Complex", 42):
        Path("exp/rf_resnet34_complex_3class_cond/phase_0/predictions.csv"),

    ("Complex", 43):
        Path("exp/rf_resnet34_complex_3class_cond_seed43/eval_test_selected/predictions.csv"),

    ("Complex", 44):
        Path("exp/rf_resnet34_complex_3class_cond_seed44/eval_test_selected/predictions.csv"),

    ("Mag+DP", 42):
        Path("exp/rf_resnet34_mag_phasediff_3class_cond/phase_0/predictions.csv"),

    ("Mag+DP", 43):
        Path("exp/rf_resnet34_mag_phasediff_3class_cond_seed43/eval_test_selected/predictions.csv"),

    ("Mag+DP", 44):
        Path("exp/rf_resnet34_mag_phasediff_3class_cond_seed44/eval_test_selected/predictions.csv"),
}

meta = {}

with MANIFEST.open() as f:
    for line in f:
        x = json.loads(line)

        meta[x["key"]] = x


def cond(x):
    return (
        x["spk"],
        x["power"],
        x["bandwidth"],
        x["vga"],
        x["band"],
    )


all_stats = {}

for run, pred_path in RUNS.items():

    if not pred_path.exists():
        print("MISSING:", pred_path)
        continue

    stats = defaultdict(
        lambda: {
            "n": 0,
            "correct": 0,
            "pred": Counter(),
        }
    )

    with pred_path.open() as f:
        reader = csv.DictReader(f)

        for r in reader:

            x = meta[r["key"]]
            c = cond(x)

            stats[c]["n"] += 1

            if r["true_label"] == r["pred_label"]:
                stats[c]["correct"] += 1

            stats[c]["pred"][r["pred_label"]] += 1

    all_stats[run] = stats


conditions = sorted(
    {
        c
        for stats in all_stats.values()
        for c in stats
    },
    key=str,
)


print()
print("=" * 150)
print("CONDITION-WISE ACCURACY BY SEED")
print("=" * 150)

header = (
    f"{'CLASS':8s} "
    f"{'PWR':>5s} "
    f"{'BW':>5s} "
    f"{'VGA':>5s} "
    f"{'BAND':>4s} "
)

for rep in ["Magnitude", "Complex", "Mag+DP"]:
    for seed in [42, 43, 44]:
        header += f"{rep[:4]}{seed:02d} ".replace(" ", "")

print(header)

for c in conditions:

    cls, power, bw, vga, band = c

    row = (
        f"{cls:8s} "
        f"{str(power):>5s} "
        f"{str(bw):>5s} "
        f"{str(vga):>5s} "
        f"{str(band):>4s} "
    )

    for rep in ["Magnitude", "Complex", "Mag+DP"]:
        for seed in [42, 43, 44]:

            s = all_stats.get((rep, seed))

            if s is None or c not in s:
                row += f"{'NA':>7s} "
                continue

            acc = (
                100.0
                * s[c]["correct"]
                / s[c]["n"]
            )

            row += f"{acc:6.2f}% "

    print(row)


print()
print("=" * 120)
print("CLASS-WISE ACCURACY BY SEED")
print("=" * 120)

for rep in ["Magnitude", "Complex", "Mag+DP"]:

    print()
    print(rep)

    for seed in [42, 43, 44]:

        stats = all_stats[(rep, seed)]

        by_cls = defaultdict(
            lambda: [0, 0]
        )

        for c, s in stats.items():

            cls = c[0]

            by_cls[cls][0] += s["correct"]
            by_cls[cls][1] += s["n"]

        print(
            f"  seed {seed}: ",
            end=""
        )

        for cls in [
            "Meac",
            "Prn",
            "Spoof",
        ]:

            correct, total = by_cls[cls]

            acc = (
                100.0
                * correct
                / total
            )

            print(
                f"{cls}={acc:6.2f}% ",
                end=""
            )

        print()


print()
print("=" * 120)
print("PREDICTION DISTRIBUTION FOR MAGNITUDE")
print("=" * 120)

for c in conditions:

    print()
    print(c)

    for seed in [42, 43, 44]:

        s = all_stats[
            ("Magnitude", seed)
        ][c]

        print(
            f"  seed {seed}: "
            f"acc={100*s['correct']/s['n']:6.2f}% "
            f"pred={dict(s['pred'])}"
        )
