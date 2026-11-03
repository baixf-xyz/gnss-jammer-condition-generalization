#!/usr/bin/env python3

import json
import random
from pathlib import Path
from collections import Counter, defaultdict

import h5py


# ============================================================
# Configuration
# ============================================================

H5_ROOT = Path(
    "/root/autodl-tmp/FraunhoferIIS_Jammertest2025/dataset"
)

OUT_DIR = Path("data/gnss_jammer_3class_cond")

SEED = 42

# 每类最多取多少条
# 先保持和旧实验接近：
# train = 2800/class
# val   = 600/class
# test  = 600/class
N_TRAIN_PER_CLASS = 2800
N_VAL_PER_CLASS = 600
N_TEST_PER_CLASS = 600


# ============================================================
# Jammertest label mapping
#
# 你前面的数据已经确认：
# Prn   = jid 1
# Meac  = jid 3
# Spoof = jid 5
#
# 如果你自己的 label id 不同，只需要改这里。
# ============================================================

JID_TO_CLASS = {
    2: "Prn",
    3: "Meac",
    4: "Spoof",
}

CLASSES = ["Prn", "Meac", "Spoof"]


# ============================================================
# Condition split
#
# condition 定义：
# (power, bandwidth, vga, band)
#
# train / val / test 不允许出现完全相同 condition。
# ============================================================

def split_of(cls, power, bw, vga, band):

    # --------------------------------------------------------
    # Prn
    # --------------------------------------------------------
    if cls == "Prn":

        # Train: VGA = 140
        if vga == 140:
            return "train"

        # Validation: VGA = 170
        if vga == 170:
            return "val"

        # Test: VGA = 100
        if vga == 100:
            return "test"

        return None

    # --------------------------------------------------------
    # Meac
    # --------------------------------------------------------
    if cls == "Meac":

        # Train:
        # VGA = 80
        # power = 25 / 30 / 35
        if (
            vga == 80
            and power in {25, 30, 35}
        ):
            return "train"

        # Validation:
        # VGA = 80
        # power = 40
        #
        # 与训练集 condition tuple 不重合
        if (
            vga == 80
            and power == 40
        ):
            return "val"

        # Test:
        # unseen receiver gain
        if vga == 100:
            return "test"

        return None

    # --------------------------------------------------------
    # Spoof
    # --------------------------------------------------------
    if cls == "Spoof":

        # Train:
        # VGA=80, power=5/10/15
        if (
            vga == 80
            and power in {5, 10, 15}
        ):
            return "train"

        # Validation:
        # VGA=80, power=20
        if (
            vga == 80
            and power == 20
        ):
            return "val"

        # Test:
        # unseen high-power + receiver gain
        if (
            power == 25
            and vga in {120, 140}
        ):
            return "test"

        return None

    return None


# ============================================================
# Helpers
# ============================================================

def norm_num(x):
    """Convert numpy scalar -> plain Python number."""
    x = float(x)

    if x.is_integer():
        return int(x)

    return x


def condition_of(x):
    return (
        x["spk"],
        x["power"],
        x["bandwidth"],
        x["vga"],
        x["band"],
    )


def sample_balanced(records, n_per_class, seed):

    rng = random.Random(seed)

    by_class = defaultdict(list)

    for x in records:
        by_class[x["spk"]].append(x)

    selected = []

    print()
    print("Available before sampling:")

    for cls in CLASSES:

        xs = by_class[cls]

        print(
            f"  {cls:8s}: {len(xs):7d}"
        )

        if len(xs) < n_per_class:
            raise RuntimeError(
                f"{cls}: need {n_per_class}, "
                f"but only {len(xs)} available"
            )

        rng.shuffle(xs)

        selected.extend(
            xs[:n_per_class]
        )

    rng.shuffle(selected)

    return selected


def write_jsonl(path, xs):

    with path.open("w") as f:

        for x in xs:
            f.write(
                json.dumps(
                    x,
                    ensure_ascii=False
                )
                + "\n"
            )


# ============================================================
# Scan H5
# ============================================================

print("=" * 80)
print("Scanning H5 files")
print("=" * 80)

h5_files = sorted(
    H5_ROOT.glob("Sample_Area1_*.h5")
)

if not h5_files:
    raise RuntimeError(
        f"No H5 found under {H5_ROOT}"
    )

print("H5 files:", len(h5_files))

for p in h5_files:
    print(" ", p.name)


all_records = {
    "train": [],
    "val": [],
    "test": [],
}

condition_stats = Counter()


for h5_path in h5_files:

    print()
    print("Reading:", h5_path.name)

    with h5py.File(h5_path, "r") as f:

        labels = f["label"]

        n = len(labels)

        for row in range(n):

            jid, power, bw, vga, band = labels[row]

            jid = int(jid)

            if jid not in JID_TO_CLASS:
                continue

            cls = JID_TO_CLASS[jid]

            power = norm_num(power)
            bw = norm_num(bw)
            vga = norm_num(vga)
            band = norm_num(band)

            split = split_of(
                cls,
                power,
                bw,
                vga,
                band,
            )

            if split is None:
                continue

            key = (
                f"{h5_path.stem}_"
                f"{row:07d}"
            )

            item = {
                "key": key,
                "h5": str(h5_path.resolve()),
                "row": int(row),
                "spk": cls,
                "jammer_id": jid,
                "power": power,
                "bandwidth": bw,
                "vga": vga,
                "band": band,
            }

            all_records[split].append(item)

            condition_stats[
                (
                    split,
                    cls,
                    power,
                    bw,
                    vga,
                    band,
                )
            ] += 1


# ============================================================
# Print condition inventory
# ============================================================

print()
print("=" * 80)
print("Selected condition inventory")
print("=" * 80)

for k, n in sorted(
    condition_stats.items(),
    key=lambda x: str(x[0])
):

    split, cls, power, bw, vga, band = k

    print(
        f"{split:5s} "
        f"{cls:8s} "
        f"power={str(power):>5s} "
        f"bw={str(bw):>5s} "
        f"vga={str(vga):>5s} "
        f"band={str(band):>3s} "
        f"n={n:7d}"
    )


# ============================================================
# Verify no identical condition tuple leakage
# ============================================================

condition_sets = {}

for split in ["train", "val", "test"]:

    condition_sets[split] = {
        condition_of(x)
        for x in all_records[split]
    }


print()
print("=" * 80)
print("Checking condition leakage")
print("=" * 80)

pairs = [
    ("train", "val"),
    ("train", "test"),
    ("val", "test"),
]

has_leak = False

for a, b in pairs:

    overlap = (
        condition_sets[a]
        & condition_sets[b]
    )

    print(
        f"{a:5s} vs {b:5s}: "
        f"{len(overlap)} overlapping conditions"
    )

    if overlap:

        has_leak = True

        for x in sorted(
            overlap,
            key=str
        ):
            print("  LEAK:", x)


if has_leak:
    raise RuntimeError(
        "Condition leakage detected."
    )


# ============================================================
# Balanced sampling
# ============================================================

print()
print("=" * 80)
print("Balanced sampling")
print("=" * 80)

train = sample_balanced(
    all_records["train"],
    N_TRAIN_PER_CLASS,
    SEED,
)

val = sample_balanced(
    all_records["val"],
    N_VAL_PER_CLASS,
    SEED + 1,
)

test = sample_balanced(
    all_records["test"],
    N_TEST_PER_CLASS,
    SEED + 2,
)


# ============================================================
# Verify key leakage
# ============================================================

train_keys = {x["key"] for x in train}
val_keys = {x["key"] for x in val}
test_keys = {x["key"] for x in test}

assert not train_keys & val_keys
assert not train_keys & test_keys
assert not val_keys & test_keys


# ============================================================
# Write manifests
# ============================================================

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

write_jsonl(
    OUT_DIR / "train.list",
    train,
)

write_jsonl(
    OUT_DIR / "val.list",
    val,
)

write_jsonl(
    OUT_DIR / "test.list",
    test,
)


# WeSpeaker class label file
with (OUT_DIR / "train_label").open("w") as f:

    for idx, cls in enumerate(CLASSES):

        f.write(
            f"{cls} {idx}\n"
        )


# ============================================================
# Summary
# ============================================================

print()
print("=" * 80)
print("FINAL SPLIT")
print("=" * 80)

for name, xs in [
    ("train", train),
    ("val", val),
    ("test", test),
]:

    cnt = Counter(
        x["spk"]
        for x in xs
    )

    print()
    print(
        f"{name}: {len(xs)}"
    )

    for cls in CLASSES:
        print(
            f"  {cls:8s}: "
            f"{cnt[cls]}"
        )


print()
print("Conditions actually used:")

for name, xs in [
    ("train", train),
    ("val", val),
    ("test", test),
]:

    print()
    print(
        "-" * 70
    )
    print(name.upper())

    cnt = Counter(
        condition_of(x)
        for x in xs
    )

    for cond, n in sorted(
        cnt.items(),
        key=lambda x: str(x[0])
    ):

        cls, power, bw, vga, band = cond

        print(
            f"{cls:8s} "
            f"power={str(power):>5s} "
            f"bw={str(bw):>5s} "
            f"vga={str(vga):>5s} "
            f"band={str(band):>3s} "
            f"n={n:5d}"
        )


print()
print("=" * 80)
print("Saved")
print("=" * 80)

print(
    OUT_DIR / "train.list"
)
print(
    OUT_DIR / "val.list"
)
print(
    OUT_DIR / "test.list"
)
print(
    OUT_DIR / "train_label"
)

print()
print("DONE")
