import matplotlib.pyplot as plt
import numpy as np

names = [
    "Magnitude",
    "Complex",
    "Mag + DeltaPhase",
]

val_f1 = [
    99.667,
    99.667,
    99.667,
]

test_f1 = [
    61.356,
    51.500,
    52.558,
]

test_std = [
    11.236,
    9.294,
    3.278,
]

x = np.arange(len(names))
w = 0.34

fig, ax = plt.subplots(figsize=(8.5, 5.5))

ax.bar(
    x - w / 2,
    val_f1,
    w,
    label="Validation Macro-F1",
)

ax.bar(
    x + w / 2,
    test_f1,
    w,
    yerr=test_std,
    capsize=5,
    label="Condition-disjoint test Macro-F1",
)

ax.set_ylabel("Macro-F1 (%)")
ax.set_xlabel("Input representation")
ax.set_title(
    "Validation Performance vs Unseen-Condition Generalization"
)

ax.set_xticks(x)
ax.set_xticklabels(names)

ax.set_ylim(0, 110)
ax.legend()

for i, v in enumerate(val_f1):
    ax.text(
        i - w / 2,
        v + 1.5,
        f"{v:.1f}",
        ha="center",
        fontsize=9,
    )

for i, v in enumerate(test_f1):
    ax.text(
        i + w / 2,
        v + test_std[i] + 1.5,
        f"{v:.1f}",
        ha="center",
        fontsize=9,
    )

fig.tight_layout()

fig.savefig(
    "figures/validation_test_gap.png",
    dpi=300,
    bbox_inches="tight",
)

fig.savefig(
    "figures/validation_test_gap.pdf",
    bbox_inches="tight",
)

print("saved validation_test_gap")
