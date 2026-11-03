import matplotlib.pyplot as plt
import numpy as np

seeds = ["42", "43", "44"]

magnitude = [
    70.577,
    64.650,
    48.842,
]

complex_ = [
    40.830,
    57.836,
    55.834,
]

magdp = [
    54.990,
    53.854,
    48.831,
]

x = np.arange(len(seeds))
w = 0.24

fig, ax = plt.subplots(figsize=(8.5, 5.5))

ax.bar(
    x - w,
    magnitude,
    w,
    label="Magnitude",
)

ax.bar(
    x,
    complex_,
    w,
    label="Complex",
)

ax.bar(
    x + w,
    magdp,
    w,
    label="Mag + DeltaPhase",
)

ax.set_xlabel("Random seed")
ax.set_ylabel("Test Macro-F1 (%)")
ax.set_title(
    "Seed-wise Test Macro-F1 under Condition-Disjoint Evaluation"
)

ax.set_xticks(x)
ax.set_xticklabels(seeds)

ax.set_ylim(0, 80)
ax.legend()

fig.tight_layout()

fig.savefig(
    "figures/seedwise_macro_f1.png",
    dpi=300,
    bbox_inches="tight",
)

fig.savefig(
    "figures/seedwise_macro_f1.pdf",
    bbox_inches="tight",
)

print("saved seedwise_macro_f1")
