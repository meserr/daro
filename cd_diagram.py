"""Critical difference diagram for the Nemenyi test."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def draw_cd(names, ranks, cd, out, highlight="D-ARO", n_left=None):
    sr, sn = zip(*sorted(zip(ranks, names)))
    k = len(sr)
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial"], "font.size": 12})
    fig, ax = plt.subplots(figsize=(10, 4.8), dpi=300)
    ax.set_xlim(0.1, k + 0.9)
    ax.set_ylim(-0.55, 1.15)
    ax.axis("off")

    ax.plot([1, k], [0, 0], color="black", linewidth=1.5)
    for r in range(1, k + 1):
        ax.plot([r, r], [-0.03, 0.03], color="black", linewidth=1.5)
        ax.text(r, -0.06, str(r), ha="center", va="top", fontsize=12, fontweight="bold")

    # groups of algorithms whose rank difference is below cd
    groups = []
    for i in range(k):
        j = max(j for j in range(i, k) if sr[j] - sr[i] <= cd)
        if j > i and not any(a <= i and j <= b for a, b in groups):
            groups.append((i, j))
    for n, (i, j) in enumerate(groups):
        y = -0.20 - 0.06 * n
        ax.plot([sr[i], sr[j]], [y, y], color="black", linewidth=4.0, solid_capstyle="butt")
    ax.text((1 + k) / 2, -0.24 - 0.06 * max(len(groups), 1), "Average Rank",
            ha="center", va="top", fontsize=13, fontweight="bold")

    mid = (1 + k) / 2
    left = [i for i in range(k) if sr[i] < mid] if n_left is None else list(range(n_left))
    right = [i for i in range(k) if i not in left]
    for side, idx in (("left", left), ("right", list(reversed(right)))):
        for n, i in enumerate(idx):
            y = 0.15 + n * 0.16
            x_text, x_line = (0.42, 0.5) if side == "left" else (k + 0.58, k + 0.5)
            hl = sn[i] == highlight
            ax.text(x_text, y, f"{sn[i]} ({sr[i]:.2f})", ha="right" if side == "left" else "left",
                    va="center", fontsize=11, fontweight="bold" if hl else "normal",
                    color="#e74c3c" if hl else "black")
            ax.plot([x_line, sr[i]], [y, y], color="gray", linewidth=1.0, alpha=0.8)
            ax.plot([sr[i], sr[i]], [y, 0], color="gray", linewidth=1.0, alpha=0.8)
            ax.plot(sr[i], 0, marker="o", color="black", markersize=4)

    x0, yb = mid - cd / 2, 1.0
    ax.plot([x0, x0 + cd], [yb, yb], color="black", linewidth=1.5)
    for x in (x0, x0 + cd):
        ax.plot([x, x], [yb - 0.03, yb + 0.03], color="black", linewidth=1.5)
    ax.text(mid, yb + 0.05, f"CD = {cd:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    plt.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
