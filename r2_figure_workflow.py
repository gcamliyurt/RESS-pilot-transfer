"""RESS R2: workflow diagram (manuscript Figure 4), redrawn for the revised method.

Four bands, top to bottom: event reconstruction and validation, decision-time
context, prediction, safety-constrained relocation. Output:
RESS_JRESS-D-26-03347/figures_R2/figure_03_workflow.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).parent / "RESS_JRESS-D-26-03347" / "figures_R2" / "figure_03_workflow.png"

# Okabe-Ito based fills, light enough for black text and distinct in grayscale.
FILL = {"recon": "#D6E6F5", "context": "#CDEBE3", "predict": "#F7E6C4", "safety": "#F9D9C4", "out": "#E6E6E6"}
EDGE = {"recon": "#0072B2", "context": "#009E73", "predict": "#E69F00", "safety": "#D55E00", "out": "#555555"}
BAND = "#F2F2F2"

W, H = 17.0, 21.0  # canvas units (cm-like)


def box(ax, x, y, w, h, text, kind, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.25",
                                fc=FILL[kind], ec=EDGE[kind], lw=1.4))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.5,
            fontweight="bold" if bold else "normal", wrap=True, linespacing=1.25)
    return (x, y, w, h)


def arrow(ax, p, q, color="#333333", style="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=12, lw=1.2, color=color,
                                 linestyle=ls, shrinkA=2, shrinkB=2))


def band(ax, y, h, label):
    ax.add_patch(FancyBboxPatch((0.2, y), 13.6, h, boxstyle="square,pad=0", fc=BAND, ec="#9A9A9A", lw=1.0))
    ax.text(14.15, y + h / 2, label, ha="left", va="center", fontsize=9.6, fontweight="bold", linespacing=1.3)


def bottom(b):
    return (b[0] + b[2] / 2, b[1])


def top(b):
    return (b[0] + b[2] / 2, b[1] + b[3])


def right(b):
    return (b[0] + b[2], b[1] + b[3] / 2)


def left(b):
    return (b[0], b[1] + b[3] / 2)


def main() -> None:
    fig, ax = plt.subplots(figsize=(17 / 2.54, 21 / 2.54))
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")

    # 1. Event reconstruction and validation
    band(ax, 16.0, 4.8, "1. Event\nreconstruction\nand validation")
    a1 = box(ax, 0.6, 18.4, 3.9, 1.9, "AIS dynamic and\nstatic data\n(13 verified\npilot boats)", "recon")
    a2 = box(ax, 5.1, 18.4, 3.9, 1.9, "Session detection\n(hull separation,\ncourse, speed,\nboarding signature)", "recon")
    a3 = box(ax, 9.6, 18.4, 3.9, 1.9, "Pilotage-log\nconfirmation\n(same ship, ±2 h)", "recon")
    a4 = box(ax, 5.1, 16.3, 8.4, 1.5, "Confirmed transfers (T, position, direction)\ntarget = distance to breakwater lights", "recon")
    arrow(ax, right(a1), left(a2))
    arrow(ax, right(a2), left(a3))
    arrow(ax, bottom(a3), (a3[0] + a3[2] / 2, a4[1] + a4[3]))

    # 2. Decision-time context
    band(ax, 10.9, 4.6, "2. Decision-time\ncontext")
    b0 = box(ax, 0.6, 13.3, 12.9, 1.6, "Decision at pilot-boat dispatch t_d\n(only AIS and weather observed at or before t_d)", "context", bold=True)
    arrow(ax, bottom(a4), (a4[0] + a4[2] / 2, b0[1] + b0[3]))
    b1 = box(ax, 0.6, 11.3, 3.0, 1.6, "Ship and\npilot-boat\nstates", "context")
    b2 = box(ax, 3.9, 11.3, 3.0, 1.6, "Kinematic\nintercept,\ntraffic", "context")
    b3 = box(ax, 7.2, 11.3, 3.0, 1.6, "Wind and wave\n(hourly)", "context")
    b4 = box(ax, 10.5, 11.3, 3.0, 1.6, "Draught, type,\nhour, month", "context")
    for b in (b1, b2, b3, b4):
        arrow(ax, (b[0] + b[2] / 2, b0[1]), top(b))

    # 3. Prediction
    band(ax, 6.6, 3.9, "3. Prediction")
    c1 = box(ax, 0.6, 8.3, 5.6, 1.8, "RF–Ridge hybrid\n(inbound, outbound)\ngrouped + temporal\nvalidation", "predict")
    c2 = box(ax, 7.0, 8.3, 6.5, 1.8, "Out-of-fold predicted\nmeeting distance  D̂", "predict")
    c3 = box(ax, 7.0, 6.75, 6.5, 1.3, "ML anchor a\nD̂ along a fold-learnt bearing", "predict", bold=True)
    for b in (b1, b2, b3, b4):
        arrow(ax, bottom(b), (b[0] + b[2] / 2 if b[0] < 6 else min(b[0] + b[2] / 2, 6.0), c1[1] + c1[3]) if b[0] < 6
              else (c1[0] + c1[2] - 0.4, c1[1] + c1[3]))
    arrow(ax, right(c1), left(c2))
    arrow(ax, bottom(c2), (c2[0] + c2[2] / 2, c3[1] + c3[3]))

    # 4. Safety-constrained relocation
    band(ax, 0.3, 5.9, "4. Safety-\nconstrained\nrelocation")
    d1 = box(ax, 0.6, 3.75, 4.0, 2.1, "Candidates\nanchor, 150–500 m\nring, centreline,\ngrid; legs routed\nthrough charted water", "safety")
    d2 = box(ax, 5.05, 3.75, 4.0, 2.1, "Safety gates\n(calibrated on\ntraining folds)\nUKC ≥ 2 m, traffic,\nbuoys, fairway", "safety")
    d3 = box(ax, 9.5, 3.75, 4.0, 2.1, "Objective\nmax Q(s) − λ·d(s, a)\nλ = 0.1 per km", "safety")
    d4 = box(ax, 5.05, 1.9, 4.0, 1.25, "Recommended point s*\nor \"no feasible point\"", "safety", bold=True)
    d5 = box(ax, 9.5, 0.5, 4.0, 1.5, "Audit outputs\ngates, risk class,\nrelocation", "out")
    arrow(ax, left(c3), (d1[0] + d1[2] / 2, d1[1] + d1[3]))
    arrow(ax, right(d1), left(d2))
    arrow(ax, right(d2), left(d3))
    arrow(ax, (d3[0] + d3[2] / 2, d3[1]), right(d4))
    arrow(ax, bottom(d4), left(d5))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
