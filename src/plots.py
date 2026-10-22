"""Figures saved to outputs/figures (used in the README)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.lines
import matplotlib.patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_MUTED = "#52514e"
GRID = "#e4e3df"
BLUE = "#2a78d6"   # also the "above the line" pole of the diverging pair
RED = "#e34948"    # "below the line" pole

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": TEXT_MUTED, "axes.titlecolor": TEXT,
    "xtick.color": TEXT_MUTED, "ytick.color": TEXT_MUTED, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.axisbelow": True,
})


def save_model_comparison(scores: dict):
    """Dot plot (not bars: the differences are a few points out of ~1070, and bars would need a cut axis)."""
    names, values = list(scores), list(scores.values())
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.scatter(values, names, s=70, color=BLUE, zorder=3)
    for v, n in zip(values, names):
        ax.annotate(f"{v:.1f}", (v, n), xytext=(8, 0), textcoords="offset points", va="center", color=TEXT)
    ax.set_xlim(min(values) - 5, max(values) + 8)
    ax.set_xlabel("Cross-validated RMSE (lower is better)")
    ax.grid(axis="y", visible=False)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(config.FIGURE_DIR / "model_comparison.png", dpi=150)
    plt.close(fig)


def save_actual_vs_predicted(y, oof):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y, oof, s=4, alpha=0.3, color=BLUE, linewidths=0)
    top = max(np.max(y), np.max(oof))
    ax.plot([0, top], [0, top], color=TEXT, lw=1.5, label="perfect prediction")
    ax.set_xlabel("Actual sales")
    ax.set_ylabel("Out-of-fold predicted sales")
    ax.legend(frameon=False, labelcolor=TEXT_MUTED)
    fig.tight_layout()
    fig.savefig(config.FIGURE_DIR / "actual_vs_predicted.png", dpi=150)
    plt.close(fig)


def save_id_number_effect(model, X, y):
    """Bars: raw average of sales / per-outlet-line - 1 for each id number. Line: the smoothed
    effect the v3 model uses. `model` is a fitted PerOutletLinearTimesIDNumber."""
    ratio = pd.Series(np.asarray(y) / model.line_.predict(X) - 1)
    raw = ratio.groupby(X["ID_Number"].to_numpy()).mean() * 100
    smooth = model.effect_ * 100
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(raw.index, raw.values, width=0.8, color=np.where(raw.values >= 0, BLUE, RED), alpha=0.55)
    ax.plot(smooth.index, smooth.values, color=TEXT, lw=2)
    ax.axhline(0, color=TEXT_MUTED, lw=1)
    ax.set_xlabel("Number at the end of the item id (FDA15 -> 15)")
    ax.set_ylabel("Sales vs the outlet's price line (%)")
    ax.set_xlim(0.3, 60.7)
    ax.grid(axis="x", visible=False)
    handles = [
        matplotlib.patches.Patch(color=BLUE, alpha=0.55, label="average for that number: sells above the line"),
        matplotlib.patches.Patch(color=RED, alpha=0.55, label="average for that number: sells below the line"),
        matplotlib.lines.Line2D([], [], color=TEXT, lw=2, label="smoothed effect used by v3"),
    ]
    ax.legend(handles=handles, frameon=False, loc="upper right", labelcolor=TEXT_MUTED, fontsize=9)
    fig.tight_layout()
    fig.savefig(config.FIGURE_DIR / "id_number_effect.png", dpi=150)
    plt.close(fig)


def save_sales_vs_mrp_by_outlet_type(X, y):
    """Small multiples: one panel per outlet type, each with its least-squares line."""
    types = ["Grocery Store", "Supermarket Type1", "Supermarket Type2", "Supermarket Type3"]
    outlet_type = X["Outlet_Type"].astype(str).to_numpy()
    mrp, sales = X["Item_MRP"].to_numpy(), np.asarray(y)
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.4), sharey=True)
    for ax, t in zip(axes, types):
        rows = outlet_type == t
        ax.scatter(mrp[rows], sales[rows], s=3, alpha=0.25, color=BLUE, linewidths=0)
        slope, intercept = np.polyfit(mrp[rows], sales[rows], 1)
        grid = np.array([mrp.min(), mrp.max()])
        ax.plot(grid, intercept + slope * grid, color=TEXT, lw=2)
        ax.set_title(f"{t}\n~{slope:.1f} sales per rupee of MRP", fontsize=10)
        ax.set_xlabel("Item_MRP")
    axes[0].set_ylabel("Item_Outlet_Sales")
    fig.tight_layout()
    fig.savefig(config.FIGURE_DIR / "sales_vs_mrp_by_outlet_type.png", dpi=150)
    plt.close(fig)
