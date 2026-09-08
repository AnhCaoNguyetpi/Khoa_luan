"""Small plotting helpers used by the experiment drivers."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def save_fig(fig, path, dpi=150):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def grouped_bars(df: pd.DataFrame, x: str, y: str, hue: str = None,
                 title: str = "", ylabel: str = "", save=None):
    if hue and hue != x:
        piv = df.pivot_table(index=x, columns=hue, values=y, aggfunc="mean")
        ax = piv.plot(kind="bar", figsize=(1.4 * len(piv) + 3, 4))
        ax.legend(title=hue, fontsize=8)
    else:
        piv = df.groupby(x)[y].mean()
        ax = piv.plot(kind="bar", figsize=(1.4 * len(piv) + 3, 4),
                      color="#4c78a8")
    ax.set_ylabel(ylabel or y)
    ax.set_xlabel(x)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    return save_fig(ax.figure, save) if save else ax


def sweep_lines(df: pd.DataFrame, x: str, y: str, hue: str,
                title: str = "", xlabel: str = "", ylabel: str = "",
                save=None, logy=False, ylim=None):
    fig, ax = plt.subplots(figsize=(6.4, 4))
    for key, g in df.groupby(hue):
        g = g.sort_values(x)
        ax.plot(g[x], g[y], marker="o", label=str(key))
    ax.set_xlabel(xlabel or x)
    ax.set_ylabel(ylabel or y)
    if logy:
        ax.set_yscale("log")
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(alpha=0.3)
    ax.set_title(title)
    ax.legend(fontsize=8)
    return save_fig(fig, save) if save else fig


def survival_curves(curves: Dict[str, tuple], title: str = "",
                    xlabel="mission time [min]",
                    ylabel="P(not yet detected)", save=None):
    """``curves``: name -> (times, fraction_undetected)."""
    fig, ax = plt.subplots(figsize=(6.4, 4))
    for name, (ts, frac) in curves.items():
        ax.step(ts, frac, where="post", label=name)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.set_title(title)
    ax.legend(fontsize=8)
    return save_fig(fig, save) if save else fig


def factorial_chart(summary: Dict[str, float], labels=("A", "B", "C", "D"),
                    metric="DSR", title="Static x Dynamic factorial",
                    save=None):
    """A/B/C/D bars with contrast annotations C-A, B-A, D-A."""
    vals = [summary.get(k, np.nan) for k in labels]
    fig, ax = plt.subplots(figsize=(5.6, 4))
    bars = ax.bar(labels, vals, color=["#999999", "#6baed6", "#fd8d3c", "#41ab5d"])
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.set_ylim(0, max(1.0, np.nanmax(vals) * 1.25))
    for b, v in zip(bars, vals):
        if np.isfinite(v):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=9)
    if len(vals) == 4 and all(np.isfinite(vals)):
        notes = []
        if labels[0] == "A":
            notes.append(f"C-A={vals[2]-vals[0]:+.2f} (data-driven, static)")
            notes.append(f"B-A={vals[1]-vals[0]:+.2f} (dynamic, baseline)")
            notes.append(f"D-A={vals[3]-vals[0]:+.2f} (full framework)")
        ax.text(0.98, 0.97, "\n".join(notes), transform=ax.transAxes,
                ha="right", va="top", fontsize=8,
                bbox=dict(fc="white", ec="#cccccc", alpha=0.9))
    ax.grid(axis="y", alpha=0.3)
    return save_fig(fig, save) if save else fig
