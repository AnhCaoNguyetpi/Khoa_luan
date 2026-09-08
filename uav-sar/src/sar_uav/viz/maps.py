"""Map-style visualisation of beliefs, terrain and mission frames."""

from __future__ import annotations

from typing import Dict, Iterable, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


def _terrain_base(ax, area):
    img = np.ones((area.H, area.W, 3))
    # green intensity by vegetation, grey by bareness
    veg = np.clip(area.veg, 0, 1)
    slope_n = np.clip(area.slope / 45.0, 0, 1)
    img[..., 0] = 0.92 - 0.55 * veg - 0.15 * slope_n
    img[..., 1] = 0.95 - 0.45 * veg - 0.25 * slope_n
    img[..., 2] = 0.85 - 0.35 * veg
    img[area.water] = (0.45, 0.65, 0.9)
    ax.imshow(img, origin="upper", extent=[0, area.W * area.cell,
                                           area.H * area.cell, 0])
    ys, xs = np.nonzero(area.trail_mask)
    if len(xs):
        ax.scatter((xs + 0.5) * area.cell, (ys + 0.5) * area.cell,
                   s=0.8, c="saddlebrown", marker="s", alpha=0.7)


def plot_belief(ax, area, belief, title="", cbar=True, vmin=0.0):
    """Belief heatmap over the terrain base map."""
    p2d = np.asarray(belief).reshape(area.shape).copy()
    p2d[area.water] = np.nan
    _terrain_base(ax, area)
    im = ax.imshow(p2d, origin="upper", cmap="inferno", alpha=0.75,
                   extent=[0, area.W * area.cell, area.H * area.cell, 0])
    if cbar:
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="p_i")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")


def render_mission_frame(area, belief, uav_xy: Dict[str, np.ndarray],
                         target_cells: Iterable[int], ipp_cell: int,
                         bases: Dict[str, np.ndarray], title="",
                         save=None, ax=None):
    """One snapshot: belief + UAV positions + true target path + IPP/bases."""
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(7, 6))
    plot_belief(ax, area, belief, title=title)
    tc = list(target_cells)
    if tc:
        rr, cc = np.divmod(np.asarray(tc), area.W)
        ax.plot((cc + 0.5) * area.cell, (rr + 0.5) * area.cell,
                "r--", lw=1.4, alpha=0.85, label="target (truth)")
        ax.plot((cc[-1] + 0.5) * area.cell, (rr[-1] + 0.5) * area.cell,
                "r*", ms=16, label="target now")
    for name, xy in uav_xy.items():
        ax.plot(xy[0], xy[1], "^", ms=10, label=name)
    for name, bxy in bases.items():
        ax.plot(bxy[0], bxy[1], "s", ms=9, mfc="white", mec="k")
    ir, ic = np.divmod(int(ipp_cell), area.W)
    ax.plot((ic + 0.5) * area.cell, (ir + 0.5) * area.cell, "w*", ms=14,
            mec="k", label="IPP")
    ax.legend(loc="upper right", fontsize=7, framealpha=0.8)
    if standalone:
        if save is not None:
            fig.savefig(save, dpi=140, bbox_inches="tight")
            plt.close(fig)


def compare_initial_beliefs(area, maps: Dict[str, np.ndarray],
                            ipp_cell: int, find_cell: Optional[int] = None,
                            elapsed_min=None, save=None):
    """Side-by-side initial probability maps for RQ1 figures."""
    n = len(maps)
    fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 4.4))
    if n == 1:
        axes = [axes]
    for ax, (name, p) in zip(axes, maps.items()):
        plot_belief(ax, area, p, title=name)
        ir, ic = np.divmod(int(ipp_cell), area.W)
        ax.plot((ic + 0.5) * area.cell, (ir + 0.5) * area.cell,
                "w*", ms=13, mec="k")
        if find_cell is not None:
            fr, fc = np.divmod(int(find_cell), area.W)
            ax.plot((fc + 0.5) * area.cell, (fr + 0.5) * area.cell,
                    "cP", ms=11, mec="k")
    fig.suptitle("Initial belief models"
                 + (f"  (elapsed={elapsed_min:.0f} min)" if elapsed_min else ""),
                 y=1.02)
    fig.tight_layout()
    if save is not None:
        fig.savefig(save, dpi=150, bbox_inches="tight")
        plt.close(fig)
