import matplotlib
matplotlib.use("Agg")           # headless-safe on any OS

from .maps import plot_belief, render_mission_frame   # noqa: E402,F401
from .plots import save_fig, grouped_bars, sweep_lines, survival_curves  # noqa: E402,F401
