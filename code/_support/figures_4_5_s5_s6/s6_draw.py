"""Drawing functions adapted from signed original sources; no experiment code runs."""
import math
import sys
from types import SimpleNamespace
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, Patch
from matplotlib.collections import PathCollection
from matplotlib.legend_handler import HandlerTuple
from matplotlib.text import Text
from matplotlib.ticker import FuncFormatter, FormatStrFormatter, MaxNLocator, StrMethodFormatter
from matplotlib.transforms import Bbox
CITIES = {
    "zhengzhou": "Zhengzhou",
    "hangzhou": "Hangzhou",
    "fuzhou": "Fuzhou",
    "chengdu": "Chengdu",
    "wuhan": "Wuhan",
}
TARGETS = (0, 1, 2, *range(5, 101, 5))
STRATEGIES = {
    "risk_priority": ("Risk-burden priority", "#CC79A7"),
    "population_priority": ("Exposure priority", "#D99A5B"),
    "random_priority": ("Random", "#7C858E"),
}
PERIODS = "10,20,50,100,200"
SIZE_MM = (155, 149)
DPI = 600
INSET_BOUNDS = (0.662, 0.11, 0.313, 0.29)
BAND_ALPHA = 0.20
STYLE = {
    "font.family": "Times New Roman",
    "font.size": 7,
    "axes.labelsize": 7,
    "axes.titlesize": 8,
    "axes.linewidth": 0.65,
    "axes.unicode_minus": False,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.width": 0.65,
    "ytick.major.width": 0.65,
    "legend.fontsize": 7,
    "legend.frameon": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "text.usetex": False,
}
def plot_city(ax, rows, city_name, letter, inset_top):
    ax.set(xlim=(0, 100), ylim=(0, 102), xticks=range(0, 101, 20),
           yticks=range(0, 101, 20))
    ax.tick_params(length=2.5, pad=2)
    ax.tick_params(top=False, right=False)
    ax.set_title(city_name, pad=6)
    ax.text(-0.085, 1.03, letter, transform=ax.transAxes,
            fontsize=10, fontweight="bold", va="bottom")

    inset = ax.inset_axes(INSET_BOUNDS)
    inset.set(xlim=(0, 10), ylim=(0, inset_top + 2),
              xticks=[0, 1, 2, 5, 10], yticks=np.linspace(0, inset_top, 4))
    inset.set_title('0–10% target', loc='left', fontsize=5.5, pad=2)
    inset.tick_params(labelsize=5.5, length=2, width=0.45, pad=1.4)
    for spine in inset.spines.values():
        spine.set_linewidth(0.45)

    for strategy, (label, color) in STRATEGIES.items():
        curve = rows.loc[rows.strategy.eq(strategy)].sort_values("restoration_pct")
        targets = curve.restoration_pct.to_numpy()
        reductions = curve.mean_EFARB_reduction.to_numpy()
        lower = curve.CI95_lower.to_numpy()
        upper = curve.CI95_upper.to_numpy()
        if tuple(targets) != TARGETS or not np.isfinite([reductions, lower, upper]).all():
            raise ValueError()
        if np.any(lower > upper) or lower.min() < -1e-8 or upper.max() > 100 + 1e-8:
            raise ValueError()
        ax.fill_between(targets, lower, upper, color=color, alpha=BAND_ALPHA, linewidth=0)
        ax.plot(targets, reductions, color=color, lw=1.15, solid_capstyle="round")

        # The inset repeats the computed target nodes. Open markers identify 1% and 2%.
        early = targets <= 10
        marked = (targets == 1) | (targets == 2)
        inset.fill_between(targets[early], lower[early], upper[early],
                           color=color, alpha=BAND_ALPHA, linewidth=0)
        inset.plot(targets[early], reductions[early], color=color, lw=0.85,
                   solid_capstyle="round")
        inset.plot(targets[marked], reductions[marked], color=color, ls="none",
                   marker="o", ms=2.6, mfc="white", mew=0.65, zorder=4)
        inset_left = 100 * INSET_BOUNDS[0]
        inset_roof = 102 * (INSET_BOUNDS[1] + INSET_BOUNDS[3])
        if np.interp(inset_left, targets, reductions) <= inset_roof:
            raise ValueError()
def draw_legend(ax):
    ax.set_axis_off()
    handles = [Line2D([], [], lw=1.15, color=color) for label, color in STRATEGIES.values()]
    labels = [label for label, color in STRATEGIES.values()]
    bands = tuple(Patch(facecolor=color, alpha=BAND_ALPHA, edgecolor="none")
                  for label, color in STRATEGIES.values())
    ax.legend([*handles, bands], [*labels, "95% CI"],
              loc="upper left", bbox_to_anchor=(0.045, 0.88),
              handlelength=2.3, labelspacing=0.53, borderaxespad=0,
              handler_map={tuple: HandlerTuple(ndivide=None, pad=0.1)})
def build_figure(curves):
    observed = set(zip(curves.city, curves.strategy))
    expected = {(city, strategy) for city in CITIES for strategy in STRATEGIES}
    if observed != expected:
        raise ValueError()
    if not curves.groupby("city").return_periods.nunique().eq(1).all():
        raise ValueError()
    coverage = curves.groupby("city", sort=False).return_periods.first().to_dict()
    early_max = curves.loc[curves.restoration_pct.le(10), "CI95_upper"].max()
    # Share the inset scale and leave room for the upper range across all cities.
    inset_top = 3 * max(10, 5 * int(np.ceil(early_max / 15)))

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(3, 2, figsize=tuple(size / 25.4 for size in SIZE_MM), dpi=DPI)
        fig.subplots_adjust(left=0.109, right=0.965, bottom=0.082, top=0.935,
                            wspace=0.168, hspace=0.34)
        for letter, (city, city_name), ax in zip("abcde", CITIES.items(), axes.flat):
            if coverage[city] != PERIODS:
                periods = ", ".join(f"T{period}" for period in coverage[city].split(","))
                city_name = f"{city_name} ({periods})"
            plot_city(ax, curves.loc[curves.city.eq(city)], city_name, letter, inset_top)
        draw_legend(axes[2, 1])
        fig.text(0.537, 0.032, "Road restoration target (%)", ha="center", va="center")
        fig.text(0.030, 0.508, "EFARB reduction (%)", rotation="vertical",
                 ha="center", va="center")
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        for ax in [*axes.flat, *(child for ax in axes.flat for child in ax.child_axes)]:
            bounds = ax.get_tightbbox(renderer)
            if bounds and not all(fig.bbox.contains(x, y) for x, y in bounds.get_points()):
                raise ValueError()
        for ax in axes.flat:
            for inset in ax.child_axes:
                bounds = inset.get_tightbbox(renderer)
                for line in ax.lines:
                    curve_path = line.get_path().transformed(line.get_transform())
                    if curve_path.intersects_bbox(bounds, filled=False):
                        raise ValueError()
    return fig
