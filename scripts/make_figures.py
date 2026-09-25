"""Publication figures for the label-noise scaling sweep.

Reads results/*.json and writes vector PDF (for LaTeX) plus 300 dpi PNG
(for the README and slides) into figures/.

    python scripts/make_figures.py

Design choices that matter for a paper:

* Markers are the measured runs; the line is the *fitted* power law, not a
  connect-the-dots. A reader can therefore judge the fit, not just the trend.
* Each series carries a distinct marker and dash pattern as well as a colour
  step, so the figures survive grayscale printing and colour-vision deficiency.
* The colour ramp is single-hue and ordered light to dark, because noise rate
  is an ordered quantity rather than four unrelated categories.
* The collapsed run (507 lines at 40% noise) is drawn as a hollow marker and
  excluded from every fit, so its exclusion is visible rather than silent.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from train import aggregate_results  # noqa: E402

FIG_DIR = ROOT / "figures"
RESULTS_DIR = ROOT / "results"

RATES = [0.0, 0.1, 0.2, 0.4]
# Single-hue ordinal ramp, light -> dark with noise rate.
COLOURS = {0.0: "#6da7ec", 0.1: "#3987e5", 0.2: "#256abf", 0.4: "#0d366b"}
MARKERS = {0.0: "o", 0.1: "s", 0.2: "^", 0.4: "D"}
DASHES = {0.0: (1, 0), 0.1: (5, 1.6), 0.2: (3, 1.4), 0.4: (1.4, 1.4)}
LABEL = {0.0: "0%", 0.1: "10%", 0.2: "20%", 0.4: "40%"}

GREY = "#4a4e5e"
LIGHT = "#d8dce6"
FLAG = "#b03428"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 9,
    "axes.labelsize": 9.5,
    "axes.titlesize": 10,
    "legend.fontsize": 8.5,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "axes.edgecolor": GREY,
    "axes.linewidth": 0.8,
    "xtick.color": GREY,
    "ytick.color": GREY,
    "text.color": "#14161c",
    "axes.labelcolor": "#14161c",
    "figure.dpi": 120,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,   # embed TrueType; required by most venues
    "ps.fonttype": 42,
})


def fit_power_law(n: np.ndarray, cer: np.ndarray) -> dict:
    """Least squares on log(CER) = log(a) - b*log(N). Returns b, a, CI, R^2."""
    x, y = np.log(n), np.log(cer)
    slope, intercept = np.polyfit(x, y, 1)
    pred = np.polyval([slope, intercept], x)
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    dof = len(x) - 2
    se = math.sqrt(ss_res / dof / ((x - x.mean()) ** 2).sum()) if dof > 0 else float("nan")
    t_crit = {1: 12.706, 2: 4.303, 3: 3.182}.get(dof, 1.96)
    return {
        "b": -slope,
        "a": math.exp(intercept),
        "ci": t_crit * se,
        "r2": 1 - ss_res / ss_tot if ss_tot else float("nan"),
        "n_points": len(x),
    }


def load() -> tuple[dict, dict, list]:
    df = aggregate_results(RESULTS_DIR)
    if df.empty:
        raise SystemExit(f"no result files found in {RESULTS_DIR}")
    valid = df[df.val_cer < 0.999]
    excluded = df[df.val_cer >= 0.999][["n_train_lines", "noise_rate", "val_cer"]]

    series, fits = {}, {}
    for r in RATES:
        g = valid[valid.noise_rate == r].sort_values("n_train_lines")
        if g.empty:
            continue
        n = g.n_train_lines.to_numpy(float)
        cer = g.val_cer.to_numpy(float)
        series[r] = (n, cer)
        fits[r] = fit_power_law(n, cer)
    return series, fits, list(excluded.itertuples(index=False))


def figure_scaling(series: dict, fits: dict, excluded: list) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6.4, 4.3))

    xs = np.concatenate([n for n, _ in series.values()])
    grid = np.logspace(np.log10(xs.min() * 0.88), np.log10(xs.max() * 1.12), 100)

    for r in RATES:
        if r not in series:
            continue
        n, cer = series[r]
        f = fits[r]
        ax.plot(grid, f["a"] * grid ** (-f["b"]), color=COLOURS[r],
                lw=1.3, dashes=DASHES[r], zorder=2, solid_capstyle="round")
        ax.plot(n, cer, marker=MARKERS[r], ms=5.5, ls="none",
                color=COLOURS[r], mec="white", mew=0.8, zorder=3,
                label=f"{LABEL[r]}  ($b$ = {f['b']:.2f}, $R^2$ = {f['r2']:.3f})")

    # The collapsed run, shown but not fitted.
    for row in excluded:
        ax.plot(row.n_train_lines, 0.62, marker="x", ms=7, mew=1.6,
                color=FLAG, ls="none", zorder=4, clip_on=False)
        ax.annotate("excluded:\nCER = 1.00", xy=(row.n_train_lines, 0.62),
                    xytext=(11, -2), textcoords="offset points",
                    color=FLAG, fontsize=7.6, va="center", linespacing=1.35)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Training lines $N$")
    ax.set_ylabel("Validation CER")

    ticks = sorted({int(v) for n, _ in series.values() for v in n})
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:,}" for t in ticks])
    ax.set_xticks([], minor=True)

    yt = [0.10, 0.15, 0.20, 0.30, 0.40]
    ax.set_yticks(yt)
    ax.set_yticklabels([f"{v:.2f}" for v in yt])
    ax.set_yticks([], minor=True)
    ax.set_ylim(0.085, 0.70)

    ax.grid(True, which="major", color=LIGHT, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    leg = ax.legend(title="Label noise rate", loc="upper right",
                    frameon=True, framealpha=1, edgecolor=LIGHT,
                    borderpad=0.7, labelspacing=0.55, handletextpad=0.7)
    leg.get_frame().set_linewidth(0.6)
    leg.get_title().set_fontsize(8.5)
    fig.tight_layout()
    return fig


def figure_exponents(fits: dict) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6.4, 2.5))
    fig.subplots_adjust(right=0.70)

    ys = list(range(len(RATES)))[::-1]
    for y, r in zip(ys, RATES):
        if r not in fits:
            continue
        f = fits[r]
        ax.errorbar(f["b"], y, xerr=f["ci"], fmt=MARKERS[r], ms=6,
                    color=COLOURS[r], ecolor=COLOURS[r], elinewidth=1.4,
                    capsize=4, capthick=1.4, mec="white", mew=0.8, zorder=3)
        note = f"$b$ = {f['b']:.2f} ± {f['ci']:.2f}"
        if f["n_points"] < 4:
            note += f"  ({f['n_points']} pts)"
        # Outside the axes, so the plotted range covers only the data and no
        # gridline runs on past it into empty space.
        ax.annotate(note, xy=(1.03, y), xycoords=("axes fraction", "data"),
                    ha="left", va="center", fontsize=8, color=COLOURS[r],
                    annotation_clip=False)

    ax.set_yticks(ys)
    ax.set_yticklabels([f"{LABEL[r]} noise" for r in RATES])
    ax.set_xlabel(r"Fitted exponent $b$ in CER $\propto N^{-b}$")
    ax.set_xlim(0.26, 0.80)
    ax.set_ylim(-0.6, len(RATES) - 0.4)
    ax.grid(True, axis="x", color=LIGHT, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)
    return fig


def main() -> None:
    FIG_DIR.mkdir(exist_ok=True)
    series, fits, excluded = load()

    outputs = [
        ("fig1_scaling_curves", figure_scaling(series, fits, excluded)),
        ("fig2_exponents", figure_exponents(fits)),
    ]
    for name, fig in outputs:
        for ext, dpi in (("pdf", None), ("png", 300)):
            path = FIG_DIR / f"{name}.{ext}"
            fig.savefig(path, dpi=dpi) if dpi else fig.savefig(path)
            print(f"wrote {path.relative_to(ROOT)}")
        plt.close(fig)

    print("\nfitted parameters")
    print(f"{'noise':>7} {'points':>7} {'b':>7} {'95% CI':>9} {'a':>8} {'R^2':>8}")
    for r in RATES:
        if r not in fits:
            continue
        f = fits[r]
        print(f"{r:>7} {f['n_points']:>7} {f['b']:>7.3f} {f['ci']:>9.3f} "
              f"{f['a']:>8.2f} {f['r2']:>8.4f}")


if __name__ == "__main__":
    main()
