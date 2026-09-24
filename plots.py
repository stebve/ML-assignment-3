"""
Plotting helpers for GridSearchCV results (Elman / Jordan / MRNN on flights).

All functions assume the scorers return NEGATIVE errors (as passenger_rmse and
passenger_mape do), and flip the sign so plots show errors: lower = better.

    plot_param_effects(gs, metric="rmse")         how sensitive is the score to each hyperparameter?
    plot_param_effects_all(searches)              the same, all three networks in one figure

Paper figures (IEEE two-column): call ieee_style() once, then pass
width=IEEE_COLUMN (3.5 in) or width=IEEE_PAGE (7.16 in), ncols=..., title=False,
and save as PDF.
    plot_param_heatmap(gs, "lags", "units")       two hyperparameters at once
    plot_top_configs(gs, n=10)                    best configurations, with spread across folds
    plot_fold_comparison(searches)                best Elman vs Jordan vs MRNN, fold by fold
    plot_test_forecast(searches, X_test, y_test, dates)   forecasts vs actual passengers
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Fixed colours per architecture (colourblind-checked order) + neutral grey for secondary marks.
ARCH_COLORS = {"elman": "#2a78d6", "jordan": "#eb6834", "mrnn": "#1baf7a"}
NEUTRAL_COLOR = "#8a8984"
INK, INK_MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
METRIC_LABELS = {"rmse": "RMSE (passengers)", "mape": "MAPE (%)", "score": "error"}

plt.rcParams.update({
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": INK_MUTED, "axes.labelcolor": INK, "axes.titleweight": "bold",
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "xtick.color": INK_MUTED, "ytick.color": INK_MUTED, "legend.frameon": False,
    "figure.dpi": 110,
})


# ---------------------------------------------------------------- helpers
def _key(gs, metric):
    k = f"mean_test_{metric}"
    return k if k in gs.cv_results_ else "mean_test_score"


def _errors(gs, metric):
    """cv_results_ as a DataFrame with positive mean/std error per configuration."""
    cv = pd.DataFrame(gs.cv_results_)
    k = _key(gs, metric)
    cv["error"] = -cv[k]
    cv["error_std"] = cv[k.replace("mean_", "std_")]
    return cv


def _varied_params(gs):
    """Hyperparameters that actually took more than one value in the grid."""
    params = [c for c in gs.cv_results_ if c.startswith("param_")]
    return [p[6:] for p in params if len(set(map(str, gs.cv_results_[p]))) > 1]


def _fold_errors(gs, metric):
    """Per-fold errors of the best configuration."""
    k = _key(gs, metric).replace("mean_test_", "")
    n = sum(1 for c in gs.cv_results_ if c.startswith("split") and c.endswith(f"_test_{k}"))
    i = gs.best_index_
    return np.array([-gs.cv_results_[f"split{f}_test_{k}"][i] for f in range(n)])


def _label(metric):
    return METRIC_LABELS.get(metric, metric)


def _ylabel(metric, compact):
    """Axis label for CV error; short in paper mode (put units in the caption)."""
    return {"rmse": "CV RMSE", "mape": "CV MAPE (%)"}.get(metric, "CV error") if compact \
        else f"Mean CV {_label(metric)}"


# ---------------------------------------------------------------- IEEE / paper layout
IEEE_COLUMN = 3.5      # inches, one column of an IEEE two-column paper
IEEE_PAGE = 7.16       # inches, full text width (figure* spanning both columns)


def ieee_style():
    """
    Switch matplotlib to paper settings: 8 pt serif text (matches IEEE body text),
    thinner lines, embedded fonts. Call once before plotting figures for the report.
    Save with fig.savefig("name.pdf") and include in LaTeX at width=\\columnwidth
    (or \\textwidth for a figure*); the figure is already that size, so nothing is rescaled.
    """

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "lines.linewidth": 1.2, "lines.markersize": 4, "axes.linewidth": 0.6,
        "grid.linewidth": 0.5, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "pdf.fonttype": 42, "ps.fonttype": 42,          # editable, embedded TrueType fonts
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "savefig.dpi": 300,
    })


def _grid(n_panels, ncols, width, panel_h):
    ncols = ncols or n_panels
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(width, panel_h * nrows), sharey=True, squeeze=False)
    spare = list(axes.flat[n_panels:])           # empty slots, e.g. 5 panels in a 3x2 grid
    for ax in spare:
        ax.axis("off")
    return fig, list(axes.flat[:n_panels]), spare


def _sorted_values(cv, p):
    return sorted(cv[f"param_{p}"].unique(), key=lambda v: (isinstance(v, str), v))


def _fmt(v):
    """
    Short tick labels: 0.001 -> 1e-3, 0.0 -> 0.
    """

    if isinstance(v, (float, np.floating)):
        if v == 0:
            return "0"
        if abs(v) < 0.01:
            return f"{v:.0e}".replace("e-0", "e-")
        return f"{v:g}"
    return str(v)


# ---------------------------------------------------------------- 1. sensitivity
def plot_param_effects(gs, metric="rmse", title=None, color=None, ncols=None, width=None, panel_height=None):
    """
    One panel per hyperparameter. Small grey dots: every configuration using that
    value. Large dot: the best configuration with that value. A flat row of large
    dots means the hyperparameter hardly matters; a steep one means it does.

    Layout: ncols panels per row (default: all in one row). For an IEEE column,
    use ieee_style() and e.g. ncols=3, width=IEEE_COLUMN. title=False hides the
    title (the LaTeX caption does that job in a paper).
    """

    cv = _errors(gs, metric)
    params = _varied_params(gs)
    color = color or ARCH_COLORS.get(getattr(gs.estimator, "arch", ""), ARCH_COLORS["elman"])
    compact = width is not None
    width = width or 3.2 * (ncols or len(params))
    panel_height = panel_height or (1.25 if compact else 3.2)
    fig, axs, spare = _grid(len(params), ncols, width, panel_height)
    for ax, p in zip(axs, params):
        vals = _sorted_values(cv, p)
        xs = np.arange(len(vals))
        rng = np.random.default_rng(0)
        for x, v in zip(xs, vals):
            e = cv.loc[cv[f"param_{p}"] == v, "error"]
            ax.scatter(x + rng.uniform(-0.12, 0.12, len(e)), e, s=6 if compact else 14,
                       color=NEUTRAL_COLOR, alpha=0.5, lw=0)
        best = [cv.loc[cv[f"param_{p}"] == v, "error"].min() for v in vals]
        ax.plot(xs, best, color=color, marker="o", mec="white", zorder=3,
                lw=1.2 if compact else 2, ms=4 if compact else 7, mew=0.8 if compact else 1.5)
        ax.set_xticks(xs, [_fmt(v) for v in vals])
        ax.set_xlabel(p, labelpad=1 if compact else 4)
        ax.set_xlim(-0.5, len(vals) - 0.5)
    fig.supylabel(_ylabel(metric, compact), fontsize=plt.rcParams["axes.labelsize"], x=0.0)
    if title is not False:
        fig.suptitle(title or "Effect of each hyperparameter (lower is better)", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout(pad=0.3 if compact else 1.08, w_pad=0.4 if compact else None, h_pad=0.6 if compact else None)
    return fig


def plot_param_effects_all(searches, metric="rmse", title=None, ncols=None, width=None, panel_height=None):
    """
    All three architectures in ONE figure: one panel per hyperparameter, one line per
    network (best CV error at each value). Replaces three separate figures, which
    saves a lot of space in a paper.

    searches : {"elman": gs_elman, "jordan": gs_jordan, "mrnn": gs_mrnn}
    Paper use: ieee_style(); plot_param_effects_all(searches, ncols=3, width=IEEE_COLUMN, title=False)
    """

    cvs = {name: _errors(gs, metric) for name, gs in searches.items()}
    params = []
    for gs in searches.values():                       # union of varied params, in grid order
        params += [p for p in _varied_params(gs) if p not in params]
    compact = width is not None
    width = width or 3.2 * (ncols or len(params))
    panel_height = panel_height or (1.25 if compact else 3.2)
    fig, axs, spare = _grid(len(params), ncols, width, panel_height)
    offsets = np.linspace(-0.1, 0.1, len(cvs))           # nudge lines apart so markers don't hide each other
    for ax, p in zip(axs, params):
        all_vals = sorted({v for cv in cvs.values() if f"param_{p}" in cv for v in cv[f"param_{p}"].unique()},
                          key=lambda v: (isinstance(v, str), v))
        pos = {v: i for i, v in enumerate(all_vals)}
        for off, (name, cv) in zip(offsets, cvs.items()):
            if f"param_{p}" not in cv:
                continue
            vals = _sorted_values(cv, p)
            best = [cv.loc[cv[f"param_{p}"] == v, "error"].min() for v in vals]
            ax.plot([pos[v] + off for v in vals], best, color=ARCH_COLORS.get(name), marker="o",
                    mec="white", label=name, lw=1.2 if compact else 2,
                    ms=4 if compact else 7, mew=0.8 if compact else 1.5)
        ax.set_xticks(range(len(all_vals)), [_fmt(v) for v in all_vals])
        ax.set_xlabel(p, labelpad=1 if compact else 4)
        ax.set_xlim(-0.5, len(all_vals) - 0.5)
    fig.supylabel(_ylabel(metric, compact), fontsize=plt.rcParams["axes.labelsize"], x=0.0)
    handles, labels = axs[0].get_legend_handles_labels()
    if title is not False:
        fig.suptitle(title or "Effect of each hyperparameter, best configuration per value (lower is better)",
                     x=0.01, ha="left", fontweight="bold")
    fig.tight_layout(pad=0.3 if compact else 1.08, w_pad=0.4 if compact else None, h_pad=0.6 if compact else None)
    if spare:                                    # use an empty grid slot for the legend
        spare[0].legend(handles, labels, loc="center", frameon=False, handlelength=1.5)
    else:                                        # otherwise a strip above the panels
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=len(labels),
                   frameon=False, handlelength=1.5, columnspacing=1.2)
    return fig


# ---------------------------------------------------------------- 3. leaderboard
def plot_top_configs(gs, n=10, metric="rmse", title=None, color=None):
    """
    Top-n configurations: mean CV error with ±1 std across folds.
    """

    cv = _errors(gs, metric).nsmallest(n, "error").iloc[::-1]
    params = _varied_params(gs)
    labels = [", ".join(f"{p}={row[f'param_{p}']}" for p in params) for _, row in cv.iterrows()]
    color = color or ARCH_COLORS.get(getattr(gs.estimator, "arch", ""), ARCH_COLORS["elman"])
    fig, ax = plt.subplots(figsize=(7.5, 0.38 * len(cv) + 1.4))
    y = np.arange(len(cv))
    ax.errorbar(cv["error"], y, xerr=cv["error_std"], fmt="o", ms=7, color=color,
                ecolor=NEUTRAL_COLOR, elinewidth=1.5, capsize=3, mec="white", mew=1.5)
    ax.set_yticks(y, labels, fontsize=9)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(f"Mean CV {_label(metric)}  (bars: ±1 std across folds)")
    ax.set_title(title or f"Top {len(cv)} configurations", loc="left")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- 4. architectures, fold by fold
def plot_fold_comparison(searches, metric="rmse", title=None):
    """
    searches : dict like {"elman": gs_elman, "jordan": gs_jordan, "mrnn": gs_mrnn}
    Each line is the BEST configuration of one architecture, scored on identical folds.
    """

    fig, ax = plt.subplots(figsize=(7.5, 4))
    series = {name: _fold_errors(gs, metric) for name, gs in searches.items()}
    n_folds = len(next(iter(series.values())))
    x = np.arange(1, n_folds + 1)
    for name, e in series.items():
        ax.plot(x, e, lw=2, marker="o", ms=7, mec="white", mew=1.5,
                color=ARCH_COLORS.get(name), label=f"{name} (mean {e.mean():.2f})")
    # end-of-line labels, nudged apart so they never overlap
    ends = sorted(((e[-1], name) for name, e in series.items()))
    lo, hi = ax.get_ylim()
    gap, placed = 0.07 * (hi - lo), []
    for yv, _ in ends:
        placed.append(max(yv, placed[-1] + gap) if placed else yv)
    shift = np.mean([yv for yv, _ in ends]) - np.mean(placed)   # keep the stack centred on the lines
    for (yv, name), yl in zip(ends, placed):
        ax.annotate(name, (x[-1], yv), xytext=(x[-1] + 0.12, yl + shift), textcoords="data",
                    va="center", fontsize=9, color=INK_MUTED)
    ax.set_ylim(min(lo, placed[0] + shift - gap), max(hi, placed[-1] + shift + gap))
    ax.set_xticks(x, [f"Fold {i}" for i in x])
    ax.set_xlim(0.7, n_folds + 1.1)
    ax.set_ylabel(_label(metric))
    ax.legend(loc="upper left", fontsize=9)
    ax.set_title(title or "Best configuration per architecture, fold by fold", loc="left")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- 5. forecasts in passengers
def plot_test_forecast(searches, X_test, y_test, dates, title=None):
    """
    Actual vs predicted passengers on the held-out test months.
    X_test must carry the back-transform base in column 0 (as in the scorer setup).
    """

    b = X_test[:, 0]
    actual = np.exp(y_test + b)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(dates, actual, color=INK, lw=2.5, label="actual", zorder=4)
    for name, gs in searches.items():
        pred = np.exp(gs.best_estimator_.predict(X_test) + b)
        rmse = np.sqrt(np.mean((pred - actual) ** 2))
        ax.plot(dates, pred, lw=2, marker="o", ms=5, color=ARCH_COLORS.get(name),
                label=f"{name} (RMSE {rmse:.1f})")
    ax.set_ylabel("Passengers (thousands)")
    ax.legend(fontsize=9, ncol=2, loc="upper left")
    ax.set_title(title or "Held-out test period: one-step-ahead forecasts", loc="left")
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig
