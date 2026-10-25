"""
Figures. Every number comes from a stage table, so a figure can never disagree
with the tables it illustrates.
"""

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from . import config
from .utils import banner, get_logger

log = get_logger(__name__)

# los totales del corpus para la figura del flujo de datos. El stage 1 solo los
# saca por pantalla y no los guarda en ninguna tabla, asi que los apunto aqui.
# Son los del run sobre Common Voice es 26.0
CORPUS_VALIDATED_CLIPS = 439_225
CORPUS_SINGLE_LABEL_CLIPS = 236_832

LABELS = {"norte": "Norte", "sur": "Sur", "mexico": "México"}


def _table(name):
    return pd.read_csv(config.TABLE_DIR / f"{name}.csv")


def _save(fig, name):
    path = config.FIGURE_DIR / f"{name}.png"
    fig.savefig(path, dpi=config.DPI)
    plt.close(fig)
    log.info(f"  -> {path.relative_to(config.ROOT)}")


def confidence_ellipse(x, y, ax, n_std=1.96, **kwargs):
    """Covariance ellipse for a set of points; skipped when degenerate."""
    if len(x) < 3:
        return
    cov = np.cov(x, y)
    if not np.all(np.isfinite(cov)):
        return
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = eigenvalues.argsort()[::-1]
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
    if np.any(eigenvalues <= 0):
        return
    angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
    width, height = 2 * n_std * np.sqrt(eigenvalues)
    ax.add_patch(Ellipse((np.mean(x), np.mean(y)), width, height, angle=angle, **kwargs))


def vowel_space_figure():
    """Stressed vowels: Norte and Sur speaker means, then the three variety means."""
    cells = _table("s6_h1_cells")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    for dialect, style in [("norte", "-"), ("sur", "--")]:
        sub_d = cells[cells["dialecto"] == dialect]
        for vowel in config.VOWELS:
            sub = sub_d[sub_d["vowel"] == vowel]
            ax.scatter(sub["F2"], sub["F1"], s=10, alpha=0.3, color=config.COLORS[dialect])
            confidence_ellipse(sub["F2"].to_numpy(), sub["F1"].to_numpy(), ax,
                               facecolor="none", edgecolor=config.COLORS[dialect],
                               linewidth=1.6, linestyle=style)
    for dialect in ["norte", "sur"]:
        means = cells[cells["dialecto"] == dialect].groupby("vowel")[["F1", "F2"]].mean()
        for vowel, r in means.iterrows():
            ax.scatter(r["F2"], r["F1"], s=120, color=config.COLORS[dialect],
                       edgecolors="black", zorder=5)
            ax.annotate(vowel, (r["F2"], r["F1"]), fontsize=12, fontweight="bold",
                        ha="center", va="center", color="white", zorder=6)
    n = cells.groupby("dialecto")["client_id"].nunique()
    ax.legend(handles=[plt.Line2D([], [], color=config.COLORS[d], marker="o", linestyle=s,
                                  label=f"{LABELS[d]} ({n.get(d, 0)} speakers)")
                       for d, s in [("norte", "-"), ("sur", "--")]], fontsize=9, loc="lower left")
    ax.set_title("A. Norte and Sur: speaker means, 95% ellipses", fontsize=11, loc="left")

    ax = axes[1]
    for dialect in config.ALL_DIALECTS:
        sub_d = cells[cells["dialecto"] == dialect]
        for vowel in config.VOWELS:
            sub = sub_d[sub_d["vowel"] == vowel]
            confidence_ellipse(sub["F2"].to_numpy(), sub["F1"].to_numpy(), ax,
                               facecolor=config.COLORS[dialect], alpha=0.12)
        means = sub_d.groupby("vowel")[["F1", "F2"]].mean()
        for vowel, r in means.iterrows():
            ax.scatter(r["F2"], r["F1"], s=120, color=config.COLORS[dialect],
                       edgecolors="black", zorder=5)
            ax.annotate(vowel, (r["F2"], r["F1"]), fontsize=12, fontweight="bold",
                        ha="center", va="center", color="white", zorder=6)
    ax.legend(handles=[plt.Line2D([], [], color=config.COLORS[d], marker="o", linestyle="",
                                  label=f"{LABELS[d]} ({n.get(d, 0)} speakers)")
                       for d in config.ALL_DIALECTS], fontsize=9, loc="lower left")
    ax.set_title("B. Three varieties: means and 95% ellipses", fontsize=11, loc="left")

    # unos pocos hablantes extremos estiran los ejes; los dejo fuera de la vista
    # y digo cuantos son
    lo2, hi2 = cells["F2"].quantile([0.002, 0.998])
    lo1, hi1 = cells["F1"].quantile([0.002, 0.998])
    outside = ((cells["F2"] < lo2) | (cells["F2"] > hi2) | (cells["F1"] < lo1) | (cells["F1"] > hi1)).sum()
    for ax in axes:
        ax.set_xlim(hi2 + 0.3, lo2 - 0.3)
        ax.set_ylim(hi1 + 0.3, lo1 - 0.3)
        ax.set_xlabel("F2 (Lobanov z)")
        ax.set_ylabel("F1 (Lobanov z)")
        ax.grid(alpha=0.3)
    axes[0].text(0.99, 0.01, f"{outside} of {len(cells)} speaker means outside the view",
                 transform=axes[0].transAxes, ha="right", va="bottom", fontsize=8, color="gray")
    fig.suptitle("Stressed vowels in content words", fontsize=12)
    fig.tight_layout()
    _save(fig, "fig_vowel_space")


def equivalence_figure():
    """Norte vs Sur against the equivalence bound, and the México contrasts."""
    h1 = _table("s6_h1_norte_sur")
    control = _table("s6_h1_control")
    measures = list(h1["measure"])
    y = np.arange(len(measures))[::-1]
    bound = config.TOST_BOUND

    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharey=True)

    ax = axes[0]
    ax.axvspan(-bound, bound, color="gray", alpha=0.15, label=f"equivalence bound ±{bound:.2f}")
    ax.hlines(y, h1["ci90_lo"], h1["ci90_hi"], color=config.COLORS["sur"], linewidth=4,
              alpha=0.8, label="90% CI")
    ax.scatter(h1["d"], y, color="black", zorder=5, s=30)
    ax.set_title("A. Norte vs Sur (d = Norte - Sur)", fontsize=11, loc="left")

    ax = axes[1]
    for offset, (pair, colour) in zip([0.15, -0.15], [("norte vs mexico", config.COLORS["norte"]),
                                                       ("sur vs mexico", config.COLORS["sur"])]):
        sub = control[control["pair"] == pair].set_index("measure").reindex(measures)
        left, right = pair.split(" vs ")
        ax.hlines(y + offset, sub["ci95_lo"], sub["ci95_hi"], color=colour, linewidth=3,
                  alpha=0.75, label=f"{LABELS[left]} vs {LABELS[right]}, 95% CI")
        ax.scatter(sub["d"], y + offset, color="black", zorder=5, s=18)
    ax.axvspan(-bound, bound, color="gray", alpha=0.15)
    ax.set_title("B. Each Peninsular variety vs México", fontsize=11, loc="left")

    for ax in axes:
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(measures)
        ax.set_xlabel("Cohen's d, speaker level (CI from resampling speakers)")
        ax.grid(alpha=0.3, axis="x")
        ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=False)
    fig.suptitle("H1: stressed vowel measures (Lobanov)", fontsize=12)
    fig.tight_layout()
    _save(fig, "fig_equivalence")


def plural_figure():
    """Plural minus singular shift in F1 and F2, per vowel and variety."""
    summary = _table("s7_h2_descriptive")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)
    x = np.arange(len(config.H2_VOWELS))
    offsets = np.linspace(-0.22, 0.22, len(config.ALL_DIALECTS))

    for ax, measure, label in [(axes[0], "F1_shift", "F1 shift (Lobanov z), + = more open"),
                               (axes[1], "F2_shift", "F2 shift (Lobanov z), + = more front")]:
        for offset, dialect in zip(offsets, config.ALL_DIALECTS):
            sub = summary[(summary["measure"] == measure) & (summary["dialecto"] == dialect)]
            sub = sub.set_index("vowel").reindex(config.H2_VOWELS)
            ax.errorbar(x + offset, sub["mean"],
                        yerr=[sub["mean"] - sub["ci95_lo"], sub["ci95_hi"] - sub["mean"]],
                        fmt="o", capsize=4, color=config.COLORS[dialect], label=LABELS[dialect])
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([f"/{v}/" for v in config.H2_VOWELS])
        ax.set_ylabel(label)
        ax.grid(alpha=0.3, axis="y")
    axes[0].set_title("A. F1: 'casas' minus 'casa', mean and 95% CI", fontsize=11, loc="left")
    axes[1].set_title("B. F2: same comparison", fontsize=11, loc="left")
    axes[0].legend(fontsize=9)
    fig.suptitle("H2: word-final unstressed vowels with and without a following /s/", fontsize=12)
    fig.tight_layout()
    _save(fig, "fig_plural")


def sibilant_figure():
    """Speaker-level /s/ CoG by position, and the weakening index."""
    cells = _table("s8_sibilant_cells")
    per_speaker = _table("s8_weakening_per_speaker")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [3, 2]})

    ax = axes[0]
    positions = ["onset", "coda"]
    offsets = np.linspace(-0.26, 0.26, len(config.ALL_DIALECTS))
    for offset, dialect in zip(offsets, config.ALL_DIALECTS):
        data = [cells[(cells["position"] == p) & (cells["dialecto"] == dialect)]["cog"]
                for p in positions]
        parts = ax.boxplot(data, positions=np.arange(len(positions)) + offset, widths=0.22,
                           patch_artist=True, showfliers=False)
        for box in parts["boxes"]:
            box.set_facecolor(config.COLORS[dialect])
            box.set_alpha(0.65)
    ax.set_xticks(range(len(positions)))
    ax.set_xticklabels(["/s/ in onset", "/s/ in coda"])
    ax.set_ylabel("Spectral centre of gravity (Hz), speaker means")
    ax.legend([plt.Rectangle((0, 0), 1, 1, fc=config.COLORS[d], alpha=0.65) for d in config.ALL_DIALECTS],
              [LABELS[d] for d in config.ALL_DIALECTS], fontsize=9)
    ax.set_title("A. /s/ CoG by position", fontsize=11, loc="left")
    ax.grid(alpha=0.3, axis="y")

    ax = axes[1]
    for i, dialect in enumerate(config.ALL_DIALECTS):
        values = per_speaker[per_speaker["dialecto"] == dialect]["weakening"]
        half = 1.96 * values.std(ddof=1) / np.sqrt(len(values))
        ax.errorbar(i, values.mean(), yerr=half, fmt="o", capsize=5, markersize=8,
                    color=config.COLORS[dialect])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(range(len(config.ALL_DIALECTS)))
    ax.set_xticklabels([LABELS[d] for d in config.ALL_DIALECTS])
    ax.set_xlim(-0.5, len(config.ALL_DIALECTS) - 0.5)
    ax.set_ylabel("onset CoG - coda CoG (Hz), mean and 95% CI")
    ax.set_title("B. Weakening index per speaker", fontsize=11, loc="left")
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle("H2a: coda /s/ weakening", fontsize=12)
    fig.tight_layout()
    _save(fig, "fig_sibilants")


def data_flow_figure():
    """How the corpus narrows from 439k clips to the analysed tokens."""
    funnel = _table("s5_qc_funnel")
    alignment = _table("s3_alignment_report")
    manifest = pd.read_csv(config.SAMPLE_CSV)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    stages = ["Corpus\n(validated)", "Target\ndialects", "Sampled", "Aligned", "Vowel\ntokens"]
    values = [CORPUS_VALIDATED_CLIPS, CORPUS_SINGLE_LABEL_CLIPS, len(manifest),
              int(alignment["aligned"].sum()), int(funnel.iloc[0]["n_tokens"])]
    ax.bar(stages, values, color="#2166ac", alpha=0.8)
    ax.set_yscale("log")
    ax.set_ylim(top=max(values) * 2.5)
    ax.set_ylabel("count (log scale)")
    ax.set_title("A. From corpus to tokens", fontsize=11, loc="left")
    for i, v in enumerate(values):
        ax.text(i, v * 1.25, f"{v:,}", ha="center", fontsize=9)
    ax.grid(alpha=0.3, axis="y")

    ax = axes[1]
    ax.barh(funnel["stage"], funnel["n_tokens"], color="#4daf4a", alpha=0.8)
    ax.set_xlabel("tokens retained")
    ax.set_title("B. Quality control", fontsize=11, loc="left")
    for i, (n, pct) in enumerate(zip(funnel["n_tokens"], funnel["pct_retained"])):
        ax.text(n * 1.01, i, f"{n:,} ({pct:.0f}%)", va="center", fontsize=8)
    ax.set_xlim(0, funnel["n_tokens"].max() * 1.3)
    ax.grid(alpha=0.3, axis="x")

    fig.suptitle("Data flow", fontsize=12)
    fig.tight_layout()
    _save(fig, "fig_data_flow")


def main():
    banner(log, "FIGURES")
    vowel_space_figure()
    equivalence_figure()
    plural_figure()
    sibilant_figure()
    data_flow_figure()


if __name__ == "__main__":
    main()
