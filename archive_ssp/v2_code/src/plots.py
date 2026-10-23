"""
Report figures that do not belong to a single stage.

All of them use speaker-level values. Plotting token clouds would separate the
two dialects nicely, but only because most Southern tokens come from one person.
"""

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy import stats

from . import config
from .utils import banner, bootstrap_ci_by_speaker, cohens_d, get_logger

log = get_logger(__name__)

# los totales del corpus para la figura del flujo de datos. El stage 1 solo los
# saca por pantalla y no los guarda en ninguna tabla, y volver a lanzarlo
# rehace la muestra, asi que los apunto aqui. Son los del run sobre Common Voice
# es 26.0, los mismos que estan en RESULTS.md
CORPUS_VALIDATED_CLIPS = 439_225
CORPUS_SINGLE_LABEL_CLIPS = 236_832


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
    ax.add_patch(Ellipse((np.mean(x), np.mean(y)), width, height,
                         angle=angle, **kwargs))


def vowel_space_figure(tokens):
    """Vowel space from speaker means, with 95% ellipses across speakers."""
    usable = tokens[tokens["lobanov_ok"]
                    & tokens["dialecto"].isin(config.CORE_DIALECTS)]
    speaker_means = (
        usable.groupby(["client_id", "dialecto", "vowel"])
        [["F1_50_lob", "F2_50_lob", "F1_50", "F2_50"]]
        .mean().reset_index()
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, (f1, f2, label) in zip(axes, [
        ("F1_50", "F2_50", "Raw Hz (confounded with vocal tract length)"),
        ("F1_50_lob", "F2_50_lob", "Lobanov z-scores (speaker-normalised)"),
    ]):
        for dialect in config.CORE_DIALECTS:
            for vowel in config.VOWELS:
                sub = speaker_means[
                    (speaker_means["dialecto"] == dialect)
                    & (speaker_means["vowel"] == vowel)
                ].dropna(subset=[f1, f2])
                if sub.empty:
                    continue
                style = dict(color=config.COLORS[dialect])
                ax.scatter(sub[f2], sub[f1], s=14, alpha=0.35, **style)
                confidence_ellipse(sub[f2].to_numpy(), sub[f1].to_numpy(), ax,
                                   facecolor="none", edgecolor=config.COLORS[dialect],
                                   linewidth=1.6,
                                   linestyle="-" if dialect == "norte" else "--")
                ax.scatter(sub[f2].mean(), sub[f1].mean(), s=120, marker="o",
                           edgecolors="black", zorder=5, **style)
                ax.annotate(vowel, (sub[f2].mean(), sub[f1].mean()),
                            fontsize=13, fontweight="bold", ha="center", va="center",
                            color="white", zorder=6)

        ax.invert_xaxis()
        ax.invert_yaxis()
        ax.set_xlabel("F2" + (" (Hz)" if "lob" not in f1 else " (z)"))
        ax.set_ylabel("F1" + (" (Hz)" if "lob" not in f1 else " (z)"))
        ax.set_title(label, fontsize=11, loc="left")
        ax.grid(alpha=0.3)

    handles = [
        plt.Line2D([], [], color=config.COLORS["norte"], marker="o", linestyle="-",
                   label=f"Norte (n={speaker_means[speaker_means['dialecto']=='norte']['client_id'].nunique()} speakers)"),
        plt.Line2D([], [], color=config.COLORS["sur"], marker="o", linestyle="--",
                   label=f"Sur (n={speaker_means[speaker_means['dialecto']=='sur']['client_id'].nunique()} speakers)"),
    ]
    axes[0].legend(handles=handles, fontsize=9, loc="lower left")

    fig.suptitle(
        "Vowel space by dialect - each point is one speaker's mean, ellipses are 95% across speakers",
        fontsize=12,
    )
    plt.tight_layout()
    path = config.FIGURE_DIR / "fig_vowel_space.png"
    plt.savefig(path, dpi=config.DPI)
    plt.close()
    log.info(f"  -> {path.relative_to(config.ROOT)}")


def outgroup_figure(tokens):
    """
    All three varieties in one vowel space, with the effect sizes per contrast.

    Whatever separates Iberian from Mexican Spanish is what a real dialect
    difference looks like here, and Norte/Sur can be read against it.
    """
    usable = tokens[tokens["lobanov_ok"]]
    present = [d for d in config.ALL_DIALECTS if d in set(usable["dialecto"])]
    if len(present) < 3:
        log.info("  (skipping outgroup figure: México not in this sample)")
        return

    speaker_means = (
        usable.groupby(["client_id", "dialecto", "vowel"])[["F1_50_lob", "F2_50_lob"]]
        .mean().reset_index()
    )

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    ax = axes[0]
    for dialect in present:
        for vowel in config.VOWELS:
            sub = speaker_means[(speaker_means["dialecto"] == dialect)
                                & (speaker_means["vowel"] == vowel)]
            sub = sub.dropna(subset=["F1_50_lob", "F2_50_lob"])
            if sub.empty:
                continue
            confidence_ellipse(sub["F2_50_lob"].to_numpy(), sub["F1_50_lob"].to_numpy(),
                               ax, facecolor=config.COLORS[dialect], alpha=0.13)
            ax.scatter(sub["F2_50_lob"].mean(), sub["F1_50_lob"].mean(),
                       s=130, color=config.COLORS[dialect], edgecolors="black", zorder=5)
            ax.annotate(vowel, (sub["F2_50_lob"].mean(), sub["F1_50_lob"].mean()),
                        fontsize=12, fontweight="bold", ha="center", va="center",
                        color="white", zorder=6)
    ax.invert_xaxis(); ax.invert_yaxis()
    ax.set_xlabel("F2 (Lobanov z)"); ax.set_ylabel("F1 (Lobanov z)")
    ax.set_title("A. Vowel space, three varieties", fontsize=11, loc="left")
    ax.legend(handles=[plt.Line2D([], [], color=config.COLORS[d], marker="o",
                                  linestyle="", label=d.capitalize())
                       for d in present], fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[1]
    path = config.TABLE_DIR / "s6_outgroup_calibration.csv"
    if path.exists():
        calib = pd.read_csv(path)
        summary = calib.groupby("pair")["abs_d"].mean()
        order = ["norte vs sur", "norte vs mexico", "sur vs mexico"]
        summary = summary.reindex([p for p in order if p in summary.index])
        colours = ["#2166ac"] + ["#999999"] * (len(summary) - 1)
        ax.barh(range(len(summary)), summary.to_numpy(), color=colours, alpha=0.85)
        ax.set_yticks(range(len(summary)))
        ax.set_yticklabels(summary.index)
        ax.set_xlabel("Mean |Cohen's d| across vowel measures (speaker level)")
        ax.axvline(0.2, color="red", linestyle=":", label="small effect (0.2)")
        ax.axvline(0.5, color="red", linestyle="--", label="medium effect (0.5)")
        ax.set_title("B. How big is a real dialect difference here?",
                     fontsize=11, loc="left")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, axis="x")

    fig.suptitle("México as an outgroup: calibrating the Norte/Sur null result", fontsize=12)
    plt.tight_layout()
    out_path = config.FIGURE_DIR / "fig_outgroup.png"
    plt.savefig(out_path, dpi=config.DPI)
    plt.close()
    log.info(f"  -> {out_path.relative_to(config.ROOT)}")


def effects_forest_figure(tokens):
    """Speaker-level effect sizes with speaker-resampled intervals."""
    usable = tokens[tokens["lobanov_ok"]
                    & tokens["dialecto"].isin(config.CORE_DIALECTS)]

    speakers = usable.groupby(["client_id", "dialecto"]).agg(
        F1_a=("F1_50_lob", "mean"),
        duration=("duration", "mean"),
        B2=("B2_50", "mean"),
        hnr=("hnr_db", "mean"),
    ).reset_index()

    per_vowel = (
        usable.groupby(["client_id", "dialecto", "vowel"])["F1_50_lob"]
        .mean().unstack()
    )
    per_vowel.columns = [f"F1 /{v}/" for v in per_vowel.columns]
    per_vowel = per_vowel.reset_index()

    merged = speakers.merge(per_vowel, on=["client_id", "dialecto"])
    measures = [c for c in merged.columns if c.startswith("F1 /")] + \
               ["duration", "B2", "hnr"]

    rows = []
    for measure in measures:
        data = merged.dropna(subset=[measure])
        a = data[data["dialecto"] == "norte"][measure]
        b = data[data["dialecto"] == "sur"][measure]
        if len(a) < 3 or len(b) < 3:
            continue
        lo, hi = bootstrap_ci_by_speaker(data, measure, "dialecto", "client_id",
                                         ("norte", "sur"))
        rows.append({"measure": measure, "d": cohens_d(a, b), "lo": lo, "hi": hi})

    effects = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(8, 6))
    y = np.arange(len(effects))
    colours = ["#d6604d" if (lo > 0 or hi < 0) else "#2166ac"
               for lo, hi in zip(effects["lo"], effects["hi"])]
    ax.hlines(y, effects["lo"], effects["hi"], color=colours, linewidth=4, alpha=0.75)
    ax.scatter(effects["d"], y, color="black", zorder=5, s=36)
    ax.axvline(0, color="red", linestyle="--", alpha=0.8)
    ax.axvspan(-0.2, 0.2, color="gray", alpha=0.12)
    ax.set_yticks(y)
    ax.set_yticklabels(effects["measure"])
    ax.set_xlabel("Cohen's d (Norte - Sur), 95% CI from resampling speakers")
    ax.set_title(
        "Speaker-level dialect effects: every interval includes zero",
        fontsize=12, loc="left",
    )
    ax.grid(alpha=0.3, axis="x")
    plt.tight_layout()
    path = config.FIGURE_DIR / "fig_effects_forest.png"
    plt.savefig(path, dpi=config.DPI)
    plt.close()
    log.info(f"  -> {path.relative_to(config.ROOT)}")


def data_flow_figure():
    """How the corpus narrows from 439k clips to the analysed sample."""
    funnel = pd.read_csv(config.TABLE_DIR / "s5_qc_funnel.csv")
    manifest = pd.read_csv(config.SAMPLE_CSV)
    alignment = pd.read_csv(config.TABLE_DIR / "s3_alignment_report.csv")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    stages = ["Corpus\n(validated)", "Target\ndialects", "Sampled", "Aligned", "Vowel\ntokens"]
    values = [
        CORPUS_VALIDATED_CLIPS,
        CORPUS_SINGLE_LABEL_CLIPS,
        len(manifest),
        int(alignment["aligned"].sum()),
        int(funnel.iloc[0]["n_tokens"]),
    ]
    ax.bar(stages, values, color="#2166ac", alpha=0.8)
    ax.set_yscale("log")
    ax.set_ylim(top=max(values) * 2.5)   # headroom so the top label clears the title
    ax.set_ylabel("count (log scale)")
    ax.set_title("A. From corpus to tokens", fontsize=11, loc="left")
    for i, v in enumerate(values):
        ax.text(i, v * 1.25, f"{v:,}", ha="center", fontsize=9)
    ax.grid(alpha=0.3, axis="y")

    ax = axes[1]
    ax.barh(funnel["stage"], funnel["n_tokens"], color="#4daf4a", alpha=0.8)
    ax.set_xlabel("tokens retained")
    ax.set_title("B. Quality control funnel", fontsize=11, loc="left")
    for i, (n, pct) in enumerate(zip(funnel["n_tokens"], funnel["pct_retained"])):
        ax.text(n * 1.01, i, f"{n:,} ({pct:.0f}%)", va="center", fontsize=8)
    ax.set_xlim(0, funnel["n_tokens"].max() * 1.25)
    ax.grid(alpha=0.3, axis="x")

    fig.suptitle("Data flow", fontsize=12)
    plt.tight_layout()
    path = config.FIGURE_DIR / "fig_data_flow.png"
    plt.savefig(path, dpi=config.DPI)
    plt.close()
    log.info(f"  -> {path.relative_to(config.ROOT)}")


def main():
    banner(log, "REPORT FIGURES")
    tokens = pd.read_csv(config.TOKENS_CSV)
    vowel_space_figure(tokens)
    outgroup_figure(tokens)
    effects_forest_figure(tokens)
    data_flow_figure()


if __name__ == "__main__":
    main()
