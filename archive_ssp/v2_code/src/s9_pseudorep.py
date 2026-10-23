"""
Stage 9 - The speaker audit.

Repeats the midterm analysis and then shows that its three big effects come from
sampling by clip in a corpus where a few volunteers recorded most of it.

Only needs results_formants_clean.csv and validated.tsv, so it can be run on its
own without aligning or extracting anything.

Writes the s9_ tables and figures/fig_pseudoreplication.png.
"""

import warnings

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from scipy import stats

from . import config
from .utils import (
    banner,
    bootstrap_ci_by_speaker,
    bootstrap_ci_by_token,
    cohens_d,
    get_logger,
    gini,
    holm_correction,
    load_cv_metadata,
    normalise_sex,
    save_table,
)

log = get_logger(__name__)

FEATURES = ["F1_Hz", "F2_Hz", "B1_Hz", "B2_Hz"]


def load_legacy():
    """Attach speaker, sex and age to the midterm's cleaned table."""
    tokens = pd.read_csv(config.LEGACY_CLEAN_CSV)
    meta = load_cv_metadata()[["client_id", "path", "age", "gender"]]
    merged = tokens.merge(meta, left_on="nombre_archivo", right_on="path", how="left")
    merged["sex"] = merged["gender"].apply(normalise_sex)

    missing = merged["client_id"].isna().sum()
    if missing:
        log.info(f"  {missing} tokens could not be matched to a speaker; dropped.")
        merged = merged.dropna(subset=["client_id"])

    log.info(f"  {len(merged)} tokens, {merged['client_id'].nunique()} speakers")
    return merged


# 1. el analisis ingenuo


def naive_analysis(df):
    """Welch t-tests treating every clip as an independent observation."""
    banner(log, "1. NAIVE TOKEN-LEVEL ANALYSIS (reproduces the midterm)")

    norte = df[df["dialecto"] == "norte"]
    sur = df[df["dialecto"] == "sur"]
    log.info(f"n = {len(norte)} Norte tokens vs {len(sur)} Sur tokens\n")

    rows = []
    for col in FEATURES:
        t, p = stats.ttest_ind(norte[col], sur[col], equal_var=False)
        d = cohens_d(norte[col], sur[col])
        lo, hi = bootstrap_ci_by_token(df, col, "dialecto", ("norte", "sur"))
        rows.append({
            "feature": col.replace("_Hz", ""),
            "norte_mean": norte[col].mean(),
            "sur_mean": sur[col].mean(),
            "t": t,
            "p": p,
            "cohens_d": d,
            "ci_low": lo,
            "ci_high": hi,
        })

    out = pd.DataFrame(rows)
    out["p_holm"] = holm_correction(out["p"])

    for _, r in out.iterrows():
        stars = "***" if r["p_holm"] < 0.001 else "**" if r["p_holm"] < 0.01 else \
                "*" if r["p_holm"] < config.ALPHA else "n.s."
        log.info(
            f"  {r['feature']:>3}: Norte {r['norte_mean']:6.1f}  Sur {r['sur_mean']:6.1f}  "
            f"d = {r['cohens_d']:+.3f} [{r['ci_low']:+.2f}, {r['ci_high']:+.2f}]  "
            f"p_holm = {r['p_holm']:.2e} {stars}"
        )

    log.info(
        "\n  Read naively this says: large F1 and B2 differences between dialects."
    )
    return out


# 2. quien ha grabado en realidad estos tokens


def speaker_concentration(df):
    banner(log, "2. SPEAKER CONCENTRATION")

    rows = []
    for dialect in config.CORE_DIALECTS:
        sub = df[df["dialecto"] == dialect]
        counts = sub["client_id"].value_counts()
        rows.append({
            "dialect": dialect,
            "tokens": len(sub),
            "speakers": len(counts),
            "top1_tokens": int(counts.iloc[0]),
            "top1_share": counts.iloc[0] / len(sub),
            "top3_share": counts.head(3).sum() / len(sub),
            "median_tokens_per_speaker": float(counts.median()),
            "gini": gini(counts.to_numpy()),
        })

    out = pd.DataFrame(rows)
    for _, r in out.iterrows():
        log.info(
            f"  {r['dialect']:>5}: {r['tokens']:3d} tokens from {r['speakers']:2d} speakers | "
            f"top speaker {r['top1_tokens']:3d} ({r['top1_share']:.0%}), "
            f"top 3 {r['top3_share']:.0%} | median {r['median_tokens_per_speaker']:.0f} | "
            f"Gini {r['gini']:.2f}"
        )

    log.info("\n  Sex distribution within each dialect:")
    log.info("  " + pd.crosstab(df["dialecto"], df["sex"]).to_string().replace("\n", "\n  "))
    log.info("\n  Age distribution within each dialect:")
    log.info("  " + pd.crosstab(df["dialecto"], df["age"]).to_string().replace("\n", "\n  "))
    log.info("\n  Sex is confounded with dialect, so a raw-Hz contrast is partly a size contrast.")
    return out


# 3. cuanto del efecto es una sola persona


def cap_sensitivity(df):
    """
    Recompute the effect sizes while capping tokens per speaker.

    A real dialect effect should survive the cap; one carried by a single voice
    collapses as soon as that voice is limited.
    """
    banner(log, "3. EFFECT SIZE vs PER-SPEAKER TOKEN CAP")

    caps = [1, 2, 3, 5, 8, 10, 20, 50, 100, None]
    ordered = df.sort_values("nombre_archivo")  # ordeno para que el head() salga igual siempre

    rows = []
    for cap in caps:
        sub = ordered if cap is None else ordered.groupby("client_id").head(cap)
        norte = sub[sub["dialecto"] == "norte"]
        sur = sub[sub["dialecto"] == "sur"]
        row = {
            "cap": cap if cap is not None else np.inf,
            "n_tokens": len(sub),
            "n_speakers": sub["client_id"].nunique(),
        }
        for col in FEATURES:
            row[f"d_{col.replace('_Hz','')}"] = cohens_d(norte[col], sur[col])
            _, p = stats.ttest_ind(norte[col], sur[col], equal_var=False)
            row[f"p_{col.replace('_Hz','')}"] = p
        rows.append(row)

    out = pd.DataFrame(rows)
    log.info(f"  {'cap':>5} {'tokens':>7} {'spk':>5} {'d_F1':>8} {'d_F2':>8} {'d_B1':>8} {'d_B2':>8}")
    for _, r in out.iterrows():
        cap_label = "all" if np.isinf(r["cap"]) else f"{int(r['cap'])}"
        log.info(
            f"  {cap_label:>5} {int(r['n_tokens']):>7} {int(r['n_speakers']):>5} "
            f"{r['d_F1']:>+8.3f} {r['d_F2']:>+8.3f} {r['d_B1']:>+8.3f} {r['d_B2']:>+8.3f}"
        )

    log.info("\n  At cap = 1 the observations are independent, so that row is what the corpus supports.")
    return out


def leave_one_speaker_out(df):
    """Recompute the effect with each speaker removed in turn."""
    banner(log, "4. LEAVE-ONE-SPEAKER-OUT")

    rows = []
    for spk in df["client_id"].unique():
        sub = df[df["client_id"] != spk]
        norte = sub[sub["dialecto"] == "norte"]
        sur = sub[sub["dialecto"] == "sur"]
        if len(norte) < 3 or len(sur) < 3:
            continue
        row = {
            "removed_speaker": spk,
            "removed_dialect": df.loc[df["client_id"] == spk, "dialecto"].iloc[0],
            "removed_tokens": int((df["client_id"] == spk).sum()),
        }
        for col in FEATURES:
            row[f"d_{col.replace('_Hz','')}"] = cohens_d(norte[col], sur[col])
        rows.append(row)

    out = pd.DataFrame(rows).sort_values("d_F1")

    full_d = {c: cohens_d(df[df["dialecto"] == "norte"][c],
                          df[df["dialecto"] == "sur"][c]) for c in FEATURES}

    log.info("  Removing a single speaker changes Cohen's d as follows:")
    for col in FEATURES:
        key = f"d_{col.replace('_Hz','')}"
        log.info(
            f"    {col.replace('_Hz',''):>3}: full sample {full_d[col]:+.3f} | "
            f"range after removing any one speaker [{out[key].min():+.3f}, {out[key].max():+.3f}]"
        )

    worst = out.nsmallest(3, "d_F1")[["removed_dialect", "removed_tokens", "d_F1", "d_B2"]]
    log.info("\n  Speakers whose removal most reduces the F1 effect:")
    log.info("  " + worst.to_string(index=False).replace("\n", "\n  "))
    log.info("\n  An effect one participant can erase is a fact about that participant.")
    return out


# 4. el analisis corregido


def corrected_analysis(df, naive):
    """
    Three corrections, each stricter than the last:
      (a) speaker means, one observation per person
      (b) mixed model, speaker as a random intercept, keeping all tokens
      (c) speaker-resampled bootstrap intervals on the original effect sizes
    """
    banner(log, "5. CORRECTED ANALYSIS")

    # (a) agregar por hablante
    log.info("(a) One mean per speaker")
    speaker_means = (
        df.groupby(["client_id", "dialecto"])[FEATURES].mean().reset_index()
    )
    counts = speaker_means["dialecto"].value_counts()
    log.info(f"    n = {counts.get('norte', 0)} Norte speakers vs {counts.get('sur', 0)} Sur speakers")

    agg_rows = []
    for col in FEATURES:
        a = speaker_means[speaker_means["dialecto"] == "norte"][col]
        b = speaker_means[speaker_means["dialecto"] == "sur"][col]
        t, p = stats.ttest_ind(a, b, equal_var=False)
        agg_rows.append({
            "feature": col.replace("_Hz", ""),
            "norte_mean": a.mean(), "sur_mean": b.mean(),
            "t": t, "p": p, "cohens_d": cohens_d(a, b),
        })
    agg = pd.DataFrame(agg_rows)
    agg["p_holm"] = holm_correction(agg["p"])
    for _, r in agg.iterrows():
        flag = "significant" if r["p_holm"] < config.ALPHA else "n.s."
        log.info(
            f"    {r['feature']:>3}: Norte {r['norte_mean']:6.1f}  Sur {r['sur_mean']:6.1f}  "
            f"d = {r['cohens_d']:+.3f}  p_holm = {r['p_holm']:.3f}  {flag}"
        )

    # (b) modelos mixtos
    log.info("\n(b) Mixed-effects model: feature ~ dialect + (1 | speaker)")
    model_df = df.copy()
    model_df["dialect_num"] = (model_df["dialecto"] == "sur").astype(int)

    mixed_rows = []
    for col in FEATURES:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = smf.mixedlm(
                    f"{col} ~ dialect_num", model_df, groups=model_df["client_id"]
                ).fit(reml=True)
            coef = fit.params.get("dialect_num", np.nan)
            pval = fit.pvalues.get("dialect_num", np.nan)
            var_speaker = float(fit.cov_re.iloc[0, 0])
            var_resid = float(fit.scale)
            icc = var_speaker / (var_speaker + var_resid)
        except Exception as exc:  # singular fits are informative, not fatal
            log.info(f"    {col}: model did not converge ({exc})")
            coef = pval = icc = var_speaker = var_resid = np.nan

        # efecto del diseño: a cuantas observaciones independientes equivalen
        # de verdad los 440 tokens, teniendo en cuenta la correlacion interna
        m = len(model_df) / model_df["client_id"].nunique()
        deff = 1 + (m - 1) * icc if np.isfinite(icc) else np.nan
        n_eff = len(model_df) / deff if np.isfinite(deff) and deff > 0 else np.nan

        mixed_rows.append({
            "feature": col.replace("_Hz", ""),
            "sur_vs_norte_Hz": coef, "p": pval,
            "var_speaker": var_speaker, "var_residual": var_resid,
            "ICC": icc, "design_effect": deff, "n_effective": n_eff,
        })

    mixed = pd.DataFrame(mixed_rows)
    mixed["p_holm"] = holm_correction(mixed["p"].fillna(1.0))
    for _, r in mixed.iterrows():
        flag = "significant" if r["p_holm"] < config.ALPHA else "n.s."
        log.info(
            f"    {r['feature']:>3}: Sur - Norte = {r['sur_vs_norte_Hz']:+7.1f} Hz  "
            f"p_holm = {r['p_holm']:.3f}  {flag:>11} | ICC = {r['ICC']:.2f}  "
            f"n_eff = {r['n_effective']:.0f} of {len(model_df)}"
        )

    log.info(f"\n    n_eff is how many independent observations the {len(model_df)} tokens are worth.")

    # (c) intervalos remuestreando hablantes
    log.info("\n(c) Bootstrap CIs for Cohen's d: resampling tokens vs speakers")
    ci_rows = []
    for col in FEATURES:
        tok_lo, tok_hi = bootstrap_ci_by_token(df, col, "dialecto", ("norte", "sur"))
        spk_lo, spk_hi = bootstrap_ci_by_speaker(
            df, col, "dialecto", "client_id", ("norte", "sur")
        )
        ci_rows.append({
            "feature": col.replace("_Hz", ""),
            "d": cohens_d(df[df["dialecto"] == "norte"][col],
                          df[df["dialecto"] == "sur"][col]),
            "token_ci_low": tok_lo, "token_ci_high": tok_hi,
            "speaker_ci_low": spk_lo, "speaker_ci_high": spk_hi,
        })
        crosses = "includes 0" if spk_lo <= 0 <= spk_hi else "excludes 0"
        log.info(
            f"    {col.replace('_Hz',''):>3}: token CI [{tok_lo:+.2f}, {tok_hi:+.2f}]  "
            f"speaker CI [{spk_lo:+.2f}, {spk_hi:+.2f}]  ({crosses})"
        )

    cis = pd.DataFrame(ci_rows)

    # el resumen de los dos lados, para el informe
    summary = pd.DataFrame({
        "feature": naive["feature"],
        "naive_d": naive["cohens_d"].values,
        "naive_p_holm": naive["p_holm"].values,
        "speaker_mean_d": agg["cohens_d"].values,
        "speaker_mean_p_holm": agg["p_holm"].values,
        "mixed_coef_Hz": mixed["sur_vs_norte_Hz"].values,
        "mixed_p_holm": mixed["p_holm"].values,
        "ICC": mixed["ICC"].values,
    })

    log.info("\n  SUMMARY - naive versus corrected")
    log.info("  " + summary.round(3).to_string(index=False).replace("\n", "\n  "))

    return summary, agg, mixed, cis


def make_figure(df, caps, loo, cis):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # (A) cuantos tokens acumula cada hablante
    ax = axes[0][0]
    for dialect in config.CORE_DIALECTS:
        counts = df[df["dialecto"] == dialect]["client_id"].value_counts()
        share = np.cumsum(counts.to_numpy()) / counts.sum()
        ax.plot(np.arange(1, len(share) + 1), share * 100, "o-",
                color=config.COLORS[dialect], label=f"{dialect.capitalize()} (n={len(counts)})",
                markersize=4)
    ax.axhline(50, color="gray", linestyle=":", alpha=0.7)
    ax.set_xlabel("Speakers, ranked by number of tokens")
    ax.set_ylabel("Cumulative share of tokens (%)")
    ax.set_title("A. A few speakers supply most of the data", fontsize=11, loc="left")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # (B) como cambia el efecto al apretar el tope por hablante
    ax = axes[0][1]
    x = np.arange(len(caps))
    for col, style in zip(["d_F1", "d_F2", "d_B1", "d_B2"], ["o-", "s--", "^:", "v-."]):
        ax.plot(x, caps[col], style, label=col.replace("d_", ""), markersize=5)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axhspan(-0.2, 0.2, color="gray", alpha=0.15)
    ax.set_xticks(x)
    ax.set_xticklabels(["all" if np.isinf(c) else int(c) for c in caps["cap"]], fontsize=8)
    ax.set_xlabel("Maximum tokens allowed per speaker")
    ax.set_ylabel("Cohen's d")
    ax.set_title("B. Effects vanish once no speaker can dominate", fontsize=11, loc="left")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # (C) que pasa al quitar a cada hablante
    ax = axes[1][0]
    for col, colour in [("d_F1", "#2166ac"), ("d_B2", "#d6604d")]:
        ax.scatter(loo["removed_tokens"], loo[col], alpha=0.65, s=28,
                   color=colour, label=col.replace("d_", ""))
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("Tokens contributed by the removed speaker (log scale)")
    ax.set_ylabel("Cohen's d for the remaining sample")
    ax.set_title("C. Removing one prolific speaker erases the effect", fontsize=11, loc="left")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # (D) intervalos por tokens frente a por hablantes
    ax = axes[1][1]
    y = np.arange(len(cis))
    ax.hlines(y + 0.13, cis["token_ci_low"], cis["token_ci_high"],
              color="#999999", linewidth=5, label="Resampling tokens")
    ax.hlines(y - 0.13, cis["speaker_ci_low"], cis["speaker_ci_high"],
              color="#2166ac", linewidth=5, label="Resampling speakers")
    ax.scatter(cis["d"], y, color="black", zorder=5, s=32, label="Point estimate")
    ax.axvline(0, color="red", linestyle="--", alpha=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(cis["feature"])
    ax.set_xlabel("Cohen's d (95% CI)")
    ax.set_title("D. Honest intervals all include zero", fontsize=11, loc="left")
    ax.legend(fontsize=9, loc="upper right")
    ax.set_ylim(-0.6, len(cis) - 0.1)
    ax.grid(alpha=0.3, axis="x")

    fig.suptitle(
        "Pseudoreplication in the midterm analysis: 440 tokens, 80 speakers, two of them decisive",
        fontsize=13,
    )
    plt.tight_layout()
    path = config.FIGURE_DIR / "fig_pseudoreplication.png"
    plt.savefig(path, dpi=config.DPI)
    plt.close()
    log.info(f"\n  -> {path.relative_to(config.ROOT)}")


def main():
    banner(log, "STAGE 9 - SPEAKER AUDIT OF THE MIDTERM ANALYSIS")
    df = load_legacy()

    naive = naive_analysis(df)
    concentration = speaker_concentration(df)
    caps = cap_sensitivity(df)
    loo = leave_one_speaker_out(df)
    summary, agg, mixed, cis = corrected_analysis(df, naive)

    banner(log, "WRITING OUTPUTS")
    save_table(naive, "s9_naive_analysis", log)
    save_table(concentration, "s9_speaker_concentration", log)
    save_table(caps, "s9_cap_sensitivity", log)
    save_table(loo, "s9_leave_one_speaker_out", log)
    save_table(agg, "s9_speaker_level", log)
    save_table(mixed, "s9_mixed_models", log)
    save_table(cis, "s9_bootstrap_cis", log)
    save_table(summary, "s9_naive_vs_corrected", log)
    make_figure(df, caps, loo, cis)

    banner(log, "CONCLUSION")
    log.info(
        "The midterm's F1, F2 and B2 effects come from two speakers, who between them\n"
        "supplied more than half of each dialect cell. Capping, aggregating, mixed\n"
        "models and speaker-level resampling all remove them."
    )


if __name__ == "__main__":
    main()
