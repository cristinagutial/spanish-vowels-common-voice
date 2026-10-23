"""
Stage 6 - The corrected statistics.

Everything here uses the speaker as the unit, either with a random intercept per
speaker or with one row per speaker before testing. Both routes are reported.

    A  the naive analysis, run again on the new data
    B  speaker-level tests
    C  mixed models, dialect x vowel
    D  vowel space area and Pillai overlap
    E  bandwidth versus recording noise
    F  what effect size this sample could detect
    G  calibration against México

Writes the s6_ tables.
"""

import re
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from scipy.spatial import ConvexHull
from statsmodels.multivariate.manova import MANOVA
from statsmodels.stats.power import TTestIndPower

from . import config
from .utils import (
    banner,
    bootstrap_ci_by_speaker,
    cohens_d,
    get_logger,
    holm_correction,
    save_table,
)

log = get_logger(__name__)


def load_tokens():
    df = pd.read_csv(config.TOKENS_CSV)
    log.info(f"{len(df):,} tokens | {df['client_id'].nunique()} speakers | "
             f"{df['clip'].nunique()} clips")
    return df


# A. el analisis ingenuo, a ver si se repite el artefacto


def section_a(df):
    banner(log, "A. NAIVE POOLED ANALYSIS ON THE NEW DATA")

    dominant = df[df["dialecto"] == "sur"]["client_id"].value_counts().index[0]
    dominant_share = (df[df["dialecto"] == "sur"]["client_id"] == dominant).mean()
    log.info(f"Most prolific Sur speaker holds {dominant_share:.1%} of that cell.\n")

    rows = []
    for vowel in config.VOWELS:
        sub = df[df["vowel"] == vowel]
        norte = sub[sub["dialecto"] == "norte"]["F1_50"]
        sur = sub[sub["dialecto"] == "sur"]["F1_50"]
        rows.append({
            "vowel": vowel,
            "norte_F1": norte.mean(), "sur_F1": sur.mean(),
            "difference": norte.mean() - sur.mean(),
            "cohens_d": cohens_d(norte, sur),
            "p": stats.ttest_ind(norte, sur, equal_var=False)[1],
        })
    naive = pd.DataFrame(rows)

    # la misma comparacion pero quitando al hablante del Sur que mas clips tiene
    trimmed = df[df["client_id"] != dominant]
    rows = []
    for vowel in config.VOWELS:
        sub = trimmed[trimmed["vowel"] == vowel]
        norte = sub[sub["dialecto"] == "norte"]["F1_50"]
        sur = sub[sub["dialecto"] == "sur"]["F1_50"]
        rows.append({
            "vowel": vowel,
            "norte_F1": norte.mean(), "sur_F1": sur.mean(),
            "difference": norte.mean() - sur.mean(),
            "cohens_d": cohens_d(norte, sur),
        })
    trimmed_df = pd.DataFrame(rows)

    comparison = naive[["vowel", "norte_F1", "sur_F1", "difference", "cohens_d"]].merge(
        trimmed_df, on="vowel", suffixes=("_pooled", "_minus_one_speaker")
    )
    log.info(comparison.round(1).to_string(index=False))

    shift = (comparison["cohens_d_pooled"] - comparison["cohens_d_minus_one_speaker"]).abs().max()
    n_dominant = int((df["client_id"] == dominant).sum())
    log.info(
        f"\nRemoving that one speaker ({n_dominant:,} tokens, {dominant_share:.0%} of the "
        f"Sur cell) shifts the largest per-vowel Cohen's d by {shift:.2f}."
    )
    if shift > 0.3:
        log.info("  One speaker moves the result, so this is not a property of the dialect.")
    else:
        log.info("  Pooled and trimmed agree: the cap leaves nobody able to dominate a cell.")
    return comparison


# B. analisis por hablante


def speaker_table(df):
    """Reduce each speaker to one row: their mean per vowel plus global measures."""
    usable = df[df["lobanov_ok"]].copy()

    per_vowel = (
        usable.groupby(["client_id", "dialecto", "sex", "vowel"])
        [["F1_50", "F2_50", "F1_50_lob", "F2_50_lob", "duration"]]
        .mean().reset_index()
    )

    wide = per_vowel.pivot_table(
        index=["client_id", "dialecto", "sex"],
        columns="vowel",
        values=["F1_50_lob", "F2_50_lob"],
    )
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    wide = wide.reset_index()

    globals_ = usable.groupby("client_id").agg(
        n_tokens=("vowel", "size"),
        mean_duration=("duration", "mean"),
        mean_B1=("B1_50", "mean"),
        mean_B2=("B2_50", "mean"),
        mean_hnr=("hnr_db", "mean"),
        mean_snr=("snr_db", "mean"),
        mean_F0=("F0_Hz", "mean"),
    ).reset_index()

    return wide.merge(globals_, on="client_id"), per_vowel


def vowel_space_area(per_vowel):
    """
    Convex hull area of a speaker's five vowels in Lobanov space.

    One number per speaker saying how expanded or compressed their vowel system
    is. Vowel reduction, the usual claim about Andalusian, predicts a smaller area.
    """
    rows = []
    for (cid, dialect, sex), group in per_vowel.groupby(["client_id", "dialecto", "sex"]):
        points = group[["F2_50_lob", "F1_50_lob"]].dropna().to_numpy()
        if len(points) < 3:
            continue
        try:
            area = ConvexHull(points).volume  # en 2D, 'volume' es el area
        except Exception:
            continue
        rows.append({
            "client_id": cid, "dialecto": dialect, "sex": sex,
            "n_vowels": len(points), "vsa": area,
        })
    return pd.DataFrame(rows)


def section_b(df):
    banner(log, "B. SPEAKER-LEVEL ANALYSIS (one observation per speaker)")

    speakers, per_vowel = speaker_table(df)
    counts = speakers["dialecto"].value_counts()
    log.info(f"n = {counts.get('norte', 0)} Norte speakers vs {counts.get('sur', 0)} Sur speakers\n")

    measures = [c for c in speakers.columns
                if c.startswith(("F1_50_lob_", "F2_50_lob_"))] + \
               ["mean_duration", "mean_B1", "mean_B2", "mean_hnr", "mean_F0"]

    rows = []
    for col in measures:
        a = speakers[speakers["dialecto"] == "norte"][col].dropna()
        b = speakers[speakers["dialecto"] == "sur"][col].dropna()
        if len(a) < 3 or len(b) < 3:
            continue
        t, p = stats.ttest_ind(a, b, equal_var=False)
        u, p_u = stats.mannwhitneyu(a, b, alternative="two-sided")
        rows.append({
            "measure": col, "n_norte": len(a), "n_sur": len(b),
            "norte_mean": a.mean(), "sur_mean": b.mean(),
            "cohens_d": cohens_d(a, b), "t": t, "p": p, "p_mannwhitney": p_u,
        })

    out = pd.DataFrame(rows)
    out["p_holm"] = holm_correction(out["p"])

    log.info(f"{'measure':<16} {'Norte':>8} {'Sur':>8} {'d':>7} {'p':>8} {'p_holm':>8}")
    for _, r in out.iterrows():
        mark = " *" if r["p_holm"] < config.ALPHA else ""
        log.info(
            f"{r['measure']:<16} {r['norte_mean']:>8.2f} {r['sur_mean']:>8.2f} "
            f"{r['cohens_d']:>+7.2f} {r['p']:>8.3f} {r['p_holm']:>8.3f}{mark}"
        )

    # area del espacio vocalico
    vsa = vowel_space_area(per_vowel)
    a = vsa[vsa["dialecto"] == "norte"]["vsa"]
    b = vsa[vsa["dialecto"] == "sur"]["vsa"]
    t, p = stats.ttest_ind(a, b, equal_var=False)
    lo, hi = bootstrap_ci_by_speaker(
        vsa.assign(dummy=vsa["client_id"]), "vsa", "dialecto", "client_id", ("norte", "sur")
    )
    log.info(
        f"\nVowel space area (convex hull in Lobanov space):\n"
        f"  Norte {a.mean():.2f} (n={len(a)})  Sur {b.mean():.2f} (n={len(b)})  "
        f"d = {cohens_d(a, b):+.2f} [{lo:+.2f}, {hi:+.2f}]  p = {p:.3f}"
    )

    return out, speakers, vsa, per_vowel


# C. modelos mixtos


def fit_mixed(df, response, formula, label):
    """
    Fit one mixed model with a random intercept per speaker.

    Incomplete rows are dropped first: statsmodels' own missing='drop' does not
    work here because the grouping vector is passed separately.
    """
    referenced = [c for c in df.columns if re.search(rf"\b{re.escape(c)}\b", formula)]
    data = df.dropna(subset=[response, "client_id"] + referenced).copy()
    if data.empty:
        log.info(f"  {label}: no complete cases")
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = smf.mixedlm(f"{response} ~ {formula}", data,
                              groups=data["client_id"]).fit(reml=True)
    except Exception as exc:
        log.info(f"  {label}: did not converge ({exc})")
        return None

    var_speaker = float(fit.cov_re.iloc[0, 0])
    icc = var_speaker / (var_speaker + float(fit.scale))
    log.info(f"\n  {label}   (n = {len(data):,} tokens, {data['client_id'].nunique()} speakers, ICC = {icc:.2f})")

    for term in fit.params.index:
        if term in ("Intercept", "Group Var") or term.startswith("C(vowel)"):
            continue
        if "dialecto" in term or "sex" in term:
            p = fit.pvalues[term]
            mark = " *" if p < config.ALPHA else ""
            log.info(f"    {term:<44} {fit.params[term]:>+9.3f}  p = {p:.3f}{mark}")
    return fit


def section_c(df):
    banner(log, "C. MIXED-EFFECTS MODELS")
    log.info("Random intercept per speaker.")

    usable = df[df["lobanov_ok"]].copy()
    results = []

    for response, formula, label in [
        ("F1_50",     "C(dialecto) * C(vowel) + C(sex)", "F1 raw Hz"),
        ("F2_50",     "C(dialecto) * C(vowel) + C(sex)", "F2 raw Hz"),
        ("F1_50_lob", "C(dialecto) * C(vowel)",          "F1 Lobanov"),
        ("F2_50_lob", "C(dialecto) * C(vowel)",          "F2 Lobanov"),
        ("duration",  "C(dialecto) * C(vowel) + stressed", "Duration"),
        ("B2_50",     "C(dialecto) * C(vowel) + C(sex)", "B2 raw Hz"),
    ]:
        fit = fit_mixed(usable, response, formula, label)
        if fit is None:
            continue
        term = "C(dialecto)[T.sur]"
        if term in fit.params.index:
            var_speaker = float(fit.cov_re.iloc[0, 0])
            results.append({
                "response": label,
                "sur_vs_norte": fit.params[term],
                "std_error": fit.bse[term],
                "p": fit.pvalues[term],
                "ICC": var_speaker / (var_speaker + float(fit.scale)),
                "n_tokens": len(usable.dropna(subset=[response])),
            })

    out = pd.DataFrame(results)
    if len(out):
        out["p_holm"] = holm_correction(out["p"])
        log.info("\n  Main dialect effect across models (Holm-corrected):")
        for _, r in out.iterrows():
            verdict = "significant" if r["p_holm"] < config.ALPHA else "n.s."
            log.info(f"    {r['response']:<12} {r['sur_vs_norte']:>+9.3f} "
                     f"(SE {r['std_error']:.3f})  p_holm = {r['p_holm']:.3f}  {verdict}")

    log.info(
        "\n  Careful: the ICC = 0 above is arithmetic, not independence. Lobanov\n"
        "  already sets each speaker's mean to 0 and SD to 1. C2 aggregates first."
    )
    return out


def section_c2(df):
    """
    Dialect x vowel on speaker-by-vowel means.

    A random intercept only protects against speakers sitting at different levels,
    and Lobanov has already removed those. Collapsing to one value per speaker and
    vowel makes everyone contribute one observation per cell, whatever their clip
    count.
    """
    banner(log, "C2. DIALECT x VOWEL ON SPEAKER MEANS (the model to trust)")

    usable = df[df["lobanov_ok"]]
    agg = (
        usable.groupby(["client_id", "dialecto", "sex", "vowel"])
        .agg(F1_lob=("F1_50_lob", "mean"),
             F2_lob=("F2_50_lob", "mean"),
             F1_hz=("F1_50", "mean"),
             F2_hz=("F2_50", "mean"),
             duration=("duration", "mean"),
             B2_hz=("B2_50", "mean"),
             n=("vowel", "size"))
        .reset_index()
    )
    # la media de una vocal solo vale si detras hay unos cuantos tokens
    agg = agg[agg["n"] >= 3]
    log.info(f"{len(agg)} speaker-vowel cells from {agg['client_id'].nunique()} speakers "
             f"(min 3 tokens per cell)\n")

    results = []
    for response, label in [
        ("F1_lob", "F1 Lobanov"), ("F2_lob", "F2 Lobanov"),
        ("F1_hz", "F1 raw Hz"), ("F2_hz", "F2 raw Hz"),
        ("duration", "Duration"), ("B2_hz", "B2 raw Hz"),
    ]:
        formula = "C(dialecto) * C(vowel)"
        if response in ("F1_hz", "F2_hz", "B2_hz"):
            formula += " + C(sex)"
        fit = fit_mixed(agg, response, formula, label)
        if fit is None:
            continue
        term = "C(dialecto)[T.sur]"
        if term in fit.params.index:
            results.append({
                "response": label,
                "sur_vs_norte": fit.params[term],
                "std_error": fit.bse[term],
                "p": fit.pvalues[term],
                "n_cells": len(agg.dropna(subset=[response])),
            })

    out = pd.DataFrame(results)
    if len(out):
        out["p_holm"] = holm_correction(out["p"])
        log.info("\n  Main dialect effect (Holm-corrected):")
        for _, r in out.iterrows():
            verdict = "significant" if r["p_holm"] < config.ALPHA else "n.s."
            log.info(f"    {r['response']:<12} {r['sur_vs_norte']:>+9.3f} "
                     f"(SE {r['std_error']:.3f})  p_holm = {r['p_holm']:.3f}  {verdict}")
    return out, agg


# D. cuanto se solapan los dos espacios vocalicos


def section_d(df):
    banner(log, "D. PILLAI SCORES - how much do the two vowel spaces overlap?")
    log.info("0 means the two distributions coincide, 1 means no overlap.\n")

    usable = df[df["lobanov_ok"]]
    rows = []
    for vowel in config.VOWELS:
        sub = usable[usable["vowel"] == vowel].dropna(subset=["F1_50_lob", "F2_50_lob"])
        if sub["dialecto"].nunique() < 2 or len(sub) < 20:
            continue
        try:
            manova = MANOVA.from_formula("F1_50_lob + F2_50_lob ~ C(dialecto)", data=sub)
            table = manova.mv_test().results["C(dialecto)"]["stat"]
            pillai = table.loc["Pillai's trace", "Value"]
            pvalue = table.loc["Pillai's trace", "Pr > F"]
        except Exception:
            continue
        rows.append({"vowel": vowel, "n": len(sub), "pillai": pillai, "p": pvalue})

    out = pd.DataFrame(rows)
    if len(out):
        for _, r in out.iterrows():
            log.info(f"  /{r['vowel']}/  Pillai = {r['pillai']:.3f}  (n = {int(r['n']):,})")
        log.info("\n  This pools tokens, so it is descriptive only.")
    return out


# E. el efecto de los bandwidths, ¿es acustico o es ruido de la grabacion?


def section_e(df, speakers):
    banner(log, "E. IS THE BANDWIDTH EFFECT ACOUSTIC OR ARTIFACTUAL?")
    log.info("Bandwidths widen with noise, and one speaker means one microphone.\n")

    usable = df[df["lobanov_ok"]].dropna(subset=["B2_50", "hnr_db"])

    r_hnr, p_hnr = stats.pearsonr(usable["hnr_db"], usable["B2_50"])
    r_snr, p_snr = stats.pearsonr(usable["snr_db"].fillna(usable["snr_db"].mean()),
                                  usable["B2_50"])
    log.info(f"  Token-level correlation  B2 ~ HNR: r = {r_hnr:+.3f} (p = {p_hnr:.2e})")
    log.info(f"  Token-level correlation  B2 ~ SNR: r = {r_snr:+.3f} (p = {p_snr:.2e})")

    a = speakers[speakers["dialecto"] == "norte"]
    b = speakers[speakers["dialecto"] == "sur"]
    for measure in ["mean_hnr", "mean_snr"]:
        t, p = stats.ttest_ind(a[measure].dropna(), b[measure].dropna(), equal_var=False)
        log.info(f"  Speaker-level {measure}: Norte {a[measure].mean():.1f} vs "
                 f"Sur {b[measure].mean():.1f}  d = {cohens_d(a[measure].dropna(), b[measure].dropna()):+.2f}  p = {p:.3f}")

    log.info("\n  B2 model with and without a noise covariate:")
    rows = []
    for formula, label in [
        ("C(dialecto) + C(vowel) + C(sex)", "without HNR"),
        ("C(dialecto) + C(vowel) + C(sex) + hnr_db", "with HNR"),
    ]:
        fit = fit_mixed(usable, "B2_50", formula, f"B2 {label}")
        if fit is not None and "C(dialecto)[T.sur]" in fit.params.index:
            rows.append({
                "model": label,
                "sur_vs_norte_Hz": fit.params["C(dialecto)[T.sur]"],
                "p": fit.pvalues["C(dialecto)[T.sur]"],
            })

    out = pd.DataFrame(rows)
    if len(out) == 2:
        before, after = out.iloc[0], out.iloc[1]
        shrink = 1 - abs(after["sur_vs_norte_Hz"]) / max(abs(before["sur_vs_norte_Hz"]), 1e-9)
        log.info(f"\n  Controlling for HNR changes the dialect coefficient by {shrink:+.0%}.")

        if before["p"] >= config.ALPHA:
            log.info("  There is no B2 dialect difference left once speakers are modelled.")
        elif shrink > 0.3:
            log.info("  Most of the effect goes away with the noise covariate.")
        else:
            log.info("  The effect survives the noise covariate.")
    return out


def section_g(df):
    """
    Outgroup calibration.

    A null result only means something if the method could have found a real
    difference. México is the control: if Norte/Sur is near zero and
    Iberian/Mexican is not, the null is about the dialects and not the method.
    """
    banner(log, "G. OUTGROUP CALIBRATION - can the method detect a known difference?")

    usable = df[df["lobanov_ok"]]
    present = [d for d in config.ALL_DIALECTS if d in set(usable["dialecto"])]
    if len(present) < 3:
        log.info("México not present in this sample; skipping outgroup calibration.")
        return pd.DataFrame()

    per_vowel = (
        usable.groupby(["client_id", "dialecto", "vowel"])
        [["F1_50_lob", "F2_50_lob"]].mean().reset_index()
    )
    wide = per_vowel.pivot_table(index=["client_id", "dialecto"], columns="vowel",
                                 values=["F1_50_lob", "F2_50_lob"])
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    wide = wide.reset_index()

    globals_ = usable.groupby("client_id").agg(
        duration=("duration", "mean"), B2=("B2_50", "mean")
    ).reset_index()
    speakers = wide.merge(globals_, on="client_id")

    counts = speakers["dialecto"].value_counts()
    log.info("Speakers per variety: " + ", ".join(f"{d}={counts.get(d, 0)}" for d in present) + "\n")

    measures = [c for c in speakers.columns if c.startswith(("F1_50_lob_", "F2_50_lob_"))] \
               + ["duration", "B2"]

    pairs = [("norte", "sur"), ("norte", "mexico"), ("sur", "mexico")]
    rows = []
    for left, right in pairs:
        for measure in measures:
            a = speakers[speakers["dialecto"] == left][measure].dropna()
            b = speakers[speakers["dialecto"] == right][measure].dropna()
            if len(a) < 5 or len(b) < 5:
                continue
            t, p = stats.ttest_ind(a, b, equal_var=False)
            rows.append({
                "pair": f"{left} vs {right}", "measure": measure,
                "n_left": len(a), "n_right": len(b),
                "cohens_d": cohens_d(a, b), "abs_d": abs(cohens_d(a, b)), "p": p,
            })

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["p_holm"] = holm_correction(out["p"])

    log.info("Mean |Cohen's d| across all vowel measures, by contrast:")
    summary = out.groupby("pair").agg(
        mean_abs_d=("abs_d", "mean"),
        max_abs_d=("abs_d", "max"),
        n_significant=("p_holm", lambda s: (s < config.ALPHA).sum()),
        n_tests=("p_holm", "size"),
    ).reindex([f"{l} vs {r}" for l, r in pairs])

    for pair, r in summary.iterrows():
        log.info(f"  {pair:<18} mean |d| = {r['mean_abs_d']:.3f}   max |d| = {r['max_abs_d']:.3f}   "
                 f"significant: {int(r['n_significant'])}/{int(r['n_tests'])}")

    log.info("\nLargest individual effects per contrast:")
    for pair in summary.index:
        top = out[out["pair"] == pair].nlargest(3, "abs_d")
        items = ", ".join(f"{r['measure']} d={r['cohens_d']:+.2f}" for _, r in top.iterrows())
        log.info(f"  {pair:<18} {items}")

    ns = summary.loc["norte vs sur", "mean_abs_d"]
    outgroup = summary.loc[["norte vs mexico", "sur vs mexico"], "mean_abs_d"].mean()
    log.info(
        f"\n  Norte/Sur averages |d| = {ns:.3f}; Iberian/Mexican averages {outgroup:.3f} "
        f"({outgroup / max(ns, 1e-9):.1f}x larger)."
    )
    return out


def section_f(speakers):
    banner(log, "F. WHAT COULD THIS SAMPLE HAVE DETECTED?")

    n_norte = (speakers["dialecto"] == "norte").sum()
    n_sur = (speakers["dialecto"] == "sur").sum()
    analysis = TTestIndPower()

    rows = []
    for power in (0.80, 0.90):
        d = analysis.solve_power(
            effect_size=None, nobs1=n_sur, ratio=n_norte / n_sur,
            alpha=config.ALPHA, power=power, alternative="two-sided",
        )
        rows.append({"power": power, "n_sur": n_sur, "n_norte": n_norte,
                     "detectable_d": d})
        log.info(f"  With {n_sur} Sur and {n_norte} Norte speakers, {power:.0%} power "
                 f"requires d >= {d:.2f}")

    log.info("\n  Anything smaller is beyond what this many speakers can resolve.")
    return pd.DataFrame(rows)


def main():
    banner(log, "STAGE 6 - CORRECTED STATISTICAL ANALYSIS")
    full = load_tokens()

    # las secciones A-F son la pregunta Norte/Sur y solo usan esos dos dialectos.
    # La G es donde entra Mexico como control
    df = full[full["dialecto"].isin(config.CORE_DIALECTS)].copy()
    if len(df) < len(full):
        log.info(f"  {len(df):,} tokens in Norte/Sur; "
                 f"{len(full) - len(df):,} outgroup tokens held for section G")

    naive = section_a(df)
    speaker_results, speakers, vsa, per_vowel = section_b(df)
    mixed = section_c(df)
    mixed_agg, agg_cells = section_c2(df)
    pillai = section_d(df)
    bandwidth = section_e(df, speakers)
    outgroup = section_g(full)
    power = section_f(speakers)

    banner(log, "WRITING OUTPUTS")
    save_table(naive, "s6_naive_reproduction", log)
    save_table(speaker_results, "s6_speaker_level", log)
    save_table(speakers, "s6_speaker_measures", log)
    save_table(vsa, "s6_vowel_space_area", log)
    save_table(per_vowel, "s6_speaker_vowel_means", log)
    save_table(mixed, "s6_mixed_models", log)
    save_table(mixed_agg, "s6_mixed_models_speaker_means", log)
    save_table(agg_cells, "s6_speaker_vowel_cells", log)
    save_table(pillai, "s6_pillai", log)
    save_table(bandwidth, "s6_bandwidth_noise", log)
    save_table(power, "s6_power", log)
    if len(outgroup):
        save_table(outgroup, "s6_outgroup_calibration", log)


if __name__ == "__main__":
    main()
