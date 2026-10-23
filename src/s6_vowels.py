"""
Stage 6 - H1, the stressed vowel space.

Norte vs Sur on stressed vowels in content words, one mean per speaker and
vowel. The question is whether the two are equivalent within d = +/-0.30, so
the main test is an equivalence test (TOST). México is the positive control:
the same measures have to pick up the Peninsular/Mexican difference.

Writes the s6_ tables.
"""

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from statsmodels.stats.power import TTestIndPower

from . import config
from .utils import banner, compare, get_logger, holm_correction, save_table

log = get_logger(__name__)

MEASURES = [(formant, vowel) for formant in ("F1", "F2") for vowel in config.VOWELS]


def h1_tokens(df):
    """Stressed vowels in content words, from speakers that can be normalised."""
    keep = (
        df["lobanov_ok"]
        & (df["stressed"] == True)
        & ~df["word"].isin(config.FUNCTION_WORDS)
    )
    return df[keep]


def speaker_vowel_cells(tokens):
    """One row per speaker and vowel, with at least MIN_TOKENS_PER_CELL tokens."""
    cells = (
        tokens.groupby(["client_id", "dialecto", "sex", "vowel"])
        .agg(F1=("F1_50_lob", "mean"), F2=("F2_50_lob", "mean"),
             F1_hz=("F1_50", "mean"), F2_hz=("F2_50", "mean"),
             n=("vowel", "size"))
        .reset_index()
    )
    return cells[cells["n"] >= config.MIN_TOKENS_PER_CELL]


def contrast(cells, left, right, bound_d=None):
    """The ten vowel measures for one pair of varieties."""
    rows = []
    for formant, vowel in MEASURES:
        sub = cells[cells["vowel"] == vowel]
        a = sub[sub["dialecto"] == left][formant]
        b = sub[sub["dialecto"] == right][formant]
        row = {"pair": f"{left} vs {right}", "measure": f"{formant} /{vowel}/"}
        row.update(compare(a, b, (left, right), bound_d=bound_d))
        rows.append(row)
    out = pd.DataFrame(rows)
    out["p_holm"] = holm_correction(out["p"])
    return out


def vowel_space_area(cells):
    """Convex hull of a speaker's five vowel means in Lobanov space."""
    rows = []
    for (cid, dialect), group in cells.groupby(["client_id", "dialecto"]):
        if group["vowel"].nunique() < len(config.VOWELS):
            continue
        area = ConvexHull(group[["F2", "F1"]].to_numpy()).volume  # en 2D 'volume' es el area
        rows.append({"client_id": cid, "dialecto": dialect, "vsa": area})
    return pd.DataFrame(rows)


def power_table(results):
    """Smallest d this sample detects with 80% and 90% power, at the median n."""
    n_left = int(results["n_norte"].median())
    n_right = int(results["n_sur"].median())
    analysis = TTestIndPower()
    rows = []
    for power in (0.80, 0.90):
        d = analysis.solve_power(effect_size=None, nobs1=n_right, ratio=n_left / n_right,
                                 alpha=config.ALPHA, power=power, alternative="two-sided")
        rows.append({"power": power, "n_norte": n_left, "n_sur": n_right, "detectable_d": d})
    return pd.DataFrame(rows)


def log_contrast(results, equivalence=False):
    for _, r in results.iterrows():
        line = (f"  {r['measure']:<8} d = {r['d']:+.2f}  "
                f"90% CI [{r['ci90_lo']:+.2f}, {r['ci90_hi']:+.2f}]  p_holm = {r['p_holm']:.3f}")
        if equivalence:
            line += f"  p_tost = {r['p_tost']:.3f}  {'equivalent' if r['equivalent'] else 'not shown'}"
        log.info(line)


def main():
    banner(log, "STAGE 6 - H1: STRESSED VOWEL SPACE")

    df = pd.read_csv(config.TOKENS_CSV)
    tokens = h1_tokens(df)
    cells = speaker_vowel_cells(tokens)
    log.info(f"{len(tokens):,} stressed content-word tokens -> {len(cells):,} speaker x vowel cells")
    log.info("Speakers: " + ", ".join(
        f"{d} {n}" for d, n in cells.groupby("dialecto")["client_id"].nunique().items()))

    banner(log, "NORTE vs SUR - EQUIVALENCE (bound d = +/-%.2f)" % config.TOST_BOUND)
    h1 = contrast(cells, "norte", "sur", bound_d=config.TOST_BOUND)
    log_contrast(h1, equivalence=True)
    log.info(f"\n  {h1['equivalent'].sum()} of {len(h1)} measures inside the bound; "
             f"{(h1['p_holm'] < config.ALPHA).sum()} differ at p_holm < {config.ALPHA}")

    banner(log, "MEXICO AS POSITIVE CONTROL")
    control = pd.concat([contrast(cells, "norte", "mexico"),
                         contrast(cells, "sur", "mexico")], ignore_index=True)
    for pair, sub in control.groupby("pair", sort=False):
        log.info(f"  {pair:<16} mean |d| = {sub['d'].abs().mean():.3f}  max |d| = "
                 f"{sub['d'].abs().max():.3f}  significant {(sub['p_holm'] < config.ALPHA).sum()}/{len(sub)}")
    log.info(f"  {'norte vs sur':<16} mean |d| = {h1['d'].abs().mean():.3f}  max |d| = "
             f"{h1['d'].abs().max():.3f}  significant {(h1['p_holm'] < config.ALPHA).sum()}/{len(h1)}")

    banner(log, "VOWEL SPACE AREA (descriptive)")
    vsa = vowel_space_area(cells)
    area = pd.DataFrame([
        dict(pair=f"{l} vs {r}", **compare(vsa[vsa["dialecto"] == l]["vsa"],
                                           vsa[vsa["dialecto"] == r]["vsa"], (l, r)))
        for l, r in [("norte", "sur"), ("norte", "mexico"), ("sur", "mexico")]
    ])
    log.info("  " + vsa.groupby("dialecto")["vsa"].agg(["mean", "size"]).round(2)
             .to_string().replace("\n", "\n  "))

    power = power_table(h1)
    for _, r in power.iterrows():
        log.info(f"\n  {r['power']:.0%} power for d >= {r['detectable_d']:.2f} "
                 f"(median n: {r['n_norte']} Norte, {r['n_sur']} Sur)")

    banner(log, "SENSITIVITY - same test with the sex-based ceiling of v2.0")
    alt_cells = speaker_vowel_cells(h1_tokens(pd.read_csv(config.TOKENS_SEXCEIL_CSV)))
    alt = contrast(alt_cells, "norte", "sur", bound_d=config.TOST_BOUND)
    log_contrast(alt, equivalence=True)
    log.info(f"\n  {alt['equivalent'].sum()} of {len(alt)} measures inside the bound; "
             f"{(alt['p_holm'] < config.ALPHA).sum()} differ at p_holm < {config.ALPHA}")

    banner(log, "WRITING OUTPUTS")
    save_table(alt, "s6_h1_norte_sur_sexceil", log)
    save_table(cells, "s6_h1_cells", log)
    save_table(h1, "s6_h1_norte_sur", log)
    save_table(control, "s6_h1_control", log)
    save_table(vsa, "s6_vowel_space_area", log)
    save_table(area, "s6_vowel_space_area_tests", log)
    save_table(power, "s6_power", log)


if __name__ == "__main__":
    main()
