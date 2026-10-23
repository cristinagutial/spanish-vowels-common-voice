"""
Stage 5 - Quality control and speaker normalisation.

Two jobs: drop tracking failures and outliers (judged per speaker and per vowel),
then put speakers on a common scale with Lobanov z-scores, since raw Hz mixes
dialect with vocal tract length.

Writes data/vowel_tokens.csv and the s5_ tables.
"""

import numpy as np
import pandas as pd

from . import config
from .utils import banner, get_logger, save_table

log = get_logger(__name__)


def apply_range_filters(df):
    """Drop measurements outside plausible ranges."""
    funnel = [("extracted", len(df))]

    df = df.dropna(subset=["F1_50", "F2_50"])
    funnel.append(("measurable at midpoint", len(df)))

    df = df[df["F1_50"].between(*config.F1_RANGE) & df["F2_50"].between(*config.F2_RANGE)]
    funnel.append(("F1/F2 in range", len(df)))

    # F2 tiene que ser mayor que F1 por definicion; si no lo es, el tracker ha
    # juntado o cambiado los formantes y el token no vale
    df = df[df["F2_50"] > df["F1_50"]]
    funnel.append(("F2 > F1", len(df)))

    return df, funnel


def reject_outliers(df, funnel):
    """
    Reject outliers by z-score within each speaker and each vowel.

    Comparing a token against the same speaker saying the same vowel catches much
    more than a global window, such as an /i/ tracked as if it were an /e/.
    """
    def flag(group):
        keep = pd.Series(True, index=group.index)
        for col in ["F1_50", "F2_50"]:
            if len(group) < 5:
                continue
            values = group[col]
            sd = values.std(ddof=1)
            if not np.isfinite(sd) or sd == 0:
                continue
            keep &= (values - values.mean()).abs() / sd <= config.OUTLIER_Z
        return keep

    mask = (
        df.groupby(["client_id", "vowel"], group_keys=False)
          .apply(flag, include_groups=False)
          .reindex(df.index, fill_value=True)
    )
    df = df[mask]
    funnel.append(("within-speaker outliers removed", len(df)))
    return df, funnel


def lobanov(df, col):
    """
    Lobanov z-scores, with each speaker's mean and SD taken over their vowel means.

    Using the five vowel means instead of all tokens means every vowel weighs the
    same, so it does not matter how many /a/ or /u/ a speaker's sentences had.
    NaN for speakers who are missing a vowel.
    """
    counts = df.groupby(["client_id", "vowel"]).size().unstack(fill_value=0)
    counts = counts.reindex(columns=config.VOWELS, fill_value=0)
    eligible = counts[(counts >= config.MIN_TOKENS_PER_VOWEL_LOBANOV).all(axis=1)].index

    means = df.groupby(["client_id", "vowel"])[col].mean().unstack()
    centre = means.mean(axis=1)
    spread = means.std(axis=1, ddof=1)

    mu = df["client_id"].map(centre)
    sigma = df["client_id"].map(spread)
    z = (df[col] - mu) / sigma
    ok = df["client_id"].isin(eligible) & (sigma > 0)
    return z.where(ok), eligible


def plausible_vowel_space(df):
    """
    Speakers whose vowel means form a real triangle: /i/ in front of /u/, /a/ below both.

    Catches tracks stuck on the same formant for every vowel, which no accent produces.
    """
    means = df.groupby(["client_id", "vowel"])[["F1_50", "F2_50"]].mean().unstack()
    f1, f2 = means["F1_50"], means["F2_50"]
    ok = (
        (f2["i"] - f2["u"] >= config.MIN_F2_I_MINUS_U)
        & (f1["a"] - f1[["i", "u"]].max(axis=1) >= config.MIN_F1_A_MINUS_CLOSE)
    )
    return set(ok[ok].index)


def process(df):
    """QC, vowel-space check and Lobanov on the F1_50/F2_50 columns."""
    df, funnel = apply_range_filters(df)
    df, funnel = reject_outliers(df, funnel)

    df["F1_50_lob"], eligible = lobanov(df, "F1_50")
    df["F2_50_lob"], _ = lobanov(df, "F2_50")
    plausible = plausible_vowel_space(df[df["client_id"].isin(eligible)])
    df["lobanov_ok"] = df["client_id"].isin(plausible)

    speakers = {
        "with tokens": df["client_id"].nunique(),
        "all 5 vowels": len(eligible),
        "plausible vowel space": len(plausible),
    }
    return df, funnel, speakers, eligible, plausible


def main():
    banner(log, "STAGE 5 - QUALITY CONTROL AND NORMALISATION")

    raw = pd.read_csv(config.TOKENS_RAW_CSV)
    log.info(f"{len(raw):,} raw vowel tokens")

    df, funnel, speakers, eligible, plausible = process(raw.drop(columns=["F1_50_sexceil", "F2_50_sexceil"]))

    banner(log, "QC FUNNEL")
    start = funnel[0][1]
    for label, n in funnel:
        log.info(f"  {label:<34} {n:>7,}  ({n/start:.1%})")

    banner(log, "NORMALISATION")
    for label, n in speakers.items():
        log.info(f"  speakers {label:<24} {n:>5}")
    failed = df[df["client_id"].isin(set(eligible) - plausible)].drop_duplicates("client_id")
    log.info(f"  dropped for an implausible vowel space: "
             f"{failed['dialecto'].value_counts().to_dict()}")

    by_dialect = df[df["lobanov_ok"]].groupby("dialecto").agg(
        tokens=("vowel", "size"), speakers=("client_id", "nunique")
    )
    log.info("  " + by_dialect.to_string().replace("\n", "\n  "))

    funnel_df = pd.DataFrame(funnel, columns=["stage", "n_tokens"])
    funnel_df["pct_retained"] = funnel_df["n_tokens"] / start * 100

    coverage = df.groupby(["client_id", "dialecto", "sex"]).agg(
        n_tokens=("vowel", "size"), n_vowel_types=("vowel", "nunique"),
    ).reset_index()
    coverage["all_vowels"] = coverage["client_id"].isin(eligible)
    coverage["plausible"] = coverage["client_id"].isin(plausible)

    df.to_csv(config.TOKENS_CSV, index=False)
    log.info(f"\n  -> {config.TOKENS_CSV.relative_to(config.ROOT)}")
    save_table(funnel_df, "s5_qc_funnel", log)
    save_table(coverage, "s5_speaker_coverage", log)

    # sensibilidad: lo mismo con el techo por sexo del v2.0
    alt = raw.drop(columns=["F1_50", "F2_50"]).rename(
        columns={"F1_50_sexceil": "F1_50", "F2_50_sexceil": "F2_50"})
    alt, _, alt_speakers, _, _ = process(alt)
    alt.to_csv(config.TOKENS_SEXCEIL_CSV, index=False)
    log.info(f"\nSex-based ceiling (sensitivity): {alt_speakers['plausible vowel space']} usable speakers")
    log.info(f"  -> {config.TOKENS_SEXCEIL_CSV.relative_to(config.ROOT)}")


if __name__ == "__main__":
    main()
