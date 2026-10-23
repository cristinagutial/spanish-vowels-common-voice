"""
Stage 5 - Quality control and speaker normalisation.

Two jobs: drop tracking failures and outliers (judged per speaker and per vowel),
then put speakers on a common scale with Lobanov z-scores and Bark, since raw Hz
mixes dialect with vocal tract length.

Writes data/vowel_tokens.csv and the s5_ tables.
"""

import numpy as np
import pandas as pd

from . import config
from .utils import banner, get_logger, save_table

log = get_logger(__name__)

# un hablante necesita unas cuantas vocales distintas para que su media y su
# desviacion sirvan de referencia en Lobanov. El umbral es un equilibrio: si lo
# subo, la media de cada hablante es mas estable, pero pierdo hablantes, y el
# numero de hablantes es justo lo que limita todo el estudio. Un clip de 5
# segundos da unas 17 vocales de 4-5 tipos, asi que subirlo mas dejaria fuera a
# todo el que solo mando un clip, que son la mayoria
MIN_TOKENS_FOR_LOBANOV = 12
MIN_VOWEL_TYPES_FOR_LOBANOV = 4


def bark(hz):
    """Traunmuller's Bark scale, closer to how we actually hear."""
    return 26.81 * hz / (1960.0 + hz) - 0.53


def apply_range_filters(df):
    """Drop measurements outside plausible ranges."""
    funnel = [("extracted", len(df))]

    df = df.dropna(subset=["F1_50", "F2_50", "B1_50", "B2_50"])
    funnel.append(("measurable at midpoint", len(df)))

    df = df[df["F1_50"].between(*config.F1_RANGE) & df["F2_50"].between(*config.F2_RANGE)]
    funnel.append(("F1/F2 in range", len(df)))

    df = df[df["B1_50"].between(*config.B1_RANGE) & df["B2_50"].between(*config.B2_RANGE)]
    funnel.append(("B1/B2 in range", len(df)))

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
          .apply(flag)
          .reindex(df.index, fill_value=True)
    )
    df = df[mask]
    funnel.append(("within-speaker outliers removed", len(df)))
    return df, funnel


def add_normalisations(df):
    """Add the Lobanov and Bark values of F1/F2 at the midpoint."""
    for col in ["F1_50", "F2_50"]:
        df[f"{col}_bark"] = bark(df[col])

    stats = df.groupby("client_id").agg(
        n_tokens=("vowel", "size"),
        n_vowel_types=("vowel", "nunique"),
    )
    eligible = stats[
        (stats["n_tokens"] >= MIN_TOKENS_FOR_LOBANOV)
        & (stats["n_vowel_types"] >= MIN_VOWEL_TYPES_FOR_LOBANOV)
    ].index

    df["lobanov_ok"] = df["client_id"].isin(eligible)

    for col in ["F1_50", "F2_50"]:
        grouped = df.groupby("client_id")[col]
        centred = df[col] - grouped.transform("mean")
        spread = grouped.transform("std")
        df[f"{col}_lob"] = np.where(
            df["lobanov_ok"] & (spread > 0), centred / spread, np.nan
        )

    return df, stats, eligible


def add_dynamics(df):
    """
    How much the formants move inside the vowel.

    A plain vowel holds its target, a diphthongised one does not. Only measurable
    now that I know where each vowel starts and ends.
    """
    df["F1_slope"] = df["F1_75"] - df["F1_25"]
    df["F2_slope"] = df["F2_75"] - df["F2_25"]
    df["F2_range"] = df[["F2_25", "F2_50", "F2_75"]].max(axis=1) - \
                     df[["F2_25", "F2_50", "F2_75"]].min(axis=1)
    return df


def main():
    banner(log, "STAGE 5 - QUALITY CONTROL AND NORMALISATION")

    df = pd.read_csv(config.TOKENS_RAW_CSV)
    log.info(f"{len(df):,} raw vowel tokens")

    df, funnel = apply_range_filters(df)
    df, funnel = reject_outliers(df, funnel)

    banner(log, "QC FUNNEL")
    start = funnel[0][1]
    for label, n in funnel:
        log.info(f"  {label:<34} {n:>7,}  ({n/start:.1%})")

    df = add_dynamics(df)
    df, speaker_stats, eligible = add_normalisations(df)

    banner(log, "NORMALISATION")
    log.info(
        f"Lobanov applied to {len(eligible)} of {df['client_id'].nunique()} speakers "
        f"(>= {MIN_TOKENS_FOR_LOBANOV} tokens and >= {MIN_VOWEL_TYPES_FOR_LOBANOV} vowel types)"
    )
    log.info(f"  covering {df['lobanov_ok'].sum():,} of {len(df):,} tokens "
             f"({df['lobanov_ok'].mean():.1%})")

    by_dialect = df[df["lobanov_ok"]].groupby("dialecto").agg(
        tokens=("vowel", "size"), speakers=("client_id", "nunique")
    )
    log.info("\nNormalisable data by dialect:")
    log.info("  " + by_dialect.to_string().replace("\n", "\n  "))

    banner(log, "VOWEL SPACE AFTER NORMALISATION")
    log.info("Raw Hz (midpoint), by dialect:")
    raw = df.groupby(["dialecto", "vowel"])[["F1_50", "F2_50"]].mean().round(0)
    log.info("  " + raw.to_string().replace("\n", "\n  "))

    log.info("\nLobanov z-scores, by dialect:")
    lob = (df[df["lobanov_ok"]]
           .groupby(["dialecto", "vowel"])[["F1_50_lob", "F2_50_lob"]].mean().round(2))
    log.info("  " + lob.to_string().replace("\n", "\n  "))

    log.info("\nSex composition after QC (tokens):")
    log.info("  " + pd.crosstab(df["dialecto"], df["sex"]).to_string().replace("\n", "\n  "))

    funnel_df = pd.DataFrame(funnel, columns=["stage", "n_tokens"])
    funnel_df["pct_retained"] = funnel_df["n_tokens"] / start * 100

    coverage = speaker_stats.reset_index()
    coverage["lobanov_eligible"] = coverage["client_id"].isin(eligible)
    coverage = coverage.merge(
        df[["client_id", "dialecto", "sex"]].drop_duplicates("client_id"),
        on="client_id", how="left",
    )

    df.to_csv(config.TOKENS_CSV, index=False)
    log.info(f"\n  -> {config.TOKENS_CSV.relative_to(config.ROOT)}")
    save_table(funnel_df, "s5_qc_funnel", log)
    save_table(coverage, "s5_speaker_coverage", log)


if __name__ == "__main__":
    main()
