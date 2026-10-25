"""
Stage 7 - H2, the vowel before a word-final /s/.

Compares word-final unstressed vowels closed by /s/ ('casas', 'libros', 'veces')
with the same vowels when nothing follows in the word ('casa', 'libro'), inside
each speaker. Most of the first kind are plurals, so I call them plural and
singular. Southern Spanish is said to open the vowel when the /s/ weakens, so
the prediction is a bigger F1 shift in Sur than in Norte, strongest for /e/ and /o/.

Both kinds have to be followed by a consonant or a pause: before a vowel the /s/
resyllabifies ('casas altas'), and a final vowel before a vowel coarticulates.

Writes the s7_ tables.
"""

import numpy as np
import pandas as pd
from scipy import stats

from . import config
from .utils import banner, compare, get_logger, save_table

log = get_logger(__name__)


def h2_tokens(df):
    """Word-final unstressed /a e o/ in content words, labelled plural or singular."""
    base = df[
        df["lobanov_ok"]
        & df["vowel"].isin(config.H2_VOWELS)
        & (df["stressed"] == False)
        & ~df["word"].isin(config.FUNCTION_WORDS)
        & (df["n_syllables"] >= 2)
    ].copy()

    plural = base["final_s"] & base["after_s"].isin(["consonant", "pause"])
    singular = base["word_final_vowel"] & base["following"].isin(["consonant", "pause"])
    base["type"] = np.select([plural, singular], ["plural", "singular"], default="")
    return base[base["type"] != ""]


def speaker_shifts(tokens):
    """Plural minus singular, per speaker and vowel, where both have enough tokens."""
    cells = (
        tokens.groupby(["client_id", "dialecto", "vowel", "type"])
        .agg(F1=("F1_50_lob", "mean"), F2=("F2_50_lob", "mean"),
             duration=("duration", "mean"), n=("vowel", "size"))
        .reset_index()
    )
    cells = cells[cells["n"] >= config.MIN_TOKENS_PER_H2_CELL]
    wide = cells.pivot_table(index=["client_id", "dialecto", "vowel"], columns="type",
                             values=["F1", "F2", "duration"])
    wide = wide.dropna()
    shifts = pd.DataFrame({
        "F1_shift": wide[("F1", "plural")] - wide[("F1", "singular")],
        "F2_shift": wide[("F2", "plural")] - wide[("F2", "singular")],
        "duration_shift": wide[("duration", "plural")] - wide[("duration", "singular")],
    }).reset_index()
    return shifts


def primary_measure(shifts):
    """Each speaker's F1 shift averaged over the mid vowels they have."""
    mid = shifts[shifts["vowel"].isin(config.H2_PRIMARY_VOWELS)]
    return mid.groupby(["client_id", "dialecto"])["F1_shift"].mean().reset_index()


def describe(shifts):
    """Mean shift with a 95% CI across speakers, by variety and vowel."""
    rows = []
    for (dialect, vowel), sub in shifts.groupby(["dialecto", "vowel"]):
        for measure in ["F1_shift", "F2_shift", "duration_shift"]:
            values = sub[measure].dropna()
            half = stats.t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values))
            rows.append({"dialecto": dialect, "vowel": vowel, "measure": measure,
                         "mean": values.mean(), "ci95_lo": values.mean() - half,
                         "ci95_hi": values.mean() + half, "n_speakers": len(values)})
    return pd.DataFrame(rows)


def main():
    banner(log, "STAGE 7 - H2: VOWELS BEFORE WORD-FINAL /s/")

    df = pd.read_csv(config.TOKENS_CSV)
    tokens = h2_tokens(df)
    log.info("Tokens by variety and type:")
    log.info("  " + pd.crosstab(tokens["dialecto"], tokens["type"]).to_string().replace("\n", "\n  "))

    shifts = speaker_shifts(tokens)
    log.info("\nSpeakers with both types, by vowel:")
    log.info("  " + pd.crosstab(shifts["dialecto"], shifts["vowel"]).to_string().replace("\n", "\n  "))

    banner(log, "PRIMARY TEST - F1 shift on /e o/, Sur vs Norte")
    primary = primary_measure(shifts)
    by = {d: primary[primary["dialecto"] == d]["F1_shift"] for d in config.ALL_DIALECTS}
    result = pd.DataFrame([dict(test="F1 shift /e o/", **compare(by["sur"], by["norte"], ("sur", "norte")))])
    r = result.iloc[0]
    log.info(f"  Sur {r['mean_sur']:+.3f} (n={r['n_sur']})  Norte {r['mean_norte']:+.3f} (n={r['n_norte']})")
    log.info(f"  d = {r['d']:+.2f}  95% CI [{r['ci95_lo']:+.2f}, {r['ci95_hi']:+.2f}]  p = {r['p']:.4f}")

    banner(log, "DESCRIPTIVE - mean shift (plural - singular)")
    summary = describe(shifts)
    f1 = summary[summary["measure"] == "F1_shift"].pivot(index="dialecto", columns="vowel", values="mean")
    log.info("F1 (Lobanov z):\n  " + f1.round(3).to_string().replace("\n", "\n  "))
    dur = summary[summary["measure"] == "duration_shift"].pivot(index="dialecto", columns="vowel", values="mean")
    log.info("Duration (ms):\n  " + (dur * 1000).round(1).to_string().replace("\n", "\n  "))

    banner(log, "WRITING OUTPUTS")
    save_table(tokens.groupby(["dialecto", "vowel", "type"]).size().reset_index(name="n_tokens"),
               "s7_h2_token_counts", log)
    save_table(shifts, "s7_h2_shifts", log)
    save_table(result, "s7_h2_primary", log)
    save_table(summary, "s7_h2_descriptive", log)


if __name__ == "__main__":
    main()
