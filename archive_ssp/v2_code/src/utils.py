"""
Helpers used by several stages: corpus metadata, effect sizes, p-value
correction and bootstrapping by speaker.
"""

import logging
import sys

import numpy as np
import pandas as pd

from . import config


def get_logger(name):
    """Console logger, so every stage prints the same way."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def banner(logger, text):
    logger.info("\n" + "=" * 74)
    logger.info(text)
    logger.info("=" * 74)


_META_COLUMNS = ["client_id", "path", "sentence", "age", "gender", "accents"]


def assign_dialect(accents):
    """
    Turn the Common Voice `accents` field into one of my dialect labels.

    Speakers can tick several regions, separated by '|'. Anything with more than
    one of my labels is dropped instead of guessed.
    """
    if not isinstance(accents, str):
        return None
    matches = [name for name, prefix in config.DIALECTS.items() if prefix in accents]
    return matches[0] if len(matches) == 1 else None


def load_cv_metadata(usecols=None):
    """Read validated.tsv and add the `dialecto` column."""
    df = pd.read_csv(
        config.VALIDATED_TSV,
        sep="\t",
        low_memory=False,
        usecols=usecols or _META_COLUMNS,
    )
    df["dialecto"] = df["accents"].apply(assign_dialect)
    return df


def normalise_sex(gender):
    """
    Collapse the gender field to male / female / unknown.

    Only used to pick the formant ceiling, which depends on vocal tract length.
    """
    if not isinstance(gender, str):
        return "unknown"
    g = gender.lower()
    if "female" in g or g == "feminine":
        return "female"
    if "male" in g or g == "masculine":
        return "male"
    return "unknown"


def cohens_d(a, b):
    """Cohen's d with a pooled standard deviation."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return np.nan
    pooled_var = ((na - 1) * a.std(ddof=1) ** 2 + (nb - 1) * b.std(ddof=1) ** 2) / (
        na + nb - 2
    )
    if pooled_var <= 0:
        return np.nan
    return (a.mean() - b.mean()) / np.sqrt(pooled_var)


def holm_correction(pvalues):
    """Holm-Bonferroni adjusted p-values."""
    p = np.asarray(pvalues, dtype=float)
    n = len(p)
    order = np.argsort(p)
    adjusted = np.empty(n, dtype=float)
    running_max = 0.0
    for rank, idx in enumerate(order):
        value = (n - rank) * p[idx]
        running_max = max(running_max, value)
        adjusted[idx] = min(running_max, 1.0)
    return adjusted


def bootstrap_ci_by_speaker(df, value_col, group_col, speaker_col,
                            groups, n_boot=None, seed=None):
    """
    Confidence interval for Cohen's d, resampling speakers instead of tokens.

    Resampling tokens assumes every measurement is independent evidence, which is
    the assumption that fails here.
    """
    rng = np.random.default_rng(config.SEED if seed is None else seed)
    n_boot = n_boot or config.BOOTSTRAP_N
    g1, g2 = groups

    speakers = {
        g: df.loc[df[group_col] == g, speaker_col].unique() for g in (g1, g2)
    }
    by_speaker = {
        spk: sub[value_col].to_numpy()
        for spk, sub in df.groupby(speaker_col)
    }

    estimates = []
    for _ in range(n_boot):
        drawn = {}
        for g in (g1, g2):
            picks = rng.choice(speakers[g], size=len(speakers[g]), replace=True)
            drawn[g] = np.concatenate([by_speaker[s] for s in picks])
        if len(drawn[g1]) > 1 and len(drawn[g2]) > 1:
            estimates.append(cohens_d(drawn[g1], drawn[g2]))

    estimates = np.asarray([e for e in estimates if np.isfinite(e)])
    if estimates.size == 0:
        return (np.nan, np.nan)
    return tuple(np.percentile(estimates, [2.5, 97.5]))


def bootstrap_ci_by_token(df, value_col, group_col, groups, n_boot=None, seed=None):
    """The naive token-level bootstrap, kept for the side-by-side comparison."""
    rng = np.random.default_rng(config.SEED if seed is None else seed)
    n_boot = n_boot or config.BOOTSTRAP_N
    g1, g2 = groups
    a = df.loc[df[group_col] == g1, value_col].to_numpy()
    b = df.loc[df[group_col] == g2, value_col].to_numpy()

    estimates = []
    for _ in range(n_boot):
        ra = rng.choice(a, size=len(a), replace=True)
        rb = rng.choice(b, size=len(b), replace=True)
        estimates.append(cohens_d(ra, rb))

    estimates = np.asarray([e for e in estimates if np.isfinite(e)])
    if estimates.size == 0:
        return (np.nan, np.nan)
    return tuple(np.percentile(estimates, [2.5, 97.5]))


def gini(counts):
    """
    Gini coefficient of a count distribution.

    0 means every speaker contributes equally, near 1 means one speaker dominates.
    """
    x = np.sort(np.asarray(counts, dtype=float))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return np.nan
    index = np.arange(1, n + 1)
    return (2.0 * (index * x).sum()) / (n * x.sum()) - (n + 1.0) / n


def save_table(df, name, logger=None, index=False):
    """Save a report table as CSV and LaTeX."""
    csv_path = config.TABLE_DIR / f"{name}.csv"
    tex_path = config.TABLE_DIR / f"{name}.tex"
    df.to_csv(csv_path, index=index)
    try:
        df.to_latex(tex_path, index=index, float_format="%.3f")
    except Exception:  # si falla el LaTeX me da igual, el CSV es lo importante
        pass
    if logger:
        logger.info(f"  -> {csv_path.relative_to(config.ROOT)}")
    return csv_path
