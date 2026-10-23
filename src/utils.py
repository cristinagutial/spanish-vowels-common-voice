"""
Helpers used by several stages: corpus metadata, effect sizes, p-value
correction and bootstrapping by speaker.
"""

import logging
import sys

import numpy as np
import pandas as pd
from scipy import stats

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

    Only used as a fallback for the formant ceiling and as a model covariate.
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


def bootstrap_d(a, b, n_boot=None, seed=None):
    """
    Bootstrap distribution of Cohen's d, resampling speakers.

    `a` and `b` hold one value per speaker, so drawing values is drawing speakers.
    """
    rng = np.random.default_rng(config.SEED if seed is None else seed)
    n_boot = n_boot or config.BOOTSTRAP_N
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return np.array([])

    ra = a[rng.integers(0, na, size=(n_boot, na))]
    rb = b[rng.integers(0, nb, size=(n_boot, nb))]
    pooled = ((na - 1) * ra.var(axis=1, ddof=1) + (nb - 1) * rb.var(axis=1, ddof=1)) / (na + nb - 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        d = (ra.mean(axis=1) - rb.mean(axis=1)) / np.sqrt(pooled)
    return d[np.isfinite(d)]


def percentile_ci(values, level=0.95):
    """Percentile interval of a bootstrap distribution."""
    if len(values) == 0:
        return (np.nan, np.nan)
    tail = (1 - level) / 2 * 100
    lo, hi = np.percentile(values, [tail, 100 - tail])
    return float(lo), float(hi)


def tost_welch(a, b, bound_d):
    """
    Two one-sided Welch tests for equivalence within +/- bound_d.

    The bound is in Cohen's d and gets turned into the measure's units with the
    pooled SD. A small p means the difference sits inside the bound.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na, nb = len(a), len(b)
    va, vb = a.var(ddof=1) / na, b.var(ddof=1) / nb
    se = np.sqrt(va + vb)
    df = (va + vb) ** 2 / (va ** 2 / (na - 1) + vb ** 2 / (nb - 1))
    pooled_sd = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    delta = bound_d * pooled_sd
    diff = a.mean() - b.mean()

    # H0 de abajo: diff <= -delta. H0 de arriba: diff >= +delta
    p_lower = stats.t.sf((diff + delta) / se, df)
    p_upper = stats.t.cdf((diff - delta) / se, df)
    return float(max(p_lower, p_upper))


def compare(a, b, labels, bound_d=None):
    """
    Welch test, Cohen's d and speaker-bootstrap intervals for two groups of speakers.

    With `bound_d` it also runs the equivalence test and says whether the 90% CI
    of d falls inside the bound.
    """
    a = pd.Series(a, dtype=float).dropna().to_numpy()
    b = pd.Series(b, dtype=float).dropna().to_numpy()
    left, right = labels
    out = {f"n_{left}": len(a), f"n_{right}": len(b),
           f"mean_{left}": a.mean() if len(a) else np.nan,
           f"mean_{right}": b.mean() if len(b) else np.nan}
    if len(a) < 3 or len(b) < 3:
        return out

    boot = bootstrap_d(a, b)
    lo95, hi95 = percentile_ci(boot, 0.95)
    lo90, hi90 = percentile_ci(boot, 0.90)
    out.update({
        "d": cohens_d(a, b),
        "ci95_lo": lo95, "ci95_hi": hi95,
        "ci90_lo": lo90, "ci90_hi": hi90,
        "p": float(stats.ttest_ind(a, b, equal_var=False)[1]),
    })
    if bound_d is not None:
        out["p_tost"] = tost_welch(a, b, bound_d)
        out["equivalent"] = bool(lo90 > -bound_d and hi90 < bound_d)
    return out


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
    """Save a results table as CSV."""
    csv_path = config.TABLE_DIR / f"{name}.csv"
    df.to_csv(csv_path, index=index)
    if logger:
        logger.info(f"  -> {csv_path.relative_to(config.ROOT)}")
    return csv_path
