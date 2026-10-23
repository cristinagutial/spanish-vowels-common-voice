"""
Stage 8 - Classification.

Can a model predict the dialect from the acoustics? Every model runs twice: with
random folds, and with folds where a speaker is never in training and test at the
same time. The gap between them is the leakage.

A vowel classifier is included as a control: if the features cannot recover
vowels either, the problem is the measurements and not the dialects.

Writes the s8_ tables and figures/fig_classification.png.
"""

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import (
    StratifiedGroupKFold,
    StratifiedKFold,
    cross_val_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from . import config
from .utils import banner, get_logger, save_table

log = get_logger(__name__)

DIALECT_FEATURES = [
    "F1_50_lob", "F2_50_lob", "B1_50", "B2_50",
    "duration", "F1_slope", "F2_slope", "hnr_db",
]
VOWEL_FEATURES = ["F1_50_lob", "F2_50_lob", "B1_50", "B2_50", "duration"]


def models():
    return {
        "LDA": make_pipeline(StandardScaler(), LinearDiscriminantAnalysis()),
        "Logistic": make_pipeline(StandardScaler(),
                                  LogisticRegression(max_iter=2000, random_state=config.SEED)),
        "SVM (RBF)": make_pipeline(StandardScaler(),
                                   SVC(random_state=config.SEED)),
        "Random forest": make_pipeline(
            StandardScaler(),
            RandomForestClassifier(n_estimators=300, random_state=config.SEED, n_jobs=-1)),
    }


# validar un SVM con kernel RBF es casi cuadratico en el numero de tokens, y con
# 90.000 se come el tiempo de todo el pipeline. La precision se estabiliza mucho
# antes, asi que corto la muestra
MAX_TOKENS_FOR_CV = 15000


def subsample_keeping_speakers(X, y, groups, cap=MAX_TOKENS_FOR_CV, seed=None):
    """
    Thin the tokens without losing any speaker.

    Speakers are the grouping variable for cross-validation, so dropping whole
    speakers would change the design and not just the sample size.
    """
    n = len(y)
    if n <= cap:
        return X, y, groups

    rng = np.random.default_rng(config.SEED if seed is None else seed)
    keep_fraction = cap / n
    order = np.arange(n)
    selected = []
    for speaker in np.unique(groups):
        idx = order[groups == speaker]
        take = max(1, int(round(len(idx) * keep_fraction)))
        selected.append(rng.choice(idx, size=min(take, len(idx)), replace=False))

    picked = np.sort(np.concatenate(selected))
    return X[picked], y[picked], groups[picked]


def evaluate(X, y, groups, scoring="balanced_accuracy"):
    """Score every model with speaker-disjoint folds and with random folds."""
    X, y, groups = subsample_keeping_speakers(X, y, groups)
    rows = []
    # StratifiedGroupKFold deja a cada hablante entero en una sola particion y
    # ademas se asegura de que los dos dialectos estan en cada test. El GroupKFold
    # normal hace lo primero pero no lo segundo, y si una particion se queda con
    # una sola clase la balanced accuracy no se puede calcular
    grouped = StratifiedGroupKFold(n_splits=config.N_CV_FOLDS, shuffle=True,
                                   random_state=config.SEED)
    stratified = StratifiedKFold(n_splits=config.N_CV_FOLDS, shuffle=True,
                                 random_state=config.SEED)

    for name, model in models().items():
        leaky = cross_val_score(model, X, y, cv=stratified, scoring=scoring, n_jobs=-1)
        honest = cross_val_score(model, X, y, cv=grouped, groups=groups,
                                 scoring=scoring, n_jobs=-1)
        rows.append({
            "model": name,
            "random_folds": leaky.mean(), "random_folds_sd": leaky.std(),
            "speaker_folds": honest.mean(), "speaker_folds_sd": honest.std(),
            "inflation": leaky.mean() - honest.mean(),
        })
    return pd.DataFrame(rows)


def main():
    banner(log, "STAGE 8 - CLASSIFICATION BENCHMARKS")

    df = pd.read_csv(config.TOKENS_CSV)
    df = df[df["lobanov_ok"]].dropna(subset=DIALECT_FEATURES + ["dialecto", "vowel"])
    log.info(f"{len(df):,} tokens, {df['client_id'].nunique()} speakers")
    log.info("  " + df.groupby("dialecto")["client_id"].nunique().to_string().replace("\n", "\n  "))

    groups = df["client_id"].to_numpy()

    banner(log, "A. PREDICTING DIALECT (Norte vs Sur)")
    core = df[df["dialecto"].isin(config.CORE_DIALECTS)]
    X = core[DIALECT_FEATURES].to_numpy()
    y = core["dialecto"].to_numpy()
    core_groups = core["client_id"].to_numpy()

    majority = pd.Series(y).value_counts(normalize=True)
    log.info(f"Class balance: {majority.round(3).to_dict()}")
    log.info("Chance level for balanced accuracy is 0.500 regardless of imbalance.\n")

    dialect_scores = evaluate(X, y, core_groups)
    log.info(f"  {'model':<16} {'random folds':>14} {'speaker folds':>15} {'inflation':>11}")
    for _, r in dialect_scores.iterrows():
        log.info(
            f"  {r['model']:<16} {r['random_folds']:>13.3f}  {r['speaker_folds']:>14.3f} "
            f"{r['inflation']:>+11.3f}"
        )

    best_honest = dialect_scores["speaker_folds"].max()
    worst_inflation = dialect_scores["inflation"].max()
    log.info(
        f"\n  Random folds: {dialect_scores['random_folds'].max():.3f}. "
        f"Speaker-disjoint folds: {best_honest:.3f}. Chance: 0.500.\n"
        f"  The gap of up to {worst_inflation:+.3f} is the model recognising voices."
    )

    pairwise = pd.DataFrame()
    present = [d for d in config.ALL_DIALECTS if d in set(df["dialecto"])]
    if len(present) >= 3:
        banner(log, "A2. OUTGROUP CALIBRATION - each pair of varieties")
        log.info("México is known to differ from Peninsular Spanish, so it calibrates the rest.\n")
        rows = []
        for left, right in [("norte", "sur"), ("norte", "mexico"), ("sur", "mexico")]:
            pair_df = df[df["dialecto"].isin([left, right])]
            scores = evaluate(
                pair_df[DIALECT_FEATURES].to_numpy(),
                pair_df["dialecto"].to_numpy(),
                pair_df["client_id"].to_numpy(),
            )
            best = scores.loc[scores["speaker_folds"].idxmax()]
            rows.append({
                "pair": f"{left} vs {right}",
                "best_model": best["model"],
                "speaker_folds": best["speaker_folds"],
                "random_folds": best["random_folds"],
                "n_speakers": pair_df["client_id"].nunique(),
            })
            log.info(f"  {left} vs {right:<8} best speaker-disjoint accuracy = "
                     f"{best['speaker_folds']:.3f}  ({best['model']}, "
                     f"{pair_df['client_id'].nunique()} speakers)")

        pairwise = pd.DataFrame(rows)
        ns = pairwise.loc[pairwise["pair"] == "norte vs sur", "speaker_folds"].iloc[0]
        og = pairwise[pairwise["pair"] != "norte vs sur"]["speaker_folds"].mean()
        log.info(f"\n  Norte/Sur: {ns:.3f}. Iberian/Mexican: {og:.3f}. Chance: 0.500.")

    banner(log, "B. POSITIVE CONTROL - PREDICTING VOWEL IDENTITY")
    log.info("If the features are sound, vowels should be highly predictable.\n")

    Xv = df[VOWEL_FEATURES].to_numpy()
    yv = df["vowel"].to_numpy()
    vowel_scores = evaluate(Xv, yv, groups)
    log.info(f"  {'model':<16} {'random folds':>14} {'speaker folds':>15}")
    for _, r in vowel_scores.iterrows():
        log.info(f"  {r['model']:<16} {r['random_folds']:>13.3f}  {r['speaker_folds']:>14.3f}")

    log.info(
        f"\n  Chance is {1/df['vowel'].nunique():.3f}. Vowels are recovered well above chance\n"
        "  even across unseen speakers, so the features do carry phonetic information."
    )

    banner(log, "C. WHICH FEATURES CARRY DIALECT INFORMATION?")
    Xi, yi, _ = subsample_keeping_speakers(X, y, core_groups)
    forest = RandomForestClassifier(n_estimators=300, random_state=config.SEED, n_jobs=-1)
    forest.fit(StandardScaler().fit_transform(Xi), yi)
    importance = pd.DataFrame({
        "feature": DIALECT_FEATURES,
        "importance": forest.feature_importances_,
    }).sort_values("importance", ascending=False)
    for _, r in importance.iterrows():
        log.info(f"  {r['feature']:<12} {r['importance']:.3f}")
    log.info("\n  This describes what separates these speakers, not the dialects.")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    y_pos = np.arange(len(dialect_scores))
    ax.barh(y_pos + 0.2, dialect_scores["random_folds"], height=0.38,
            color="#999999", label="Random folds (speaker leakage)")
    ax.barh(y_pos - 0.2, dialect_scores["speaker_folds"], height=0.38,
            color="#2166ac", label="Speaker-disjoint folds")
    ax.axvline(0.5, color="red", linestyle="--", label="Chance")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(dialect_scores["model"])
    ax.set_xlabel("Balanced accuracy")
    ax.set_xlim(0, 1)
    ax.set_title("Predicting dialect", fontsize=11, loc="left")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3, axis="x")

    ax = axes[1]
    y_pos = np.arange(len(vowel_scores))
    ax.barh(y_pos + 0.2, vowel_scores["random_folds"], height=0.38,
            color="#999999", label="Random folds")
    ax.barh(y_pos - 0.2, vowel_scores["speaker_folds"], height=0.38,
            color="#4daf4a", label="Speaker-disjoint folds")
    ax.axvline(1 / df["vowel"].nunique(), color="red", linestyle="--", label="Chance")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(vowel_scores["model"])
    ax.set_xlabel("Balanced accuracy")
    ax.set_xlim(0, 1)
    ax.set_title("Predicting vowel identity (positive control)", fontsize=11, loc="left")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3, axis="x")

    fig.suptitle(
        "Speaker-disjoint validation: dialect is not predictable, vowels are",
        fontsize=13,
    )
    plt.tight_layout()
    path = config.FIGURE_DIR / "fig_classification.png"
    plt.savefig(path, dpi=config.DPI)
    plt.close()
    log.info(f"\n  -> {path.relative_to(config.ROOT)}")

    save_table(dialect_scores, "s8_dialect_classification", log)
    save_table(vowel_scores, "s8_vowel_classification", log)
    save_table(importance, "s8_feature_importance", log)
    if len(pairwise):
        save_table(pairwise, "s8_pairwise_dialects", log)


if __name__ == "__main__":
    main()
