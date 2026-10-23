"""
Stage 7 - Clustering, scored against the real vowel labels.

k-means scored with the adjusted Rand index: how much of the five vowels an
unsupervised partition recovers. Unlike silhouette, ARI can be compared across
feature sets of different sizes.

Writes the s7_ tables and figures/fig_clustering.png.
"""

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from . import config
from .utils import banner, get_logger, save_table

log = get_logger(__name__)

# silhouette_score monta la matriz de distancias entera, que con 90.000 tokens es
# inviable. Con una muestra sale practicamente el mismo numero. El ARI es exacto
# y barato, asi que solo muestreo para el silhouette
SILHOUETTE_SAMPLE = 10000


def silhouette(X, labels):
    return silhouette_score(X, labels, sample_size=min(SILHOUETTE_SAMPLE, len(X)),
                            random_state=config.SEED)

FEATURE_SETS = {
    "F1/F2 raw Hz":        ["F1_50", "F2_50"],
    "F1/F2 Lobanov":       ["F1_50_lob", "F2_50_lob"],
    "F1/F2/B1/B2 raw Hz":  ["F1_50", "F2_50", "B1_50", "B2_50"],
    "F1/F2/B1/B2 Lobanov": ["F1_50_lob", "F2_50_lob", "B1_50", "B2_50"],
    "F1/F2 + duration":    ["F1_50_lob", "F2_50_lob", "duration"],
}


def sweep_k(X, truth, k_range=range(2, 9)):
    """Silhouette and ARI over a range of k."""
    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=config.SEED, n_init=20)
        labels = km.fit_predict(X)
        rows.append({
            "k": k,
            "silhouette": silhouette(X, labels),
            "ari": adjusted_rand_score(truth, labels),
        })
    return pd.DataFrame(rows)


def main():
    banner(log, "STAGE 7 - CLUSTERING AGAINST KNOWN VOWEL LABELS")

    df = pd.read_csv(config.TOKENS_CSV)
    df = df[df["lobanov_ok"]].dropna(
        subset=["F1_50_lob", "F2_50_lob", "F1_50", "F2_50", "B1_50", "B2_50", "duration"]
    )
    log.info(f"{len(df):,} tokens, {df['vowel'].nunique()} vowel categories\n")

    truth = df["vowel"].to_numpy()

    # cuanto recupera cada conjunto de features con el k verdadero (5)
    banner(log, "A. DO CLUSTERS RECOVER THE FIVE SPANISH VOWELS? (k = 5)")
    log.info("ARI: 1.0 = perfect recovery, 0.0 = no better than chance.\n")

    rows = []
    for name, cols in FEATURE_SETS.items():
        X = StandardScaler().fit_transform(df[cols])
        km = KMeans(n_clusters=5, random_state=config.SEED, n_init=20)
        labels = km.fit_predict(X)
        rows.append({
            "features": name,
            "n_dimensions": len(cols),
            "ari": adjusted_rand_score(truth, labels),
            "silhouette": silhouette(X, labels),
        })

    recovery = pd.DataFrame(rows).sort_values("ari", ascending=False)
    log.info(f"  {'features':<22} {'dims':>5} {'ARI':>7} {'silhouette':>11}")
    for _, r in recovery.iterrows():
        log.info(f"  {r['features']:<22} {int(r['n_dimensions']):>5} "
                 f"{r['ari']:>7.3f} {r['silhouette']:>11.3f}")

    best = recovery.iloc[0]
    log.info(f"\n  Best recovery: {best['features']} (ARI = {best['ari']:.3f}).")

    # la pregunta de los bandwidths, ahora bien planteada
    banner(log, "B. DOES BANDWIDTH HELP? (the midterm's question, with a valid metric)")
    two_d = recovery[recovery["features"] == "F1/F2 Lobanov"].iloc[0]
    four_d = recovery[recovery["features"] == "F1/F2/B1/B2 Lobanov"].iloc[0]
    log.info(
        f"  F1/F2 only:        ARI = {two_d['ari']:.3f}, silhouette = {two_d['silhouette']:.3f}\n"
        f"  F1/F2 + bandwidth: ARI = {four_d['ari']:.3f}, silhouette = {four_d['silhouette']:.3f}\n"
    )
    log.info(
        f"  Silhouette changes by {four_d['silhouette'] - two_d['silhouette']:+.3f} and ARI by "
        f"{four_d['ari'] - two_d['ari']:+.3f}. Bandwidth makes the vowels "
        + ("harder" if four_d["ari"] < two_d["ari"] else "easier") + " to recover."
    )

    # elegir k
    banner(log, "C. CHOOSING k: what silhouette says versus what is true")
    sweeps = {}
    for name in ["F1/F2 raw Hz", "F1/F2 Lobanov"]:
        X = StandardScaler().fit_transform(df[FEATURE_SETS[name]])
        sweep = sweep_k(X, truth)
        sweeps[name] = sweep
        k_sil = int(sweep.loc[sweep["silhouette"].idxmax(), "k"])
        k_ari = int(sweep.loc[sweep["ari"].idxmax(), "k"])
        log.info(f"\n  {name}:")
        log.info(f"    silhouette peaks at k = {k_sil}; ARI peaks at k = {k_ari} "
                 f"(true number of vowels = 5)")
        for _, r in sweep.iterrows():
            log.info(f"      k={int(r['k'])}: silhouette={r['silhouette']:.3f}  ARI={r['ari']:.3f}")

    log.info("\n  Silhouette prefers few compact blobs, which is why the midterm found k = 3.")

    # que vocales se confunden entre si
    banner(log, "D. WHICH VOWELS DOES CLUSTERING CONFUSE?")
    X = StandardScaler().fit_transform(df[FEATURE_SETS["F1/F2 Lobanov"]])
    labels = KMeans(n_clusters=5, random_state=config.SEED, n_init=20).fit_predict(X)
    confusion = pd.crosstab(df["vowel"], labels)
    log.info("  " + confusion.to_string().replace("\n", "\n  "))

    dominant = confusion.idxmax(axis=0)
    log.info(f"\n  Each cluster's most frequent vowel: {dominant.to_dict()}")

    # la figura
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    ax = axes[0]
    for name, sweep in sweeps.items():
        ax.plot(sweep["k"], sweep["silhouette"], "o-", label=f"{name} (silhouette)")
        ax.plot(sweep["k"], sweep["ari"], "s--", label=f"{name} (ARI)")
    ax.axvline(5, color="gray", linestyle=":", label="true k = 5")
    ax.set_xlabel("k")
    ax.set_ylabel("score")
    ax.set_title("Silhouette and ARI disagree about k", fontsize=11, loc="left")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    ax = axes[1]
    for vowel in config.VOWELS:
        sub = df[df["vowel"] == vowel]
        ax.scatter(sub["F2_50_lob"], sub["F1_50_lob"], s=5, alpha=0.25,
                   color=config.VOWEL_COLORS[vowel], label=f"/{vowel}/")
        ax.annotate(vowel, (sub["F2_50_lob"].mean(), sub["F1_50_lob"].mean()),
                    fontsize=15, fontweight="bold", ha="center",
                    color=config.VOWEL_COLORS[vowel])
    ax.invert_xaxis(); ax.invert_yaxis()
    ax.set_xlabel("F2 (Lobanov z)"); ax.set_ylabel("F1 (Lobanov z)")
    ax.set_title("True vowel categories", fontsize=11, loc="left")
    ax.grid(alpha=0.3)

    ax = axes[2]
    palette = ["#e41a1c", "#377eb8", "#4daf4a", "#ff7f00", "#984ea3"]
    for cl in range(5):
        mask = labels == cl
        ax.scatter(df["F2_50_lob"].to_numpy()[mask], df["F1_50_lob"].to_numpy()[mask],
                   s=5, alpha=0.25, color=palette[cl], label=f"C{cl+1}")
    ax.invert_xaxis(); ax.invert_yaxis()
    ax.set_xlabel("F2 (Lobanov z)"); ax.set_ylabel("F1 (Lobanov z)")
    ax.set_title(f"k-means, k=5 (ARI = {adjusted_rand_score(truth, labels):.2f})",
                 fontsize=11, loc="left")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle("Unsupervised clustering evaluated against known vowel identities", fontsize=13)
    plt.tight_layout()
    path = config.FIGURE_DIR / "fig_clustering.png"
    plt.savefig(path, dpi=config.DPI)
    plt.close()
    log.info(f"\n  -> {path.relative_to(config.ROOT)}")

    save_table(recovery, "s7_feature_set_recovery", log)
    for name, sweep in sweeps.items():
        slug = name.replace("/", "").replace(" ", "_").lower()
        save_table(sweep, f"s7_k_sweep_{slug}", log)
    save_table(confusion.reset_index(), "s7_cluster_confusion", log)


if __name__ == "__main__":
    main()
