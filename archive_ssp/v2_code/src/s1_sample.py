"""
Stage 1 - Sampling by speaker.

Builds the list of clips. Every row keeps its client_id and a random rank inside
its speaker, so a per-speaker cap can be applied later.

Writes data/sample_manifest.csv and tables/s1_sample_design.csv.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .utils import banner, get_logger, gini, load_cv_metadata, normalise_sex, save_table

log = get_logger(__name__)


def build_manifest(available=None, dialects=None):
    """
    Build the clip manifest.

    `available` is what is on disk (None = the whole corpus), `dialects` is which
    labels to keep. All three are included by default, México as well, so a plain
    run_all.py does not rebuild the sample without the control group.
    """
    dialects = dialects or config.ALL_DIALECTS

    log.info("Reading corpus metadata...")
    meta = load_cv_metadata()
    log.info(f"  {len(meta):,} validated clips in the corpus")

    df = meta[meta["dialecto"].isin(dialects)].copy()
    log.info(f"  {len(df):,} clips carry exactly one of the target dialect labels")

    if available is not None:
        before = len(df)
        df = df[df["path"].isin(available)]
        log.info(f"  {len(df):,} of those are present on disk (from {before:,})")

    df["sex"] = df["gender"].apply(normalise_sex)
    df["age"] = df["age"].fillna("unknown")

    # el tope de hablantes va antes de elegir clips, asi me quedo con hablantes
    # enteros y no con un trocito de muchos
    for dialect, ceiling in config.MAX_SPEAKERS_PER_DIALECT.items():
        if dialect not in dialects:
            continue
        speakers = df.loc[df["dialecto"] == dialect, "client_id"].unique()
        if len(speakers) <= ceiling:
            continue
        elegidos = pd.Series(speakers).sample(n=ceiling, random_state=config.SEED)
        drop = (df["dialecto"] == dialect) & (~df["client_id"].isin(set(elegidos)))
        log.info(f"  {dialect}: sampling {ceiling} of {len(speakers):,} speakers")
        df = df[~drop]

    # orden aleatorio dentro de cada hablante, pero con semilla para que salga
    # igual siempre. Si usara el orden del fichero, el tope se quedaria siempre
    # con los mismos clips y no con una muestra sin sesgo
    df = df.sample(frac=1.0, random_state=config.SEED).reset_index(drop=True)
    df["speaker_clip_rank"] = df.groupby("client_id").cumcount() + 1

    df = df.sort_values(["dialecto", "client_id", "speaker_clip_rank"]).reset_index(drop=True)

    keep = ["path", "client_id", "dialecto", "sex", "age", "sentence", "speaker_clip_rank"]
    return df[keep]


def describe(df, cap):
    """Summarise the sample at a given per-speaker cap."""
    filas = []
    for dialect, sub in df.groupby("dialecto"):
        capped = sub[sub["speaker_clip_rank"] <= cap]
        counts = capped["client_id"].value_counts()
        filas.append({
            "dialect": dialect,
            "clips_all": len(sub),
            "speakers": sub["client_id"].nunique(),
            f"clips_at_cap{cap}": len(capped),
            "median_clips_per_speaker": float(counts.median()) if len(counts) else 0.0,
            "gini_all": gini(sub["client_id"].value_counts().to_numpy()),
            f"gini_at_cap{cap}": gini(counts.to_numpy()) if len(counts) else np.nan,
            "pct_female": (capped["sex"] == "female").mean() * 100,
            "pct_male": (capped["sex"] == "male").mean() * 100,
        })
    return pd.DataFrame(filas)


def main():
    parser = argparse.ArgumentParser(description="Build the speaker-aware sample manifest.")
    parser.add_argument(
        "--clips-dir", default=None,
        help="Directory of available mp3s. Defaults to audios_filtrados; pass the "
             "full corpus clips directory once it has been downloaded.",
    )
    parser.add_argument(
        "--all-corpus", action="store_true",
        help="Ignore what is on disk and select from the entire corpus. Use this "
             "to produce the extraction list for the corpus tarball.",
    )
    parser.add_argument(
        "--dialects", nargs="+", default=None,
        help=f"Dialects to include. Options: {list(config.DIALECTS)}. "
             f"Defaults to the full study design: {config.ALL_DIALECTS}",
    )
    parser.add_argument(
        "--cap", type=int, default=None,
        help="Trim the manifest to this many clips per speaker. Needed when "
             "selecting from the full corpus, where keeping every clip would mean "
             "extracting tens of thousands of files to no benefit.",
    )
    args = parser.parse_args()

    banner(log, "STAGE 1 - SPEAKER-AWARE SAMPLING")

    if args.all_corpus:
        available = None
        log.info("Selecting from the full corpus (no on-disk restriction).")
    else:
        clips_dir = config.AUDIO_DIR if args.clips_dir is None else Path(args.clips_dir)
        available = {p.name for p in clips_dir.glob("*.mp3")}
        log.info(f"Restricting to {len(available):,} clips found in {clips_dir.name}/")

    df = build_manifest(available=available, dialects=args.dialects)

    if args.cap:
        before = len(df)
        df = df[df["speaker_clip_rank"] <= args.cap]
        log.info(f"Capped at {args.cap} clips per speaker: {len(df):,} of {before:,} clips")

    banner(log, "SAMPLE DESIGN")
    log.info(f"Total clips in manifest: {len(df):,}")
    log.info(f"Total speakers:          {df['client_id'].nunique():,}\n")

    design = describe(df, config.MAX_CLIPS_PER_SPEAKER)
    log.info(design.round(2).to_string(index=False))

    log.info("\nClips per speaker, before capping:")
    for dialect, sub in df.groupby("dialecto"):
        counts = sub["client_id"].value_counts()
        log.info(
            f"  {dialect:>5}: min {counts.min()}, median {counts.median():.0f}, "
            f"max {counts.max()}  |  speakers with 1 clip: {(counts == 1).sum()}"
        )

    log.info("\nSex by dialect (all clips):")
    log.info("  " + pd.crosstab(df["dialecto"], df["sex"]).to_string().replace("\n", "\n  "))

    # el equilibrio de sexo importa por hablante, no por clip, porque el hablante
    # es la unidad del analisis
    speakers = df.drop_duplicates("client_id")
    log.info("\nSex by dialect (unique speakers - the number that actually matters):")
    log.info("  " + pd.crosstab(speakers["dialecto"], speakers["sex"]).to_string().replace("\n", "\n  "))

    df.to_csv(config.SAMPLE_CSV, index=False)
    log.info(f"\n  -> {config.SAMPLE_CSV.relative_to(config.ROOT)}")
    save_table(design, "s1_sample_design", log)

    n_sur_speakers = df[df["dialecto"] == "sur"]["client_id"].nunique()
    if n_sur_speakers < 40:
        log.info(f"\nNOTE: only {n_sur_speakers} Sur speakers available. See the power analysis in stage 6.")


if __name__ == "__main__":
    main()
