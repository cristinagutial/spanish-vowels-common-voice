"""
Stage 3 - Forced alignment with MFA.

MFA needs its own conda environment, so this calls it from the command line. The
output turns every clip into a list of phonemes with their start and end times.

Writes work/mfa_aligned/ and tables/s3_alignment_report.csv.
"""

import shutil
import subprocess
import sys

import pandas as pd

from . import config
from .utils import banner, get_logger, save_table

log = get_logger(__name__)


def run_mfa(extra_args=None):
    """Run `mfa align` inside its conda environment."""
    command = [
        "conda", "run", "--no-capture-output", "-n", config.MFA_ENV,
        "mfa", "align",
        str(config.MFA_CORPUS_DIR),
        config.MFA_DICTIONARY,
        config.MFA_ACOUSTIC,
        str(config.MFA_ALIGNED_DIR),
        "--num_jobs", str(config.MFA_NUM_JOBS),
        "--clean",
        "--overwrite",
        "--output_format", "long_textgrid",
        "--use_mp",
    ]
    command += extra_args or []

    log.info("Running:\n  " + " ".join(command) + "\n")
    result = subprocess.run(command, text=True)
    return result.returncode


def report(manifest):
    """Compare the TextGrids produced against the clips submitted."""
    grids = sorted(config.MFA_ALIGNED_DIR.rglob("*.TextGrid"))
    aligned_stems = {g.stem for g in grids}

    manifest = manifest.copy()
    manifest["stem"] = manifest["path"].str.replace(".mp3", "", regex=False)
    manifest["aligned"] = manifest["stem"].isin(aligned_stems)

    log.info(f"\nTextGrids written: {len(grids)} for {len(manifest)} clips submitted")

    by_dialect = manifest.groupby("dialecto")["aligned"].agg(["sum", "count"])
    by_dialect["rate"] = by_dialect["sum"] / by_dialect["count"]
    log.info("\nAlignment success by dialect:")
    for dialect, row in by_dialect.iterrows():
        log.info(f"  {dialect:>5}: {int(row['sum']):3d} / {int(row['count']):3d}  ({row['rate']:.1%})")

    failed = manifest[~manifest["aligned"]]
    if len(failed):
        log.info(f"\n{len(failed)} clips failed to align. First few:")
        for row in failed.head(5).itertuples():
            log.info(f"  {row.path}: {row.sentence[:60]}")
        # cuando falla suele ser por una palabra que no esta en el diccionario o
        # porque la transcripcion no coincide con el audio; son fallos del corpus
        log.info("  Usually an out-of-vocabulary word, or a transcript that does not match the audio.")

    return manifest[["path", "client_id", "dialecto", "sex", "age", "aligned"]]


def main():
    banner(log, "STAGE 3 - FORCED ALIGNMENT")

    if not config.MFA_CORPUS_DIR.exists() or not any(config.MFA_CORPUS_DIR.iterdir()):
        log.info("No prepared corpus found. Run stage 2 first.")
        sys.exit(1)

    if shutil.which("conda") is None:
        log.info("conda not found on PATH; cannot reach the MFA environment.")
        sys.exit(1)

    # borro los TextGrids anteriores antes de alinear. El --clean de MFA solo
    # limpia su base de datos, no lo que ya escribio, y como los nombres de
    # hablante se reasignan en cada run, un TextGrid viejo puede acabar en una
    # carpeta que ahora es de otra persona (y el stage 4 busca el audio por ahi)
    if config.MFA_ALIGNED_DIR.exists():
        stale = list(config.MFA_ALIGNED_DIR.rglob("*.TextGrid"))
        if stale:
            log.info(f"Removing {len(stale)} TextGrids from a previous run")
            shutil.rmtree(config.MFA_ALIGNED_DIR)

    n_wav = len(list(config.MFA_CORPUS_DIR.rglob("*.wav")))
    n_speakers = len([d for d in config.MFA_CORPUS_DIR.iterdir() if d.is_dir()])
    log.info(f"Aligning {n_wav} clips from {n_speakers} speakers")
    log.info(f"  acoustic model: {config.MFA_ACOUSTIC}")
    log.info(f"  dictionary:     {config.MFA_DICTIONARY}\n")

    code = run_mfa()
    if code != 0:
        log.info(f"\nMFA exited with status {code}.")

    if not config.MFA_ALIGNED_DIR.exists():
        log.info("No output directory was produced; alignment did not run.")
        sys.exit(1)

    manifest = pd.read_csv(config.SAMPLE_CSV)
    result = report(manifest)
    save_table(result, "s3_alignment_report", log)


if __name__ == "__main__":
    main()
