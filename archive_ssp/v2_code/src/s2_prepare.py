"""
Stage 2 - Audio and transcripts for the aligner.

Converts every mp3 to 16 kHz mono WAV and writes the transcript next to it as a
.lab file. One folder per speaker, because MFA uses that to adapt to each voice.

Writes work/mfa_corpus/ and data/speaker_map.csv.
"""

import multiprocessing as mp
import re
import shutil
import unicodedata
from pathlib import Path

import pandas as pd
import parselmouth

from . import config
from .utils import banner, get_logger

log = get_logger(__name__)

# lo que MFA no deberia ver nunca en una transcripcion. Dejo las vocales con
# tilde y la ñ/ü, y todo lo demas que no sea letra pasa a ser un espacio
_PUNCT = re.compile(r"[^a-záéíóúüñ ]+")
_WHITESPACE = re.compile(r"\s+")


def clean_transcript(sentence):
    """
    Turn a Common Voice sentence into plain lowercase words.

    Accents are kept because the Spanish MFA dictionary uses them; stripping them
    would turn known words into unknown ones.
    """
    if not isinstance(sentence, str):
        return ""
    text = unicodedata.normalize("NFC", sentence).lower()
    text = text.replace("’", "").replace("'", "")
    text = _PUNCT.sub(" ", text)
    return _WHITESPACE.sub(" ", text).strip()


def _convert_one(job):
    """Read an mp3, downsample to 16 kHz mono, write the wav and the lab."""
    src, wav_path, lab_path, transcript = job
    try:
        sound = parselmouth.Sound(str(src))
        if sound.n_channels > 1:
            sound = sound.convert_to_mono()
        if sound.sampling_frequency != config.MFA_SAMPLE_RATE:
            sound = sound.resample(config.MFA_SAMPLE_RATE)
        # al remuestrear los picos se pueden pasar de rango y al guardarlos se
        # recortan, y ese recorte mete distorsion que estropea el espectro.
        # Bajar el pico es solo una ganancia, no mueve los formantes
        sound.scale_peak(0.99)
        sound.save(str(wav_path), parselmouth.SoundFileFormat.WAV)
        lab_path.write_text(transcript, encoding="utf-8")
        return (src.name, True, sound.duration, "")
    except Exception as exc:
        return (src.name, False, 0.0, str(exc))


def main():
    banner(log, "STAGE 2 - AUDIO AND TRANSCRIPT PREPARATION")

    manifest = pd.read_csv(config.SAMPLE_CSV)
    log.info(f"{len(manifest)} clips in the manifest")

    manifest["transcript"] = manifest["sentence"].apply(clean_transcript)
    empty = (manifest["transcript"] == "")
    if empty.any():
        log.info(f"  dropping {empty.sum()} clips with an empty transcript")
        manifest = manifest[~empty].copy()

    # nombres de carpeta cortos para cada hablante, guardando la equivalencia
    # para poder volver siempre al client_id de Common Voice
    speakers = sorted(manifest["client_id"].unique())
    speaker_map = {cid: f"spk{i:04d}" for i, cid in enumerate(speakers, start=1)}
    manifest["speaker"] = manifest["client_id"].map(speaker_map)

    pd.DataFrame(
        {"client_id": list(speaker_map), "speaker": list(speaker_map.values())}
    ).merge(
        manifest[["client_id", "dialecto", "sex", "age"]].drop_duplicates("client_id"),
        on="client_id", how="left",
    ).to_csv(config.DATA_DIR / "speaker_map.csv", index=False)

    # borro la carpeta entera y no solo los ficheros: los nombres spk0001... se
    # asignan en cada run, asi que una carpeta vieja puede tener un nombre que
    # ahora es de otro hablante
    if config.MFA_CORPUS_DIR.exists():
        log.info("  clearing previous MFA corpus directory")
        shutil.rmtree(config.MFA_CORPUS_DIR)

    jobs = []
    for row in manifest.itertuples():
        speaker_dir = config.MFA_CORPUS_DIR / row.speaker
        speaker_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(row.path).stem
        jobs.append((
            config.AUDIO_DIR / row.path,
            speaker_dir / f"{stem}.wav",
            speaker_dir / f"{stem}.lab",
            row.transcript,
        ))

    log.info(f"Converting {len(jobs)} clips to {config.MFA_SAMPLE_RATE} Hz mono WAV...")
    with mp.Pool(processes=max(1, mp.cpu_count() - 1)) as pool:
        results = pool.map(_convert_one, jobs, chunksize=16)

    ok = [r for r in results if r[1]]
    failed = [r for r in results if not r[1]]
    total_seconds = sum(r[2] for r in ok)

    log.info(f"  converted: {len(ok)}")
    log.info(f"  failed:    {len(failed)}")
    for name, _, _, err in failed[:5]:
        log.info(f"    {name}: {err}")
    log.info(f"  total audio: {total_seconds / 60:.1f} minutes across "
             f"{manifest['speaker'].nunique()} speaker directories")

    words = manifest["transcript"].str.split().explode()
    log.info(f"  transcripts: {len(words):,} word tokens, {words.nunique():,} types")
    log.info(f"\n  -> {config.MFA_CORPUS_DIR.relative_to(config.ROOT)}/")


if __name__ == "__main__":
    main()
