"""
Stage 4 - Formant measurement, vowel by vowel.

Walks the TextGrids and writes one row per vowel: F1/F2/B1/B2 at three points,
duration, F0, HNR, lexical stress and what comes after the vowel.

Writes data/vowel_tokens_raw.csv.
"""

import multiprocessing as mp
import re
import unicodedata
import warnings

import numpy as np
import pandas as pd
import parselmouth
from parselmouth.praat import call
from praatio import textgrid

from . import config
from .utils import banner, get_logger

log = get_logger(__name__)

# para decidir si una /s/ va en coda: todo lo que no sea vocal ni semiconsonante
# lo cuento como consonante
GLIDES = {"j", "w"}
NON_PHONE = {"spn", "sil", "sp", ""}

# palabras atonas: no llevan acento aunque las reglas de ortografia se lo darian
CLITICS = {
    "el", "la", "los", "las", "un", "una", "unos", "unas", "lo",
    "de", "del", "al", "a", "en", "con", "por", "para", "sin", "so",
    "que", "se", "le", "les", "me", "te", "nos", "os",
    "y", "e", "o", "u", "ni", "si",
    "su", "sus", "mi", "mis", "tu", "tus",
}

ACCENTED = {"á": "a", "é": "e", "í": "i", "ó": "o", "ú": "u"}
ORTHO_VOWELS = "aeiouáéíóúü"


def _vowel_groups(word):
    """Runs of orthographic vowels, as (position, text) pairs."""
    return [(m.start(), m.group()) for m in re.finditer(f"[{ORTHO_VOWELS}]+", word)]


def stressed_nucleus_index(word, n_nuclei):
    """
    Which vowel of the word carries the stress, as a 0-based index.

    Spanish spelling marks stress unambiguously, so this applies the rules rather
    than guessing. The syllable count comes from the aligner. Returns None when
    the spelling and the phones disagree about how many syllables there are.
    """
    if n_nuclei < 1:
        return None
    if n_nuclei == 1:
        return 0

    groups = _vowel_groups(word)

    # si hay tilde, manda la tilde
    for group_idx, (_, text) in enumerate(groups):
        if any(ch in ACCENTED for ch in text):
            # solo me fio si los grupos de vocales y los nucleos cuadran en numero
            return group_idx if len(groups) == n_nuclei else None

    # si no, las reglas de siempre: llanas si acaban en vocal, -n o -s, y agudas
    # en los demas casos
    last = word[-1] if word else ""
    return n_nuclei - 2 if last in "aeiouáéíóú ns" else n_nuclei - 1


def is_consonant(label):
    return label not in NON_PHONE and label not in GLIDES and label not in config.VOWELS


def coda_s_context(phones, i, word_end_index):
    """
    True when the vowel at index `i` is followed by an /s/ in coda.

    An /s/ is in the coda when it is word-final or followed by a consonant. An
    /s/ before a vowel is a syllable onset ('ca-sa') and is not the context I want.
    """
    if i + 1 >= len(phones) or phones[i + 1][2] != "s":
        return False
    if i + 1 == word_end_index:          # /s/ final de palabra
        return True
    if i + 2 < len(phones):
        return is_consonant(phones[i + 2][2])
    return True


def ceiling_for(sex):
    return {
        "male": config.CEILING_MALE,
        "female": config.CEILING_FEMALE,
    }.get(sex, config.CEILING_UNKNOWN)


def _safe(fn, *args):
    """Praat returns NaN or raises on undefined frames; normalise everything to NaN."""
    try:
        value = fn(*args)
        return float(value) if value is not None and np.isfinite(value) else np.nan
    except Exception:
        return np.nan


def _process_one(job):
    """Measure every vowel in one aligned clip."""
    grid_path, wav_path, meta = job
    rows = []
    try:
        tg = textgrid.openTextgrid(str(grid_path), includeEmptyIntervals=False)
        phone_tier = tg.getTier("phones")
        word_tier = tg.getTier("words")

        phones = [(iv.start, iv.end, iv.label) for iv in phone_tier.entries]
        words = [(iv.start, iv.end, iv.label) for iv in word_tier.entries]
        if not phones or not words:
            return rows, f"{grid_path.stem}: empty tiers"

        sound = parselmouth.Sound(str(wav_path))
        ceiling = ceiling_for(meta["sex"])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            formant = sound.to_formant_burg(
                max_number_of_formants=config.MAX_N_FORMANTS,
                maximum_formant=ceiling,
            )
            pitch = sound.to_pitch()
            harmonicity = sound.to_harmonicity_cc()
            intensity = sound.to_intensity()

        # SNR aproximado del clip: los frames que estan a menos de 25 dB del mas
        # fuerte los cuento como voz y el resto como fondo, y el SNR es la
        # diferencia entre las dos medias. Si cogiera percentiles a secas estaria
        # midiendo lo hondos que son los silencios y no el ruido de fondo
        levels = intensity.values[0]
        levels = levels[np.isfinite(levels)]
        if levels.size:
            threshold = np.percentile(levels, 95) - 25.0
            speech = levels[levels >= threshold]
            noise = levels[levels < threshold]
            snr_db = float(speech.mean() - noise.mean()) if noise.size and speech.size else np.nan
        else:
            snr_db = np.nan

        # a cada fonema le asigno la palabra en la que esta, para luego poder
        # saber el acento y si es final de palabra
        word_of_phone = []
        for start, end, _ in phones:
            mid = (start + end) / 2
            match = next((w for w in words if w[0] <= mid <= w[1]), None)
            word_of_phone.append(match)

        for i, (start, end, label) in enumerate(phones):
            if label not in config.VOWELS:
                continue
            duration = end - start
            if duration < config.MIN_VOWEL_DURATION:
                continue

            word = word_of_phone[i]
            word_text = word[2] if word else ""
            word_text_norm = unicodedata.normalize("NFC", word_text.lower())

            # posiciones de los fonemas de esta palabra, para localizar el nucleo
            # y donde acaba
            same_word = [k for k, w in enumerate(word_of_phone) if w is word] if word else [i]
            nuclei = [k for k in same_word if phones[k][2] in config.VOWELS]
            nucleus_pos = nuclei.index(i) if i in nuclei else None
            word_end_index = same_word[-1] if same_word else i

            stress_idx = stressed_nucleus_index(word_text_norm, len(nuclei))
            if word_text_norm in CLITICS:
                stressed = False
            elif stress_idx is None or nucleus_pos is None:
                stressed = None
            else:
                stressed = (nucleus_pos == stress_idx)

            row = {
                "clip": grid_path.stem,
                "client_id": meta["client_id"],
                "dialecto": meta["dialecto"],
                "sex": meta["sex"],
                "age": meta["age"],
                "vowel": label,
                "start": start,
                "end": end,
                "duration": duration,
                "word": word_text_norm,
                "n_syllables": len(nuclei),
                "nucleus_pos": nucleus_pos,
                "stressed": stressed,
                "prev_phone": phones[i - 1][2] if i > 0 else "",
                "next_phone": phones[i + 1][2] if i + 1 < len(phones) else "",
                "word_final_vowel": i == word_end_index,
                "coda_s": coda_s_context(phones, i, word_end_index),
                "ceiling_Hz": ceiling,
                "snr_db": snr_db,
            }

            for proportion in config.MEASURE_POINTS:
                t = start + proportion * duration
                tag = f"{int(proportion * 100)}"
                row[f"F1_{tag}"] = _safe(formant.get_value_at_time, 1, t)
                row[f"F2_{tag}"] = _safe(formant.get_value_at_time, 2, t)
                row[f"B1_{tag}"] = _safe(formant.get_bandwidth_at_time, 1, t)
                row[f"B2_{tag}"] = _safe(formant.get_bandwidth_at_time, 2, t)

            midpoint = start + 0.5 * duration
            row["F0_Hz"] = _safe(pitch.get_value_at_time, midpoint)

            # cuando la armonicidad no esta definida Praat pone -200 dB en vez de
            # un valor vacio. Si eso entra en la media, el hablante sale
            # ruidosisimo solo porque tenia frames sordos
            hnr = _safe(call, harmonicity, "Get value at time...", midpoint, "linear")
            row["hnr_db"] = hnr if np.isfinite(hnr) and hnr > -100 else np.nan
            rows.append(row)

        return rows, ""
    except Exception as exc:
        return rows, f"{grid_path.stem}: {exc}"


def main():
    banner(log, "STAGE 4 - PER-VOWEL FORMANT EXTRACTION")

    manifest = pd.read_csv(config.SAMPLE_CSV)
    manifest["stem"] = manifest["path"].str.replace(".mp3", "", regex=False)
    meta_by_stem = manifest.set_index("stem").to_dict("index")

    grids = sorted(config.MFA_ALIGNED_DIR.rglob("*.TextGrid"))
    log.info(f"{len(grids)} aligned clips")

    jobs = []
    for grid in grids:
        meta = meta_by_stem.get(grid.stem)
        if meta is None:
            continue
        wav = config.MFA_CORPUS_DIR / grid.parent.name / f"{grid.stem}.wav"
        if wav.exists():
            jobs.append((grid, wav, meta))

    log.info(f"Measuring vowels in {len(jobs)} clips "
             f"(ceiling {config.CEILING_MALE:.0f}/{config.CEILING_FEMALE:.0f} Hz by sex)...")

    with mp.Pool(processes=max(1, mp.cpu_count() - 1)) as pool:
        results = pool.map(_process_one, jobs, chunksize=8)

    rows, errors = [], []
    for r, err in results:
        rows.extend(r)
        if err:
            errors.append(err)

    df = pd.DataFrame(rows)
    log.info(f"\nExtracted {len(df):,} vowel tokens from {df['clip'].nunique()} clips")
    if errors:
        log.info(f"  {len(errors)} clips raised errors; first few:")
        for e in errors[:5]:
            log.info(f"    {e}")

    log.info(f"\nTokens per vowel:")
    log.info("  " + df["vowel"].value_counts().to_string().replace("\n", "\n  "))

    log.info(f"\nTokens per dialect:")
    log.info("  " + df["dialecto"].value_counts().to_string().replace("\n", "\n  "))

    log.info(f"\nVowel x dialect:")
    log.info("  " + pd.crosstab(df["vowel"], df["dialecto"]).to_string().replace("\n", "\n  "))

    stress_known = df["stressed"].notna()
    log.info(f"\nStress resolved for {stress_known.sum():,} of {len(df):,} tokens "
             f"({stress_known.mean():.1%}); {df['stressed'].sum():.0f} stressed")
    log.info(f"Coda-/s/ context: {df['coda_s'].sum():,} tokens "
             f"({df['coda_s'].mean():.1%})")
    log.info(f"Median vowel duration: {df['duration'].median()*1000:.0f} ms")

    df.to_csv(config.TOKENS_RAW_CSV, index=False)
    log.info(f"\n  -> {config.TOKENS_RAW_CSV.relative_to(config.ROOT)}")


if __name__ == "__main__":
    main()
