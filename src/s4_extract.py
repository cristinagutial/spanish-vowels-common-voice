"""
Stage 4 - Formant measurement, vowel by vowel.

Walks the TextGrids and writes one row per vowel: F1 and F2 at the midpoint,
duration, lexical stress and what comes after the vowel. The formant ceiling is
chosen per speaker, by trying a grid of ceilings and keeping the one with the
least spread inside each vowel.

Writes data/vowel_tokens_raw.csv and the s4_ceilings table.
"""

import multiprocessing as mp
import re
import unicodedata
import warnings

import numpy as np
import pandas as pd
import parselmouth
from praatio import textgrid

from . import config
from .utils import banner, get_logger, save_table

log = get_logger(__name__)

# todo lo que no sea vocal ni semiconsonante lo cuento como consonante
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


def following_context(phones, k):
    """What comes after phone k: 'pause', 'vowel', 'consonant' or 'other'."""
    if k + 1 >= len(phones):
        return "pause"
    # los silencios se quitan al leer el TextGrid, asi que una pausa es un hueco
    if phones[k + 1][0] - phones[k][1] > config.PAUSE_GAP:
        return "pause"
    label = phones[k + 1][2]
    if label in ("sil", "sp", ""):
        return "pause"
    if label == "spn":
        return "other"
    if label in config.VOWELS or label in GLIDES:
        return "vowel"
    return "consonant"


def final_s_context(phones, i, word_end_index):
    """
    What follows the /s/ when vowel i is followed by a word-final /s/ ('casas').

    None when the vowel is not followed by a word-final /s/.
    """
    if i + 1 != word_end_index or phones[i + 1][2] != "s":
        return None
    return following_context(phones, i + 1)


def s_position(phones, k, word_final):
    """
    Syllable position of the /s/ at index k: 'onset', 'coda' or 'prevocalic_final'.

    Coda means before a consonant or a pause. A word-final /s/ before a vowel
    ('los amigos') is resyllabified in most varieties, so it gets its own label
    and stays out of the coda/onset test. None when what follows is unknown.
    """
    after = following_context(phones, k)
    if after in ("consonant", "pause"):
        return "coda"
    if after == "vowel":
        return "prevocalic_final" if word_final else "onset"
    return None


def ceiling_for(sex):
    return {
        "male": config.CEILING_MALE,
        "female": config.CEILING_FEMALE,
    }.get(sex, config.CEILING_UNKNOWN)


def best_ceiling(f1, f2, vowels, ceilings):
    """
    The ceiling with the least within-vowel spread, or None with too little data.

    f1 and f2 have one row per token and one column per ceiling. A wrong ceiling
    merges or splits formants on some tokens, and that shows up as spread. A track
    stuck at the same F2 for every vowel also has little spread, so stage 5 drops
    speakers whose vowel space has collapsed.
    """
    f1 = np.asarray(f1, dtype=float)
    f2 = np.asarray(f2, dtype=float)
    vowels = np.asarray(vowels)

    # solo los tokens que se pueden medir con todos los techos, para que cada
    # techo se juzgue sobre los mismos datos
    complete = np.all(np.isfinite(f1) & np.isfinite(f2) & (f1 > 0) & (f2 > 0), axis=1)
    f1, f2, vowels = f1[complete], f2[complete], vowels[complete]

    usable = [v for v in config.VOWELS if (vowels == v).sum() >= config.MIN_TOKENS_FOR_CEILING]
    if len(usable) < config.MIN_VOWELS_FOR_CEILING:
        return None

    cost = np.zeros(f1.shape[1])
    for v in usable:
        mask = vowels == v
        cost += np.log(f1[mask]).var(axis=0, ddof=1) + np.log(f2[mask]).var(axis=0, ddof=1)
    return float(np.asarray(ceilings)[int(np.argmin(cost))])


def _safe(fn, *args):
    """Praat returns NaN or raises on undefined frames; normalise everything to NaN."""
    try:
        value = fn(*args)
        return float(value) if value is not None and np.isfinite(value) else np.nan
    except Exception:
        return np.nan


def _read_vowels(grid_path):
    """Vowel tokens of one TextGrid, with word, stress and context, but no formants yet."""
    tg = textgrid.openTextgrid(str(grid_path), includeEmptyIntervals=False)
    phones = [(iv.start, iv.end, iv.label) for iv in tg.getTier("phones").entries]
    words = [(iv.start, iv.end, iv.label) for iv in tg.getTier("words").entries]
    if not phones or not words:
        return []

    # a cada fonema le asigno la palabra en la que esta, para luego poder
    # saber el acento y si es final de palabra
    word_of_phone = []
    for start, end, _ in phones:
        mid = (start + end) / 2
        word_of_phone.append(next((w for w in words if w[0] <= mid <= w[1]), None))

    rows = []
    for i, (start, end, label) in enumerate(phones):
        if label not in config.VOWELS:
            continue
        duration = end - start
        if duration < config.MIN_VOWEL_DURATION:
            continue

        word = word_of_phone[i]
        word_text = unicodedata.normalize("NFC", (word[2] if word else "").lower())

        # posiciones de los fonemas de esta palabra, para localizar el nucleo
        # y donde acaba
        same_word = [k for k, w in enumerate(word_of_phone) if w is word] if word else [i]
        nuclei = [k for k in same_word if phones[k][2] in config.VOWELS]
        nucleus_pos = nuclei.index(i) if i in nuclei else None
        word_end_index = same_word[-1] if same_word else i

        stress_idx = stressed_nucleus_index(word_text, len(nuclei))
        if word_text in CLITICS:
            stressed = False
        elif stress_idx is None or nucleus_pos is None:
            stressed = None
        else:
            stressed = (nucleus_pos == stress_idx)

        after_s = final_s_context(phones, i, word_end_index)
        rows.append({
            "clip": grid_path.stem,
            "vowel": label,
            "start": start,
            "end": end,
            "duration": duration,
            "word": word_text,
            "n_syllables": len(nuclei),
            "nucleus_pos": nucleus_pos,
            "stressed": stressed,
            "prev_phone": phones[i - 1][2] if i > 0 else "",
            "next_phone": phones[i + 1][2] if i + 1 < len(phones) else "",
            "word_final_vowel": i == word_end_index,
            "following": following_context(phones, i),
            "final_s": after_s is not None,
            "after_s": after_s,
        })
    return rows


def _measure(sound, times, ceiling):
    """F1 and F2 at the given times with one ceiling."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        formant = sound.to_formant_burg(max_number_of_formants=config.MAX_N_FORMANTS,
                                        maximum_formant=ceiling)
    f1 = [_safe(formant.get_value_at_time, 1, t) for t in times]
    f2 = [_safe(formant.get_value_at_time, 2, t) for t in times]
    return np.array(f1), np.array(f2)


def _process_speaker(job):
    """Measure every vowel of one speaker with every ceiling, then keep the best one."""
    meta, clips = job
    grid = np.array(config.CEILING_GRID)
    rows, sounds, errors = [], [], []

    for grid_path, wav_path in clips:
        try:
            clip_rows = _read_vowels(grid_path)
            if clip_rows:
                sounds.append((parselmouth.Sound(str(wav_path)), len(rows), len(clip_rows)))
                rows.extend(clip_rows)
        except Exception as exc:
            errors.append(f"{grid_path.stem}: {exc}")

    if not rows:
        return [], None, errors

    mids = np.array([(r["start"] + r["end"]) / 2 for r in rows])
    f1 = np.full((len(rows), len(grid)), np.nan)
    f2 = np.full((len(rows), len(grid)), np.nan)
    for sound, first, n in sounds:
        times = mids[first:first + n]
        for j, ceiling in enumerate(grid):
            f1[first:first + n, j], f2[first:first + n, j] = _measure(sound, times, ceiling)

    vowels = np.array([r["vowel"] for r in rows])
    stressed = np.array([r["stressed"] is True for r in rows])

    # primero con las tonicas, que son las que menos varian; si no llegan, con todas
    ceiling, source = best_ceiling(f1[stressed], f2[stressed], vowels[stressed], grid), "stressed"
    if ceiling is None:
        ceiling, source = best_ceiling(f1, f2, vowels, grid), "all"
    if ceiling is None:
        ceiling, source = ceiling_for(meta["sex"]), "sex"

    def values_at(c):
        if c in grid:
            j = int(np.where(grid == c)[0][0])
            return f1[:, j], f2[:, j]
        out1, out2 = np.full(len(rows), np.nan), np.full(len(rows), np.nan)
        for sound, first, n in sounds:
            out1[first:first + n], out2[first:first + n] = _measure(sound, mids[first:first + n], c)
        return out1, out2

    best_f1, best_f2 = values_at(ceiling)
    # tambien con el techo por sexo del v2.0, para el analisis de sensibilidad
    sex_f1, sex_f2 = values_at(ceiling_for(meta["sex"]))

    for k, r in enumerate(rows):
        r.update({
            "client_id": meta["client_id"],
            "dialecto": meta["dialecto"],
            "sex": meta["sex"],
            "age": meta["age"],
            "ceiling_Hz": ceiling,
            "F1_50": best_f1[k],
            "F2_50": best_f2[k],
            "F1_50_sexceil": sex_f1[k],
            "F2_50_sexceil": sex_f2[k],
        })

    summary = {
        "client_id": meta["client_id"], "dialecto": meta["dialecto"], "sex": meta["sex"],
        "ceiling_Hz": ceiling, "ceiling_source": source,
        "n_tokens": len(rows), "n_stressed": int(stressed.sum()),
    }
    return rows, summary, errors


def main():
    banner(log, "STAGE 4 - PER-VOWEL FORMANTS, CEILING PER SPEAKER")

    manifest = pd.read_csv(config.SAMPLE_CSV)
    manifest["stem"] = manifest["path"].str.replace(".mp3", "", regex=False)
    meta_by_stem = manifest.set_index("stem").to_dict("index")

    grids = sorted(config.MFA_ALIGNED_DIR.rglob("*.TextGrid"))
    log.info(f"{len(grids)} aligned clips")

    by_speaker = {}
    for grid in grids:
        meta = meta_by_stem.get(grid.stem)
        if meta is None:
            continue
        wav = config.MFA_CORPUS_DIR / grid.parent.name / f"{grid.stem}.wav"
        if wav.exists():
            by_speaker.setdefault(meta["client_id"], (meta, []))[1].append((grid, wav))

    log.info(f"{len(by_speaker)} speakers, {len(config.CEILING_GRID)} ceilings from "
             f"{config.CEILING_GRID[0]:.0f} to {config.CEILING_GRID[-1]:.0f} Hz")

    with mp.Pool(processes=max(1, mp.cpu_count() - 1)) as pool:
        results = pool.map(_process_speaker, list(by_speaker.values()), chunksize=4)

    rows, summaries, errors = [], [], []
    for r, s, e in results:
        rows.extend(r)
        if s is not None:
            summaries.append(s)
        errors.extend(e)

    df = pd.DataFrame(rows)
    ceilings = pd.DataFrame(summaries)

    log.info(f"\nExtracted {len(df):,} vowel tokens from {df['clip'].nunique()} clips")
    if errors:
        log.info(f"  {len(errors)} clips raised errors, e.g. {errors[0]}")

    log.info("\nCeiling source: " + ", ".join(
        f"{k} {v}" for k, v in ceilings["ceiling_source"].value_counts().items()))
    log.info("Median ceiling by declared sex (Hz):")
    log.info("  " + ceilings.groupby("sex")["ceiling_Hz"].median().to_string().replace("\n", "\n  "))
    log.info("Median ceiling by dialect (Hz):")
    log.info("  " + ceilings.groupby("dialecto")["ceiling_Hz"].median().to_string().replace("\n", "\n  "))

    stress_known = df["stressed"].notna()
    log.info(f"\nStress resolved for {stress_known.mean():.1%} of tokens; "
             f"{(df['stressed'] == True).sum():,} stressed")
    log.info(f"Vowels before a word-final /s/: {df['final_s'].sum():,}")

    df.to_csv(config.TOKENS_RAW_CSV, index=False)
    log.info(f"\n  -> {config.TOKENS_RAW_CSV.relative_to(config.ROOT)}")
    save_table(ceilings, "s4_ceilings", log)


if __name__ == "__main__":
    main()
