"""
Formant track check: a sample of vowels to look at by eye and to measure by hand in Praat.

Not a pipeline stage. Picks 20 random stressed vowels (4 per vowel) plus the cases most
likely to be wrong, draws each one's spectrogram with the Burg tracks on top, and writes
a table with empty columns for hand measurements and a Praat script that opens each vowel.

    ./speech/bin/python -m src.check_tracks

Writes output/track_check/.
"""

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import parselmouth

from . import config
from .utils import banner, get_logger

log = get_logger(__name__)

OUT_DIR = config.OUTPUT / "track_check"
PER_VOWEL = 4
PER_EDGE = 3
PANELS_PER_SHEET = 6
CONTEXT = 0.06          # segundos de contexto a cada lado de la vocal
MAX_FREQ = 5500.0


def pick_sample(tokens, ceilings, cells):
    """Random stressed vowels, plus speakers at the ceiling edges and the extreme cells."""
    usable = tokens[tokens["lobanov_ok"] & (tokens["stressed"] == True)
                    & ~tokens["word"].isin(config.FUNCTION_WORDS)]
    rng = np.random.default_rng(config.SEED)
    picks = []

    for vowel in config.VOWELS:
        sub = usable[usable["vowel"] == vowel]
        for i in rng.choice(len(sub), PER_VOWEL, replace=False):
            picks.append((sub.index[i], "random"))

    # los techos en el borde de la rejilla son los mas sospechosos
    for edge in (config.CEILING_GRID[0], config.CEILING_GRID[-1]):
        speakers = ceilings[(ceilings["ceiling_Hz"] == edge)
                            & (ceilings["ceiling_source"] != "sex")]["client_id"]
        speakers = [s for s in speakers if s in set(usable["client_id"])]
        for spk in rng.choice(speakers, PER_EDGE, replace=False):
            sub = usable[usable["client_id"] == spk]
            picks.append((sub.index[rng.integers(len(sub))], f"ceiling at {edge:.0f} Hz"))

    # las tres celdas con |z| > 4 del H1
    for _, cell in cells[(cells["F1"].abs() > 4) | (cells["F2"].abs() > 4)].iterrows():
        sub = usable[(usable["client_id"] == cell["client_id"]) & (usable["vowel"] == cell["vowel"])]
        picks.append((sub.index[rng.integers(len(sub))], "extreme H1 cell"))

    sample = tokens.loc[[i for i, _ in picks]].copy()
    sample["reason"] = [r for _, r in picks]
    return sample.reset_index(drop=True)


def draw_token(ax, sound, formant, row):
    """Spectrogram of the vowel with context, Burg F1-F3 dots and the measured midpoint."""
    t0 = max(0.0, row["start"] - CONTEXT)
    t1 = min(sound.duration, row["end"] + CONTEXT)
    part = sound.extract_part(from_time=t0, to_time=t1, preserve_times=True)
    spec = part.to_spectrogram(window_length=0.005, maximum_frequency=MAX_FREQ)
    db = 10 * np.log10(np.maximum(spec.values, 1e-12))
    ax.pcolormesh(spec.xs(), spec.ys(), db, vmin=db.max() - 60, cmap="Greys", shading="auto")

    times = np.arange(t0, t1, 0.005)
    for n, colour in [(1, "#d62728"), (2, "#1f77b4"), (3, "#2ca02c")]:
        values = [formant.get_value_at_time(n, t) for t in times]
        ax.scatter(times, values, s=4, color=colour)
    mid = (row["start"] + row["end"]) / 2
    ax.scatter([mid, mid], [row["F1_50"], row["F2_50"]], marker="x", s=60, color="orange",
               linewidths=2, zorder=5)
    ax.axvline(row["start"], color="orange", linewidth=0.8)
    ax.axvline(row["end"], color="orange", linewidth=0.8)
    ax.set_ylim(0, MAX_FREQ)
    ax.set_title(f"#{row['token_id']} /{row['vowel']}/ '{row['word']}' {row['dialecto']}  "
                 f"ceiling {row['ceiling_Hz']:.0f}\nF1 {row['F1_50']:.0f}  F2 {row['F2_50']:.0f}  "
                 f"({row['reason']})", fontsize=8, loc="left")
    ax.tick_params(labelsize=7)


def main():
    banner(log, "FORMANT TRACK CHECK")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    tokens = pd.read_csv(config.TOKENS_CSV)
    ceilings = pd.read_csv(config.TABLE_DIR / "s4_ceilings.csv")
    cells = pd.read_csv(config.TABLE_DIR / "s6_h1_cells.csv")
    speakers = pd.read_csv(config.DATA_DIR / "speaker_map.csv").set_index("client_id")["speaker"]

    sample = pick_sample(tokens, ceilings, cells)
    sample.insert(0, "token_id", range(1, len(sample) + 1))
    sample["speaker_dir"] = sample["client_id"].map(speakers)
    sample["wav"] = [str(config.MFA_CORPUS_DIR / d / f"{c}.wav")
                     for d, c in zip(sample["speaker_dir"], sample["clip"])]
    sample["textgrid"] = [str(config.MFA_ALIGNED_DIR / d / f"{c}.TextGrid")
                          for d, c in zip(sample["speaker_dir"], sample["clip"])]
    log.info(f"{len(sample)} tokens: " + ", ".join(
        f"{k} {v}" for k, v in sample["reason"].value_counts().items()))

    for first in range(0, len(sample), PANELS_PER_SHEET):
        chunk = sample.iloc[first:first + PANELS_PER_SHEET]
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        for ax in axes.flat[len(chunk):]:
            ax.axis("off")
        for ax, (_, row) in zip(axes.flat, chunk.iterrows()):
            sound = parselmouth.Sound(row["wav"])
            formant = sound.to_formant_burg(max_number_of_formants=config.MAX_N_FORMANTS,
                                            maximum_formant=row["ceiling_Hz"])
            draw_token(ax, sound, formant, row)
        fig.suptitle("Burg tracks (red F1, blue F2, green F3); orange x = value used in the analysis",
                     fontsize=10)
        fig.tight_layout()
        path = OUT_DIR / f"sheet_{first // PANELS_PER_SHEET + 1}.png"
        fig.savefig(path, dpi=130)
        plt.close(fig)
        log.info(f"  -> {path.relative_to(config.ROOT)}")

    columns = ["token_id", "reason", "dialecto", "sex", "vowel", "word", "clip", "start", "end",
               "ceiling_Hz", "F1_50", "F2_50", "wav", "textgrid"]
    out = sample[columns].copy()
    for col in ["claude_check", "claude_note", "F1_praat", "F2_praat", "cristina_check", "cristina_note"]:
        out[col] = ""
    out.to_csv(OUT_DIR / "track_check_sample.csv", index=False)
    out[["token_id", "wav", "textgrid", "start", "end", "ceiling_Hz"]].to_csv(
        OUT_DIR / "track_check_sample.tsv", sep="\t", index=False)
    log.info(f"  -> {(OUT_DIR / 'track_check_sample.csv').relative_to(config.ROOT)}")


if __name__ == "__main__":
    main()
