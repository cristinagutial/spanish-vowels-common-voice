"""
Stage 8 - H2a, the check that Southern speakers weaken coda /s/.

H2 only makes sense if the /s/ in these recordings really weakens in the South.
Aspiration lowers the spectral centre of gravity, so coda /s/ should have a lower
CoG in Sur than in Norte, and onset /s/ should not differ.

The /s/ is measured on the original 48 kHz mp3 (the 16 kHz WAVs stop at 8 kHz,
below much of the /s/ energy), on the middle 50% of the segment and with a
1-11 kHz band, so voicing and low-frequency noise do not pull the CoG down.

Writes data/sibilant_tokens.csv and the s8_ tables.
"""

import multiprocessing as mp
import warnings

import numpy as np
import pandas as pd
import parselmouth
import statsmodels.formula.api as smf
from parselmouth.praat import call
from praatio import textgrid

from . import config
from .s4_extract import s_position
from .utils import banner, compare, get_logger, holm_correction, save_table

log = get_logger(__name__)


def centre_of_gravity(sound, start, end):
    """CoG of the middle part of a segment, inside the S_BAND frequency band."""
    duration = end - start
    t1 = start + config.S_WINDOW[0] * duration
    t2 = start + config.S_WINDOW[1] * duration
    part = sound.extract_part(from_time=t1, to_time=t2,
                              window_shape=parselmouth.WindowShape.HANNING,
                              relative_width=1.0, preserve_times=False)
    spectrum = part.to_spectrum()
    low, high = config.S_BAND
    call(spectrum, "Filter (pass Hann band)", low, high, 100.0)
    return float(spectrum.get_centre_of_gravity(power=2.0))


def _measure_clip(job):
    """Measure every /s/ in one clip, with its syllable position."""
    grid_path, mp3_path, meta = job
    rows = []
    try:
        tg = textgrid.openTextgrid(str(grid_path), includeEmptyIntervals=False)
        phones = [(iv.start, iv.end, iv.label) for iv in tg.getTier("phones").entries]
        words = [(iv.start, iv.end, iv.label) for iv in tg.getTier("words").entries]
        sound = parselmouth.Sound(str(mp3_path))
        if sound.n_channels > 1:
            sound = sound.convert_to_mono()

        for k, (start, end, label) in enumerate(phones):
            if label != "s" or end - start < config.MIN_S_DURATION:
                continue
            word = next((w for w in words if w[0] <= (start + end) / 2 <= w[1]), None)
            word_final = word is not None and abs(end - word[1]) < 1e-6
            position = s_position(phones, k, word_final)
            if position is None:
                continue
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    cog = centre_of_gravity(sound, start, end)
            except Exception:
                continue
            if not np.isfinite(cog):
                continue
            rows.append({
                "clip": grid_path.stem,
                "client_id": meta["client_id"],
                "dialecto": meta["dialecto"],
                "sex": meta["sex"],
                "word": word[2] if word else "",
                "position": position,
                "duration": end - start,
                "cog_Hz": cog,
            })
        return rows, ""
    except Exception as exc:
        return rows, f"{grid_path.stem}: {exc}"


def extract_sibilants():
    manifest = pd.read_csv(config.SAMPLE_CSV)
    manifest["stem"] = manifest["path"].str.replace(".mp3", "", regex=False)
    meta_by_stem = manifest.set_index("stem").to_dict("index")

    jobs = []
    for grid in sorted(config.MFA_ALIGNED_DIR.rglob("*.TextGrid")):
        meta = meta_by_stem.get(grid.stem)
        mp3 = config.AUDIO_DIR / f"{grid.stem}.mp3"
        if meta is not None and mp3.exists():
            jobs.append((grid, mp3, meta))

    log.info(f"Measuring /s/ in {len(jobs)} clips...")
    with mp.Pool(processes=max(1, mp.cpu_count() - 1)) as pool:
        results = pool.map(_measure_clip, jobs, chunksize=8)

    errors = [e for _, e in results if e]
    if errors:
        log.info(f"  {len(errors)} clips raised errors, e.g. {errors[0]}")
    return pd.DataFrame([r for chunk, _ in results for r in chunk])


def speaker_cells(s_df):
    """One mean per speaker and position, from at least MIN_S_TOKENS_PER_CELL tokens."""
    cells = (
        s_df[s_df["position"].isin(["coda", "onset"])]
        .groupby(["client_id", "dialecto", "sex", "position"])
        .agg(cog=("cog_Hz", "mean"), dur=("duration", "mean"), n=("cog_Hz", "size"))
        .reset_index()
    )
    return cells[cells["n"] >= config.MIN_S_TOKENS_PER_CELL]


def main():
    banner(log, "STAGE 8 - H2a: CODA /s/ CHECK")

    s_df = extract_sibilants()
    s_df.to_csv(config.SIBILANTS_CSV, index=False)
    counts = pd.crosstab(s_df["dialecto"], s_df["position"])
    log.info(f"{len(s_df):,} /s/ tokens\n  " + counts.to_string().replace("\n", "\n  "))

    log.info("\nCoG percentiles by position (Hz):")
    pct = s_df.groupby("position")["cog_Hz"].quantile([0.05, 0.5, 0.95]).unstack().round(0)
    log.info("  " + pct.to_string().replace("\n", "\n  "))

    cells = speaker_cells(s_df)
    log.info(f"\n{len(cells)} speaker x position cells from {cells['client_id'].nunique()} speakers")

    banner(log, "NORTE vs SUR BY POSITION")
    rows = []
    for position in ["coda", "onset"]:
        for measure in ["cog", "dur"]:
            sub = cells[cells["position"] == position]
            rows.append(dict(position=position, measure=measure,
                             **compare(sub[sub["dialecto"] == "norte"][measure],
                                       sub[sub["dialecto"] == "sur"][measure], ("norte", "sur"))))
    by_position = pd.DataFrame(rows)
    by_position["p_holm"] = holm_correction(by_position["p"])
    for _, r in by_position.iterrows():
        log.info(f"  {r['position']:<6} {r['measure']:<4} Norte {r['mean_norte']:>8.3f}  "
                 f"Sur {r['mean_sur']:>8.3f}  d = {r['d']:+.2f}  p_holm = {r['p_holm']:.4f}")

    # la interaccion es la prediccion de verdad: diferencia en coda y no en ataque
    model_data = cells[cells["dialecto"].isin(config.CORE_DIALECTS)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = smf.mixedlm("cog ~ C(dialecto) * C(position, Treatment('coda')) + C(sex)",
                          model_data, groups=model_data["client_id"]).fit(reml=True)
    term = [t for t in fit.params.index if ":" in t][0]
    interaction = pd.DataFrame([{"term": term, "estimate_Hz": fit.params[term],
                                 "std_error": fit.bse[term], "p": fit.pvalues[term],
                                 "n_speakers": model_data["client_id"].nunique()}])
    log.info(f"\n  dialect x position on CoG: {fit.params[term]:+.1f} Hz, p = {fit.pvalues[term]:.4f}")

    banner(log, "WEAKENING INDEX (onset CoG - coda CoG, per speaker)")
    wide = cells.pivot_table(index=["client_id", "dialecto"], columns="position", values="cog").dropna()
    wide["weakening"] = wide["onset"] - wide["coda"]
    wide = wide.reset_index()
    index_rows = []
    for dialect in config.ALL_DIALECTS:
        values = wide[wide["dialecto"] == dialect]["weakening"]
        index_rows.append({"dialecto": dialect, "mean_Hz": values.mean(),
                           "sd_Hz": values.std(ddof=1), "n_speakers": len(values)})
        log.info(f"  {dialect:<7} {values.mean():+7.0f} Hz  (n = {len(values)})")
    weak = pd.DataFrame(index_rows)
    sur_norte = compare(wide[wide["dialecto"] == "sur"]["weakening"],
                        wide[wide["dialecto"] == "norte"]["weakening"], ("sur", "norte"))
    log.info(f"  Sur vs Norte: d = {sur_norte['d']:+.2f}, p = {sur_norte['p']:.4g}")

    banner(log, "WRITING OUTPUTS")
    save_table(counts.reset_index(), "s8_position_counts", log)
    save_table(cells, "s8_sibilant_cells", log)
    save_table(by_position, "s8_sibilant_norte_sur", log)
    save_table(interaction, "s8_interaction", log)
    save_table(weak, "s8_weakening_index", log)
    save_table(wide, "s8_weakening_per_speaker", log)


if __name__ == "__main__":
    main()
