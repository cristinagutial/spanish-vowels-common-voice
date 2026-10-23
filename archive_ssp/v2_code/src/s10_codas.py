"""
Stage 10 - Coda /s/ in Southern Spanish.

Syllable-final /s/ weakens to [h] or disappears in the South, and the vowel
before it is said to open and lengthen. Only testable with alignment, since I
need to know where each /s/ is and whether it sits in a coda.

Two measurements: the /s/ itself, where aspiration lowers the spectral centre of
gravity and onset /s/ is the within-speaker control, and the vowel before it,
where compensation predicts a dialect x context interaction.

Writes the s10_ tables and figures/fig_codas.png.
"""

import multiprocessing as mp
import warnings

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import parselmouth
import statsmodels.formula.api as smf
from praatio import textgrid
from scipy import stats

from . import config
from .utils import banner, cohens_d, get_logger, holm_correction, save_table
from .s4_extract import is_consonant

log = get_logger(__name__)

MIN_S_DURATION = 0.020


def _measure_s(job):
    """Measure every /s/ in one clip, tagged as coda or onset."""
    grid_path, wav_path, meta = job
    rows = []
    try:
        tg = textgrid.openTextgrid(str(grid_path), includeEmptyIntervals=False)
        phones = [(iv.start, iv.end, iv.label) for iv in tg.getTier("phones").entries]
        words = [(iv.start, iv.end, iv.label) for iv in tg.getTier("words").entries]
        sound = parselmouth.Sound(str(wav_path))

        for i, (start, end, label) in enumerate(phones):
            if label != "s":
                continue
            duration = end - start
            if duration < MIN_S_DURATION:
                continue

            # coda si es final de palabra o le sigue consonante; ataque si le sigue vocal
            following = phones[i + 1][2] if i + 1 < len(phones) else ""
            word = next((w for w in words if w[0] <= (start + end) / 2 <= w[1]), None)
            word_final = word is not None and abs(end - word[1]) < 1e-6

            if not following or word_final:
                position = "coda"
            elif is_consonant(following):
                position = "coda"
            elif following in config.VOWELS or following in {"j", "w"}:
                position = "onset"
            else:
                continue

            preceded_by_vowel = i > 0 and phones[i - 1][2] in config.VOWELS

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    segment = sound.extract_part(from_time=start, to_time=end)
                    spectrum = segment.to_spectrum()
                    cog = spectrum.get_centre_of_gravity(power=2)
                    sd = spectrum.get_standard_deviation(power=2)
                    intensity = segment.get_rms()
            except Exception:
                continue

            if not np.isfinite(cog):
                continue

            rows.append({
                "clip": grid_path.stem,
                "client_id": meta["client_id"],
                "dialecto": meta["dialecto"],
                "sex": meta["sex"],
                "position": position,
                "duration": duration,
                "cog_Hz": float(cog),
                "spectral_sd": float(sd) if np.isfinite(sd) else np.nan,
                "rms": float(intensity),
                "preceded_by_vowel": preceded_by_vowel,
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
        if meta is None:
            continue
        wav = config.MFA_CORPUS_DIR / grid.parent.name / f"{grid.stem}.wav"
        if wav.exists():
            jobs.append((grid, wav, meta))

    log.info(f"Measuring /s/ segments in {len(jobs)} clips...")
    with mp.Pool(processes=max(1, mp.cpu_count() - 1)) as pool:
        results = pool.map(_measure_s, jobs, chunksize=8)

    rows = [r for chunk, _ in results for r in chunk]
    return pd.DataFrame(rows)


def analyse_sibilants(s_df):
    banner(log, "A. THE /s/ ITSELF")

    log.info(f"{len(s_df):,} /s/ tokens  "
             f"({(s_df['position'] == 'coda').sum():,} coda, "
             f"{(s_df['position'] == 'onset').sum():,} onset)\n")

    log.info("Token-level means (descriptive only - tokens are clustered by speaker):")
    desc = s_df.groupby(["dialecto", "position"])[["cog_Hz", "duration", "rms"]].agg(
        ["mean", "count"]
    ).round(3)
    log.info("  " + desc.to_string().replace("\n", "\n  "))

    # por hablante: una media por persona y posicion
    cells = (
        s_df.groupby(["client_id", "dialecto", "sex", "position"])
        .agg(cog=("cog_Hz", "mean"), dur=("duration", "mean"),
             rms=("rms", "mean"), n=("cog_Hz", "size"))
        .reset_index()
    )
    cells = cells[cells["n"] >= 3]

    log.info(f"\nSpeaker-level cells with >= 3 tokens: {len(cells)} "
             f"from {cells['client_id'].nunique()} speakers")

    rows = []
    for position in ["coda", "onset"]:
        sub = cells[cells["position"] == position]
        for measure in ["cog", "dur", "rms"]:
            a = sub[sub["dialecto"] == "norte"][measure].dropna()
            b = sub[sub["dialecto"] == "sur"][measure].dropna()
            if len(a) < 3 or len(b) < 3:
                continue
            t, p = stats.ttest_ind(a, b, equal_var=False)
            rows.append({
                "position": position, "measure": measure,
                "n_norte": len(a), "n_sur": len(b),
                "norte": a.mean(), "sur": b.mean(),
                "cohens_d": cohens_d(a, b), "p": p,
            })

    # el español de Mexico mantiene mucho la /s/, asi que deberia parecerse al
    # Norte. Si solo se debilita el Sur, es mucho mejor prueba que un contraste de
    # dos, donde no puedo distinguir si es que el Sur debilita o el Norte refuerza
    if "mexico" in set(cells["dialecto"]):
        log.info("\n  Coda-/s/ weakening index by variety "
                 "(onset CoG - coda CoG; larger = more weakening in codas):")
        wide = cells.pivot_table(index=["client_id", "dialecto"],
                                 columns="position", values="cog")
        if {"coda", "onset"}.issubset(wide.columns):
            wide["weakening"] = wide["onset"] - wide["coda"]
            for dialect in config.ALL_DIALECTS:
                values = wide.xs(dialect, level="dialecto")["weakening"].dropna() \
                    if dialect in wide.index.get_level_values("dialecto") else pd.Series(dtype=float)
                if len(values):
                    log.info(f"    {dialect:<7} {values.mean():>+8.0f} Hz  (n = {len(values)} speakers)")
            iberian = [d for d in ("norte", "mexico")
                       if d in wide.index.get_level_values("dialecto")]
            if "sur" in wide.index.get_level_values("dialecto") and iberian:
                sur_vals = wide.xs("sur", level="dialecto")["weakening"].dropna()
                other = pd.concat([wide.xs(d, level="dialecto")["weakening"].dropna()
                                   for d in iberian])
                if len(sur_vals) >= 3 and len(other) >= 3:
                    t, p = stats.ttest_ind(sur_vals, other, equal_var=False)
                    log.info(f"    Sur vs (Norte + México): d = {cohens_d(sur_vals, other):+.2f}, "
                             f"p = {p:.4f}")

    out = pd.DataFrame(rows)
    if len(out):
        out["p_holm"] = holm_correction(out["p"])
        log.info(f"\n  {'position':<8} {'measure':<8} {'Norte':>10} {'Sur':>10} {'d':>7} {'p_holm':>8}")
        for _, r in out.iterrows():
            mark = " *" if r["p_holm"] < config.ALPHA else ""
            log.info(f"  {r['position']:<8} {r['measure']:<8} {r['norte']:>10.1f} "
                     f"{r['sur']:>10.1f} {r['cohens_d']:>+7.2f} {r['p_holm']:>8.3f}{mark}")

    # la interaccion es la hipotesis de verdad: efecto en coda pero no en ataque
    log.info("\n  Interaction test: does the dialect difference depend on position?")
    model_data = cells.dropna(subset=["cog"])
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = smf.mixedlm("cog ~ C(dialecto) * C(position) + C(sex)",
                              model_data, groups=model_data["client_id"]).fit(reml=True)
        term = "C(dialecto)[T.sur]:C(position)[T.onset]"
        if term in fit.params.index:
            log.info(f"    dialect x position on spectral CoG: "
                     f"{fit.params[term]:+.1f} Hz, p = {fit.pvalues[term]:.3f}")
            log.info("    A significant interaction is what aspiration predicts.")
    except Exception as exc:
        log.info(f"    model did not converge ({exc})")

    return out, cells


def analyse_vowels():
    banner(log, "B. THE VOWEL BEFORE CODA /s/")
    log.info("Compensatory opening predicts a dialect x context interaction.\n")

    df = pd.read_csv(config.TOKENS_CSV)
    df = df[df["lobanov_ok"]].copy()
    df["context"] = np.where(df["coda_s"], "pre_coda_s", "elsewhere")

    log.info("Token counts by dialect and context:")
    log.info("  " + pd.crosstab(df["dialecto"], df["context"]).to_string().replace("\n", "\n  "))

    cells = (
        df.groupby(["client_id", "dialecto", "sex", "context"])
        .agg(F1_lob=("F1_50_lob", "mean"), F2_lob=("F2_50_lob", "mean"),
             duration=("duration", "mean"), n=("F1_50_lob", "size"))
        .reset_index()
    )
    cells = cells[cells["n"] >= 3]

    both = cells.groupby("client_id")["context"].nunique()
    complete = both[both == 2].index
    paired = cells[cells["client_id"].isin(complete)]
    log.info(f"\n{len(complete)} speakers contribute both contexts "
             f"({paired.groupby('dialecto')['client_id'].nunique().to_dict()})")

    rows = []
    for measure in ["F1_lob", "F2_lob", "duration"]:
        try:
            data = paired.dropna(subset=[measure])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = smf.mixedlm(f"{measure} ~ C(dialecto) * C(context)",
                                  data, groups=data["client_id"]).fit(reml=True)
            term = "C(dialecto)[T.sur]:C(context)[T.pre_coda_s]"
            rows.append({
                "measure": measure,
                "interaction": fit.params.get(term, np.nan),
                "p": fit.pvalues.get(term, np.nan),
                "n_speakers": data["client_id"].nunique(),
            })
        except Exception as exc:
            log.info(f"  {measure}: model did not converge ({exc})")

    out = pd.DataFrame(rows)
    if len(out):
        out["p_holm"] = holm_correction(out["p"].fillna(1.0))
        log.info("\n  dialect x context interactions:")
        for _, r in out.iterrows():
            verdict = "significant" if r["p_holm"] < config.ALPHA else "n.s."
            log.info(f"    {r['measure']:<10} {r['interaction']:>+8.4f}  "
                     f"p_holm = {r['p_holm']:.3f}  {verdict}")

    # contraste pareado dentro del hablante, que es la forma mas directa de mirarlo
    log.info("\n  Within-speaker shift from 'elsewhere' to 'pre-coda /s/':")
    wide = paired.pivot_table(index=["client_id", "dialecto"], columns="context",
                              values=["F1_lob", "duration"])
    for measure in ["F1_lob", "duration"]:
        if (measure, "pre_coda_s") not in wide.columns:
            continue
        delta = (wide[(measure, "pre_coda_s")] - wide[(measure, "elsewhere")]).dropna()
        grouped = delta.groupby("dialecto")
        norte = grouped.get_group("norte") if "norte" in grouped.groups else pd.Series(dtype=float)
        sur = grouped.get_group("sur") if "sur" in grouped.groups else pd.Series(dtype=float)
        if len(norte) >= 3 and len(sur) >= 3:
            t, p = stats.ttest_ind(norte, sur, equal_var=False)
            log.info(f"    {measure:<10} Norte {norte.mean():+.4f} (n={len(norte)})  "
                     f"Sur {sur.mean():+.4f} (n={len(sur)})  d = {cohens_d(norte, sur):+.2f}  p = {p:.3f}")

    return out, cells


def make_figure(s_cells, vowel_cells):
    # meto a Mexico si esta: como mantiene la /s/, deberia ir con el Norte y dejar
    # al Sur solo como el que debilita las codas
    dialects = [d for d in config.ALL_DIALECTS if d in set(s_cells["dialecto"])]
    span = 0.26 if len(dialects) > 2 else 0.18
    offsets = np.linspace(-span, span, len(dialects))
    width = 0.7 / max(len(dialects), 1)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    ax = axes[0]
    positions = ["onset", "coda"]
    for offset, dialect in zip(offsets, dialects):
        data = [s_cells[(s_cells["position"] == p) & (s_cells["dialecto"] == dialect)]["cog"].dropna()
                for p in positions]
        parts = ax.boxplot(data, positions=np.arange(len(positions)) + offset,
                           widths=width, patch_artist=True, showfliers=False)
        for box in parts["boxes"]:
            box.set_facecolor(config.COLORS[dialect])
            box.set_alpha(0.65)
    ax.set_xticks(range(len(positions)))
    ax.set_xticklabels([f"/s/ in {p}" for p in positions])
    ax.set_ylabel("Spectral centre of gravity (Hz)")
    ax.set_title("A. Aspiration lowers /s/ CoG", fontsize=11, loc="left")
    handles = [plt.Rectangle((0, 0), 1, 1, fc=config.COLORS[d], alpha=0.65)
               for d in dialects]
    ax.legend(handles, [d.capitalize() for d in dialects], fontsize=9)
    ax.grid(alpha=0.3, axis="y")

    ax = axes[1]
    for offset, dialect in zip(offsets, dialects):
        data = [s_cells[(s_cells["position"] == p) & (s_cells["dialecto"] == dialect)]["dur"].dropna() * 1000
                for p in positions]
        parts = ax.boxplot(data, positions=np.arange(len(positions)) + offset,
                           widths=width, patch_artist=True, showfliers=False)
        for box in parts["boxes"]:
            box.set_facecolor(config.COLORS[dialect])
            box.set_alpha(0.65)
    ax.set_xticks(range(len(positions)))
    ax.set_xticklabels([f"/s/ in {p}" for p in positions])
    ax.set_ylabel("Duration (ms)")
    ax.set_title("B. /s/ duration by position", fontsize=11, loc="left")
    ax.grid(alpha=0.3, axis="y")

    ax = axes[2]
    contexts = ["elsewhere", "pre_coda_s"]
    for offset, dialect in zip(offsets, dialects):
        data = [vowel_cells[(vowel_cells["context"] == c) & (vowel_cells["dialecto"] == dialect)]["F1_lob"].dropna()
                for c in contexts]
        parts = ax.boxplot(data, positions=np.arange(len(contexts)) + offset,
                           widths=width, patch_artist=True, showfliers=False)
        for box in parts["boxes"]:
            box.set_facecolor(config.COLORS[dialect])
            box.set_alpha(0.65)
    ax.set_xticks(range(len(contexts)))
    ax.set_xticklabels(["other contexts", "before coda /s/"])
    ax.set_ylabel("F1 (Lobanov z)")
    ax.set_title("C. Vowel opening before coda /s/", fontsize=11, loc="left")
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle("Testing the Andalusian coda-/s/ process", fontsize=13)
    plt.tight_layout()
    path = config.FIGURE_DIR / "fig_codas.png"
    plt.savefig(path, dpi=config.DPI)
    plt.close()
    log.info(f"\n  -> {path.relative_to(config.ROOT)}")


def main():
    banner(log, "STAGE 10 - THE ANDALUSIAN CODA-/s/ HYPOTHESIS")

    s_df = extract_sibilants()
    log.info(f"Extracted {len(s_df):,} /s/ tokens\n")
    s_df.to_csv(config.DATA_DIR / "sibilant_tokens.csv", index=False)

    s_results, s_cells = analyse_sibilants(s_df)
    vowel_results, vowel_cells = analyse_vowels()
    make_figure(s_cells, vowel_cells)

    save_table(s_results, "s10_sibilant_speaker_level", log)
    save_table(vowel_results, "s10_vowel_interactions", log)


if __name__ == "__main__":
    main()
