# North vs. South Peninsular Spanish — Vowel Space Analysis

Cristina Gutiérrez Alcalá · Speech Signal Processing · Prof. Wolfgang Wokurek

Acoustic comparison of Northern and Southern Peninsular Spanish, using the
dialect-labelled recordings of Mozilla Common Voice. Forced alignment gives me
the phoneme labels, and all the statistics are done with the speaker as the unit.

---

## What I found

**The two vowel spaces are indistinguishable at the speaker level.** F1, F2,
bandwidths, duration, vowel space area: nothing comes out significant once I
treat the speaker as the unit instead of the clip. 461 Northern and 186 Southern
speakers, nothing above |d| = 0.17, and the design could have detected d ≥ 0.24.

**It is not that the method is blunt.** México is in the study as a control group
and it does separate from both Peninsular varieties (mean |d| 2.5 times larger,
significant in 6 of 24 measures). So the null result is about Norte and Sur, not
about the pipeline.

**My earlier version of this analysis found huge effects and they were an
artifact.** Pooling clips without looking at who recorded them gave F1 *d* = 1.34
and B2 *d* = −1.39. Both came from two individual speakers, who between them had
54 % of the Northern and 80 % of the Southern samples. Take those two out and the
effects drop to *d* = 0.01 and −0.09. This is what `src/s9_pseudorep.py` does.

**One hypothesis does survive, and it survives well.** Southern speakers weaken
coda /s/: the spectral centre of gravity drops in codas (*d* = +0.56) but not in
onsets (*d* = +0.06), so the dialect × position interaction is significant
(*p* = 0.006). Against Norte and México pooled, the weakening index gives
*d* = +0.63, *p* < 0.0001, and the three varieties line up exactly as the
dialectology books say: Norte +359 Hz, México +836 Hz, Sur +1022 Hz.

The vowel before coda /s/ also moves (dialect × context on F1, *p* = 0.005), but
the effect is small and the direction needs care: both varieties close the vowel,
the South just closes it less.

These survive because they are within-speaker contrasts — each speaker is their
own control, so unequal clip counts cannot fake them.

The methodological point: global between-group comparisons on a volunteer corpus
are fragile, and targeted within-speaker contrasts are not.

---

## Setup

Two environments. The analysis runs in a virtualenv, and forced alignment needs
its own conda environment because MFA pulls in Kaldi.

```bash
./speech/bin/pip install -r requirements.txt
```

```bash
mamba create -y -n mfa -c conda-forge montreal-forced-aligner ffmpeg
conda run -n mfa mfa model download acoustic spanish_mfa
conda run -n mfa mfa model download dictionary spanish_mfa
```

It needs `validated.tsv` (the Common Voice 26.0 metadata) and `audios_corpus/`
(the sampled mp3s). `audios_filtrados/` is the original 903-clip sample and is
only used by stage 9. Neither the audio nor the TSVs are in the zip, they are too
big; this is how I rebuild the sample from the corpus tarball:

```bash
tar -xzf cv-corpus-26.0-2026-06-12-es.tar.gz --fast-read 'cv-corpus-*/es/validated.tsv'
./speech/bin/python -m src.s1_sample --all-corpus --dialects norte sur mexico --cap 8
tar -xzf cv-corpus-26.0-2026-06-12-es.tar.gz --strip-components=3 -C audios_corpus \
    -T output/data/extract_list.txt
```

The TSVs sit at the front of the archive, so `--fast-read` pulls them out in a
second instead of streaming the whole 48 GB.

## Running it

```bash
./speech/bin/python run_all.py
```

```bash
./speech/bin/python run_all.py --list
```

```bash
./speech/bin/python run_all.py --from 5
```

```bash
./speech/bin/python run_all.py --only 9
```

A full run takes about 15 minutes, most of it in stages 2–4 (conversion,
alignment, extraction) and stage 8 (cross-validation). Every stage saves its
output in `output/data/`, so I can re-run the later ones on their own.

## Tests

```bash
./speech/bin/python -m unittest discover -s tests -v
```

The main test builds vowels with formants I already know and checks that the
extractor gets them back. It also documents a real limitation: linear prediction
overestimates F1 for close vowels, by +13 % for /i/ and +34 % for /u/, because F1
gets close to F0 and the poles interfere. It happens the same in both dialects so
it cannot create a difference, but /u/ is the least reliable measurement here.

---

## The pipeline

| Stage | Module | What it does |
|-------|--------|--------------|
| 1 | `s1_sample` | Builds the clip manifest keyed on `client_id`, with a reproducible clip ranking inside each speaker so the cap can be applied later |
| 2 | `s2_prepare` | mp3 → 16 kHz mono WAV (peak-scaled so it does not clip) plus `.lab` transcripts, one folder per speaker for MFA |
| 3 | `s3_align` | Montreal Forced Aligner with `spanish_mfa`; 5,572 of 5,577 clips align |
| 4 | `s4_extract` | One row per vowel: F1/F2/B1/B2 at 25/50/75 %, duration, F0, HNR, stress, coda-/s/ context. Burg ceiling by sex |
| 5 | `s5_normalize` | Range filters, per-speaker per-vowel outlier rejection, Lobanov and Bark |
| 6 | `s6_stats` | Speaker-level tests, mixed models, Pillai, vowel space area, the bandwidth-vs-noise check, outgroup calibration, power |
| 7 | `s7_cluster` | k-means scored with ARI against the true vowel labels |
| 8 | `s8_classify` | Dialect and vowel classification, speaker-disjoint vs. speaker-leaking CV |
| 9 | `s9_pseudorep` | The speaker audit of my midterm analysis |
| 10 | `s10_codas` | Coda-/s/ aspiration and vowel opening, with México as an /s/-retaining control |
| 11 | `plots` | The report figures |

Paths, seed and thresholds are all in `src/config.py`. Nothing else hard-codes a
path or a magic number.

## The data

| | count |
|---|---|
| Corpus | Common Voice es 26.0 (2026-06-12) |
| Clips analysed | 5,577 (2,639 Norte, 1,199 Sur, 1,739 México) |
| Speakers | 957 (469 Norte, 188 Sur, 300 México) |
| Clips aligned | 5,572 (99.9 %) |
| Total audio | 484 minutes |
| Vowel tokens extracted | 100,053 |
| Tokens after QC | 89,958 (90 %) |
| Speakers that can be normalised | 906 |

What matters is the speaker counts, not the token counts, because they are what
bounds every inference. With 186 Southern and 461 Northern speakers the study has
80 % power for *d* ≥ 0.24. Clips are capped at 8 per speaker, which brought the
Gini coefficient of clips per speaker down from 0.85 to 0.18.

## What changed from the midterm

The old scripts (`extractor.py`, `extractor_v2.py`, `limpieza*.py`,
`analisis*.py`, `etiquetar_vocales.py`, `preprocesador.py`) are still here, and
stage 9 reproduces them as the "naive" baseline.

- **Sampling** by speaker instead of by clip, seeded, and multi-label accent
  strings are dropped instead of being forced into one dialect.
- **Measurement** of every vowel in every clip with a known identity, instead of
  one measurement per clip at the loudest frame with no idea which vowel it was.
  This is also what made vowel labelling possible: my earlier text-based attempt
  resolved 14 of 903 clips.
- **Normalisation**, which I did not have. Raw Hz mixes dialect with vocal tract
  length and the dialect cells are not balanced by sex.
- **Inference** at the speaker level everywhere, instead of t-tests over clips.
- **Clustering** scored with ARI against the true labels. Comparing silhouette
  between a 2-D and a 4-D feature set was not valid, because silhouette falls
  with dimensionality on its own. Silhouette still peaks at k = 3 here, same as
  before, but ARI peaks at k = 4 and the true number of vowels is 5 — so k = 3
  was a property of the metric and not of Spanish.
- **Bandwidth** re-checked with HNR and SNR as covariates. Once the speakers are
  in the model there is no B2 difference left to explain.
