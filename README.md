# Northern vs. Southern Peninsular Spanish vowels — v3.0

Cristina Gutiérrez Alcalá · Experimental Phonetics (Katrin Schneider) · *Topics in Phonetics*

An acoustic study of stressed vowels and of vowels before word-final /s/ in Northern and
Southern Peninsular Spanish, with Mexican Spanish as a control, using the dialect-labelled
recordings of Mozilla Common Voice. Forced alignment gives the phone labels, and every test
uses the speaker as the unit.

v3.0 builds on the v2.0 pipeline from my Speech Signal Processing project. The design, the
hypotheses and the analysis choices are in [PLAN_v3.md](PLAN_v3.md), and every number is in
[RESULTS.md](RESULTS.md). Everything from the SSP project (midterm code, report, v2.0 outputs)
is in `archive_ssp/`.

---

## Hypotheses

- **H1.** Stressed vowels do not differ between Norte and Sur by more than *d* = ±0.30
  (equivalence test). México is the positive control.
- **H2.** Before a word-final /s/ (*casas*), Southern word-final unstressed /e/ and /o/ are more
  open than without the /s/ (*casa*), relative to the same contrast in the North.
- **H2a.** Southern speakers weaken coda /s/ (lower spectral centre of gravity in codas, no
  difference in onsets). This is the check that H2 depends on.

## What I found

- **H1, partly supported.** 6 of 10 stressed-vowel measures are equivalent between Norte and Sur
  within *d* = ±0.30, and none differs after Holm. The other 4 are inconclusive, all with
  \|*d*\| ≤ 0.17. F1 of /e i o/ and F2 of /u/ are equivalent in every version of the analysis.
- **H2, not supported.** Sur's shift before word-final /s/ goes the predicted way (*d* = +0.19)
  but is not significant (*p* = 0.15, 34 % power). Only Sur /e/ really opens; for /a/ the
  difference is that Norte and México close it and Sur does not.
- **H2a, supported.** Coda /s/ is weaker in Sur (*d* = +0.46), onset /s/ is not
  (*d* = −0.06), and the interaction is +596 Hz. The three varieties line up Norte < México < Sur.
- The analysis was run three times on the same day because of two measurement problems, both
  documented in PLAN_v3.md §4.5 and RESULTS.md §6.
- **v3.1** (40 clips per speaker instead of 8, same speakers): H1 8/10 equivalent; H2
  *d* = +0.23, *p* = 0.0495, and *d* = +0.40, *p* = 0.003 on the new clips alone; H2a the same.
  It backs up the v3.0 estimates but is not an independent replication (RESULTS.md §9).
- A visual check of 29 formant tracks found about 1 mismeasured token in 20, mostly /i/ F2
  (RESULTS.md §5); the hand measurement in Praat is still to do.

## Setup

Two environments. The analysis runs in a virtualenv, and forced alignment needs its own conda
environment because MFA pulls in Kaldi.

```bash
./speech/bin/pip install -r requirements.txt
```

```bash
mamba create -y -n mfa -c conda-forge montreal-forced-aligner ffmpeg
```

```bash
conda run -n mfa mfa model download acoustic spanish_mfa
```

```bash
conda run -n mfa mfa model download dictionary spanish_mfa
```

It needs `validated.tsv` (the Common Voice 26.0 metadata) and `audios_corpus/` (the sampled
mp3s). The sample, the WAVs and the TextGrids from stages 1–3 are already in `output/`.

## Running it

The usual run reuses the sample and the alignment:

```bash
./speech/bin/python run_all.py --from 4
```

```bash
./speech/bin/python run_all.py --list
```

Stage 4 is the slow one (about 10 minutes); the rest take seconds. Every stage saves its output
in `output/`, so the later ones can be re-run on their own.

### v3.1: 40 clips per speaker

v3.1 reruns the whole pipeline with up to 40 clips per speaker instead of 8, in its own
folder so v3.0 is not touched (design in PLAN_v3.md §4.7). The extra mp3s come from the
corpus tarball with the list in `output_v31/data/extract_list.txt`.

```bash
SPEECH_OUTPUT=output_v31 ./speech/bin/python -m src.s1_sample --all-corpus --cap 40
```

```bash
SPEECH_OUTPUT=output_v31 ./speech/bin/python run_all.py --from 2
```

### Formant track check

Draws a sample of 29 vowels with their Burg tracks, for checking by eye and by hand in Praat
(`output/track_check/`; open `open_in_praat.praat` in Praat to step through them).

```bash
./speech/bin/python -m src.check_tracks
```

## Tests

```bash
./speech/bin/python -m unittest discover -s tests -v
```

## The pipeline

| Stage | Module | What it does |
|-------|--------|--------------|
| 1 | `s1_sample` | Clip manifest keyed on `client_id`, capped at 8 clips per speaker |
| 2 | `s2_prepare` | mp3 → 16 kHz mono WAV plus `.lab` transcripts, one folder per speaker |
| 3 | `s3_align` | Montreal Forced Aligner with `spanish_mfa` |
| 4 | `s4_extract` | One row per vowel: F1/F2 at the midpoint, duration, stress, what follows. Formant ceiling chosen per speaker from a 4000–6500 Hz grid |
| 5 | `s5_normalize` | Range filters, per-speaker per-vowel outliers, Lobanov over the five vowel means |
| 6 | `s6_vowels` | H1: stressed vowels in content words, equivalence test, México control, vowel space area |
| 7 | `s7_plural` | H2: *casa*-type vs *casas*-type word-final vowels, within speaker |
| 8 | `s8_sibilants` | H2a: /s/ centre of gravity by position, on the 48 kHz mp3 |
| 9 | `plots` | The figures, read from the stage tables |

Paths, seed and thresholds are all in `src/config.py`.
