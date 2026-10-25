# Northern vs. Southern Peninsular Spanish vowels

Cristina Gutiérrez Alcalá · Experimental Phonetics (Katrin Schneider) · *Topics in Phonetics*

An acoustic study of stressed vowels and of vowels before word-final /s/ in Northern and
Southern Peninsular Spanish, with Mexican Spanish as a reference, on the dialect-labelled
recordings of Mozilla Common Voice. It builds on the pipeline from my Speech Signal Processing
project. The plan is in [PLAN_v3.md](PLAN_v3.md) and the results in [RESULTS.md](RESULTS.md).

## What I found

- **H1, partly supported.** 6 of 10 stressed-vowel measures are equivalent between Norte and Sur
  within *d* = ±0.30, and none differs. The other 4 are small but inconclusive.
- **H2, not supported.** The vowel before word-final /s/ opens more in Sur, as predicted
  (*d* = +0.19), but not significantly (*p* = 0.15). The effect comes from /e/.
- **H2a, supported.** Sur weakens coda /s/ (*d* = +0.46) and not onset /s/.

## Setup

The analysis runs in a virtualenv; forced alignment needs its own conda environment.

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

It needs `validated.tsv` (Common Voice 26.0 metadata) and `audios_corpus/` (the 5,577 sampled
mp3s). Neither is in the repository.

## Running it

The usual run reuses the sample and the alignment:

```bash
./speech/bin/python run_all.py --from 4
```

Stage 4 takes about 10 minutes; the rest take seconds.

```bash
./speech/bin/python -m unittest discover -s tests -v
```

## The pipeline

| Stage | Module | What it does |
|-------|--------|--------------|
| 1 | `s1_sample` | Clip list keyed on the speaker, up to 8 clips per speaker |
| 2 | `s2_prepare` | mp3 → 16 kHz WAV plus transcripts for the aligner |
| 3 | `s3_align` | Montreal Forced Aligner with `spanish_mfa` |
| 4 | `s4_extract` | F1/F2 at each vowel midpoint, with the formant ceiling chosen per speaker |
| 5 | `s5_normalize` | Quality control and Lobanov normalisation |
| 6 | `s6_vowels` | H1: stressed vowels, equivalence test, México reference |
| 7 | `s7_plural` | H2: *casa* vs *casas* |
| 8 | `s8_sibilants` | H2a: /s/ centre of gravity by position |
| 9 | `plots` | The figures |

Paths and thresholds are in `src/config.py`. Older material is in `archive_ssp/` (Speech Signal
Processing project) and `archive_v3/` (earlier runs and extra analyses).
