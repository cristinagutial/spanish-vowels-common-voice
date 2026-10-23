# Project Improvement Plan
**North vs. South Peninsular Spanish — Vowel Space Analysis**
Cristina Gutiérrez Alcalá · Speech Signal Processing · Prof. Wokurek
Plan drafted 2026-08-02, for the 12-page written report.

---

## 0. Executive summary

The midterm analysis reports three large effects (F1 *d* = 1.34, F2 *d* = −0.33, B2
*d* = −1.39). A speaker-level audit of the same 440 tokens shows all three are
produced by **two individual speakers**. The study is currently a comparison of
two people, not two dialects.

This plan rebuilds the study so that the unit of analysis is the *speaker*, not
the *clip*, adds real phoneme labels via forced alignment, and reframes the
report around the discovery — which is a stronger contribution than the original
descriptive claim.

---

## 1. Diagnosis

### 1.1 Pseudoreplication (fatal)

`results_formants_clean.csv` merged against `client_id` in `validated.tsv`:

| dialect | tokens | speakers | tokens from top speaker |
|---------|--------|----------|-------------------------|
| Norte   | 244    | 54       | 132 (**54 %**)          |
| Sur     | 196    | 26       | 157 (**80 %**)          |

Welch t-tests re-run after removing only those two speakers (n = 151 tokens):

| feature | original *d* | original *p* | corrected *d* | corrected *p* |
|---------|-------------|--------------|---------------|---------------|
| F1      | +1.34       | < 0.001      | **+0.01**     | 0.97          |
| F2      | −0.33       | < 0.001      | **+0.04**     | 0.82          |
| B1      | ~0          | n.s.         | +0.12         | 0.52          |
| B2      | −1.39       | < 0.001      | **−0.09**     | 0.64          |

Speaker-level aggregation (one mean per speaker, ≥3 tokens, n = 12 Norte / 4 Sur)
gives F1 479 vs 448 Hz, *p* = 0.50. **No effect survives.**

The t-test assumes independent observations. 440 clips from 80 speakers are not
independent; the effective sample size is closer to 80, and with one speaker
supplying 80 % of a cell, closer to 2. This inflates *df* enormously and
guarantees significance regardless of whether a dialect effect exists.

### 1.2 Confounds with the grouping variable

| | Norte | Sur |
|---|---|---|
| male | 221 (92 %) | 190 (97 %) |
| female | 20 | 6 |
| modal age band | fifties | sixties |

Sex determines vocal-tract length and therefore formant frequencies directly.
Age band is almost perfectly confounded with dialect because each cell is
dominated by one speaker. Any raw-Hz comparison between these groups is
uninterpretable.

### 1.3 One token per clip

`extractor_v2.py` takes the global intensity maximum of each ~5 s clip and
returns a single F1/F2/B1/B2 quadruple. Consequences:

- The vowel identity at that peak is unknown (this is why `etiquetar_vocales.py`
  failed — 14 of 903 clips resolved to an unambiguous vowel).
- Clips average 5.0 s ≈ 15–20 vowels; >90 % of the available signal is discarded.
- The resulting "vowel space" is a distribution over an arbitrary mixture of
  phonemes, so between-group differences may simply reflect *which* vowels
  happened to be loudest — a confound that cannot be ruled out post hoc.
- The k = 3 result cannot be interpreted as "three perceptual zones of Spanish";
  it is k-means on an unlabelled mixture.

### 1.4 Methodological issues

| # | Issue | Fix |
|---|-------|-----|
| 1 | No formant normalisation | Lobanov z-score by speaker + Bark; report raw and normalised |
| 2 | Fixed 5000 Hz ceiling for all speakers | Sex-adaptive (5000 M / 5500 F) or iterative optimal ceiling |
| 3 | Silhouette compared across 2-D and 4-D | Invalid — silhouette falls with dimensionality by construction. Use ARI against known vowel labels instead |
| 4 | B2 effect may be recording quality | LPC bandwidth tracks SNR; one speaker = one microphone. Add per-clip SNR as covariate |
| 5 | 4 uncorrected t-tests | Holm–Bonferroni; report bootstrap CIs on effect sizes |
| 6 | `.sample(n=300)` unseeded | Seeded, versioned sampling manifest |
| 7 | 903 mp3s vs 600 metadata rows | Orphans from an earlier run; rebuild the sample cleanly |
| 8 | Dead `prueba.wav` test code | Real unit tests on synthetic vowels |
| 9 | Serial extraction, no caching | `multiprocessing`, cached intermediates |
| 10 | No `requirements.txt`, no entry point | Pinned env + `run_all.py` |

---

## 2. What gets built

### Tier 1 — Validity (mandatory)

**T1.1 Speaker-aware sampling.**
Rebuild the sample from `validated.tsv` keyed on `client_id`, not `path`.
Record `client_id`, `gender`, `age`, `accents` for every clip. Seeded.
Cap clips per speaker so no individual can dominate a cell. Target design:
maximise speaker count first, clips per speaker second.

**T1.2 Forced alignment (MFA).**
`validated.tsv` supplies the transcript for every clip. Montreal Forced Aligner
with `spanish_mfa` acoustic model + dictionary produces phone-level TextGrids.
This yields, per clip:

- vowel identity (/a e i o u/)
- vowel onset/offset → **duration** as a new feature
- stress position
- **following segment** → enables the coda-/s/ analysis (§ Tier 3)

Expected yield: ~15 000 labelled vowel tokens from 903 clips, versus 440
unlabelled ones now. *Fallback if MFA install fails:* De Jong & Wempe style
syllable-nucleus detection in pure Python — many tokens per clip, but no
phoneme identity.

**T1.3 Better extraction.**
Measure at the vowel midpoint plus the 25 %/50 %/75 % points (captures
diphthongisation and formant dynamics). Sex-adaptive ceiling. Reject tokens
shorter than 30 ms. Parallelised.

**T1.4 Normalisation.**
Lobanov (speaker z-score, the sociophonetic standard), Bark, and raw Hz.
Lobanov requires multiple vowel categories per speaker — now possible thanks to
T1.2, and impossible before it.

**T1.5 SNR estimation.**
Per-clip SNR to test whether the B2 effect is acoustic or an artifact of
recording conditions.

### Tier 2 — Statistics done properly

**T2.1 Linear mixed-effects models.** `statsmodels`:
```
F1_lobanov ~ dialect * vowel + sex + age + (1 | speaker)
```
Speaker as random intercept is the actual fix for pseudoreplication.

**T2.2** Holm-corrected p-values; bootstrap CIs on all effect sizes
(resampled **by speaker**, not by token).

**T2.3** Sociophonetic metrics: per-vowel centroid distances, vowel-space area
(convex hull), Pillai scores for overlap.

**T2.4** Power analysis — given 34 Sur speakers, what effect size is detectable?
This bounds what the study can honestly claim.

### Tier 3 — New comparisons

**T3.1 Mexican Spanish outgroup** (needs the download; 1 525 speakers available).
Answers "is the Norte/Sur difference large?" against a variety known to differ.
Turns a two-way test into a graded comparison.

**T3.2 Andalusian coda-/s/ hypothesis.** *The* documented Sur feature: coda /s/
weakens/aspirates, and the preceding vowel is claimed to open and lengthen
compensatorily. With alignment we can test:
```
F1 ~ dialect * coda_s_context + (1 | speaker)
```
A significant interaction is a real, falsifiable, literature-grounded finding —
qualitatively better than describing a global vowel-space shift.

**T3.3 Speaker-independent classification.** LDA / SVM / random forest with
`GroupKFold` grouped by speaker (ungrouped CV would leak speaker identity and
inflate accuracy — worth demonstrating both, it makes the pseudoreplication
point a second time). Report accuracy vs. chance, confusion matrix, feature
importance.

**T3.4 Sex as a factor**, not a nuisance: does the dialect effect differ by sex?

### Tier 4 — Code quality

```
Speech_Project/
├── run_all.py                 # one-command pipeline
├── requirements.txt
├── src/
│   ├── config.py              # paths, seed, constants
│   ├── s1_sample.py           # speaker-aware sampling
│   ├── s2_prepare.py          # mp3 → 16 kHz wav + .lab transcripts
│   ├── s3_align.py            # MFA wrapper → TextGrids
│   ├── s4_extract.py          # per-vowel formants, parallel
│   ├── s5_normalize.py        # Lobanov / Bark / QC / SNR
│   ├── s6_stats.py            # mixed models, effect sizes, CIs
│   ├── s7_cluster.py          # k-means + ARI against true labels
│   ├── s8_classify.py         # grouped CV
│   ├── s9_pseudorep.py        # the naive-vs-corrected demonstration
│   └── plots.py
├── tests/                     # synthetic-vowel tests for the extractor
└── output/{figures,tables,data}/
```
Seeds fixed, intermediates cached, logging, docstrings in English throughout.

---

## 3. Report structure (12 pages)

| § | Content | pp. |
|---|---------|-----|
| 1 | Introduction — dialect variation, F1/F2, the research question | 1 |
| 2 | Related work — Andalusian vowel system, sociophonetic normalisation, Common Voice as a corpus | 1.5 |
| 3 | Corpus & sampling — incl. the speaker-distribution problem in Common Voice | 1.5 |
| 4 | Method — alignment, extraction, normalisation, statistical model | 2 |
| 5 | **A cautionary result** — naive analysis, its large effects, the speaker audit, the collapse | 1.5 |
| 6 | Corrected results — mixed models, per-vowel, coda-/s/, outgroup, classification | 3 |
| 7 | Discussion — what survives, what doesn't, what Common Voice can and cannot support | 1 |
| 8 | Limitations & conclusions | 0.5 |

Section 5 is the spine. It's a genuine methodological contribution and it
explains why the corrected analysis is designed the way it is.

---

## 4. Execution order

| Step | Depends on | Status |
|------|-----------|--------|
| MFA install (`mamba create -n mfa`) | — | **DONE** — MFA 3.4.1 + ffmpeg in conda env `mfa` |
| `spanish_mfa` acoustic model + dictionary | MFA install | **DONE** — both verified via `mfa model list` |
| T4 scaffold + config | — | **DONE** — `src/`, `config.py`, `requirements.txt`, `run_all.py` |
| T1.1 sampling | scaffold | **DONE** — `s1_sample.py`, 903 clips / 127 speakers |
| T1.2 alignment | MFA install | **DONE** — `s3_align.py`, 898/903 aligned |
| T1.3–1.5 extraction | alignment | **DONE** — `s4_extract.py`, 16,951 vowel tokens |
| Normalisation + QC | extraction | **DONE** — `s5_normalize.py`, Lobanov for 110 speakers |
| T2 statistics | extraction | **DONE** — `s6_stats.py` |
| T3.2 coda /s/ | extraction | **DONE** — `s10_codas.py`, **positive result** |
| T3.3 classification | extraction | **DONE** — `s8_classify.py` |
| Clustering with ARI | extraction | **DONE** — `s7_cluster.py` |
| S9 pseudoreplication demo | legacy CSVs | **DONE** — `s9_pseudorep.py` |
| Tests | extraction | **DONE** — 24 tests passing |
| T3.1 México outgroup | corpus download | **DONE** — 300 México speakers, corpus 26.0 |

Full pipeline runs end to end in ~8 minutes: `./speech/bin/python run_all.py`.
All numbers are collected in `RESULTS.md`.

### Scale-up to corpus 26.0 (2026-08-02)

| | midterm | first rebuild | final |
|---|---|---|---|
| Sur speakers | 26 | 34 | **188** |
| Norte speakers | 54 | 93 | **469** |
| México speakers | — | — | **300** |
| Vowel tokens | 440 | 15,249 | **89,958** |
| Gini (clips/speaker, Sur) | 0.85 | 0.85 | **0.18** |
| Detectable effect (80 % power) | — | *d* ≥ 0.63 | ***d* ≥ 0.24** |

Three issues surfaced during the scale-up, each fixed and worth recording:

1. **Stale alignment output.** MFA's `--clean` resets its own database but leaves
   previous TextGrids in place, and speaker directory names are assigned per run,
   so a leftover grid could sit under a name that now belongs to someone else.
   Stage 3 now deletes the output tree before aligning.
2. **Quadratic bottlenecks.** `silhouette_score` builds a full pairwise distance
   matrix and RBF SVM training is roughly quadratic; both were intractable at
   90,000 tokens. Silhouette now samples 10,000 points, and cross-validation runs
   on a 15,000-token subsample that retains every speaker.
3. **A reproducibility bug.** `s1_sample` defaulted to Norte and Sur, so a plain
   `run_all.py` silently rebuilt the sample without the México control and
   overwrote the outgroup results. The default is now the full study design.

Conclusions that changed with the larger sample:

- The Norte/Sur null is now **well-powered** rather than merely undetected.
- The coda-/s/ finding **strengthened** (*d* = +0.63 against Norte and México
  pooled) and gained a three-way ordering matching the literature.
- The compensatory vowel-opening effect **shrank sharply**, from *d* = −0.83 at 34
  Sur speakers to −0.17 at 168. The interaction stays significant, but the report
  should present it as small — and the shrinkage itself as a worked example of
  regression to the mean.

Strategy: build the entire pipeline against the 903 clips already on disk. If
the download lands in time, rerun with more speakers + México. If it doesn't,
the report still stands on the corrected small-sample analysis.

---

## 5. Honest expected outcome

With 34 Sur speakers, there is a real chance that **no significant dialect
effect survives**. That is a publishable result in this context: it means
Common Voice's speaker distribution does not support the claim at this sample
size, and it explains why the naive analysis appeared to find one. The report
is designed so that either outcome — effect or no effect — yields a complete,
defensible 12 pages.
