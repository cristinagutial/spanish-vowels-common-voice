# v3.0 Plan — Experimental Phonetics
**Northern vs. Southern Peninsular Spanish vowels, Common Voice**
Cristina Gutiérrez Alcalá · Experimental Phonetics (Katrin Schneider) · module *Topics in Phonetics*
Drafted 2026-10-22.

---

## 0. Context

- Module: *Topics in Phonetics* = Experimental Phonetics + Advanced Speech Perception, 6 ECTS
  together. Only the smaller experiment is needed (the larger one is for the Project Seminar).
- Deliverables from the course slides: a short presentation of the planned experiment
  (21.01 or 28.01.2027), then experiment + results in an individual session with the lecturer
  (mid-Feb to end of March 2027). The slides do not mention a written report.
- v3.0 starts from **v2.0** (the August 2026 rebuild in `src/`). The midterm (v1: `extractor*.py`,
  `analisis*.py`, `limpieza*.py`, `etiquetar_vocales.py`, `preprocesador.py`) is left out entirely.
- v2.0 as submitted is preserved in `CODE_GutiérrezAlcalá.zip` (checked identical to `src/` on
  2026-10-22).

## 1. Scope decisions

| decision | choice | status |
|---|---|---|
| Data | Corpus study only, same Common Voice 26.0 sample (957 speakers) | decided 2026-10-22 |
| Lab recordings | None | decided 2026-10-22 |
| Coda /s/ | Kept only as a check that supports H2, not as its own result | decided 2026-10-22 |
| Output | `RESULTS.md`, tables and figures; Cristina makes the slides separately | decided 2026-10-22 |
| Midterm files | Moved to `archive_ssp/`, not deleted | done 2026-10-22 |

Stages 1–3 (sampling, conversion, alignment) are reused as they are. No new sample, no new
alignment.

## 2. Research question and hypotheses

**RQ.** Do Northern and Southern Peninsular Spanish differ in vowel quality, either across the
whole vowel space or in the context where the literature predicts a difference (before coda /s/)?

**H1 — global vowel space.** Stressed vowels do not differ between Norte and Sur by more than
the smallest effect of interest (§4.1). México is the positive control: the same procedure has
to detect Peninsular vs. Mexican differences.

**H2 — vowels before plural /s/.** In the South, word-final unstressed mid vowels are more open
(higher F1) when a plural /s/ follows (*casas*, *libros*, *veces*) than when no /s/ follows
(*casa*, *libro*), relative to the same contrast in the North. Strongest for /e/ and /o/; /a/ is
expected to front (F2) rather than open.

**H2a — manipulation check.** Coda /s/ is weaker in Sur than in Norte (lower centre of gravity in
codas, no difference in onsets). If this fails, H2 cannot be interpreted.

## 3. Materials and measures

| | v2.0 | v3.0 |
|---|---|---|
| Formant ceiling | Fixed by declared sex (5000 / 5500 / 5250 Hz) | Optimised per speaker (Escudero et al. 2009), §3.1 |
| Vowels analysed | All, stressed and unstressed pooled | H1: stressed vowels in non-function words. H2: the matched word-final set |
| Normalisation | Lobanov over pooled tokens, ≥ 4 vowel types | Lobanov over the 5 vowel-category means, all 5 vowels required |
| QC | Range filters incl. B1/B2 (drops 8 %), outliers | F1/F2 ranges + within-speaker outliers; no bandwidth filter |
| Measures | F1, F2, B1, B2, duration, F0, HNR, SNR | F1, F2, duration. Bandwidths, HNR and SNR dropped |
| /s/ CoG | Whole segment, full spectrum, 16 kHz audio | Middle 50 % of the segment, 1 kHz high-pass, native sample rate |
| /s/ position | Word-final /s/ always coda | Coda = before a consonant or pause; word-final before a vowel is its own category, excluded from the test |

### 3.1 Per-speaker ceiling

For each speaker, run Burg with ceilings from 4000 to 6500 Hz in 100 Hz steps and keep the one
that minimises the summed within-vowel variance of log F1 and log F2 over that speaker's stressed
tokens (based on the variance idea in Escudero et al. 2009). This drops the dependency on the
self-reported sex field (105 of the 956 speakers have none). Only tokens measurable at every ceiling count,
and a vowel needs ≥ 3 tokens, with ≥ 3 vowels per speaker. If the stressed tokens are not enough,
all tokens are used; if those are not enough either, the old sex-based ceiling is the fallback.

The grid was first 4500–6500 Hz. A pilot on 6 speakers (ceilings only, no results) put one male
speaker on the 4500 Hz floor, and that range was meant for female voices, so the grid starts at
4000 Hz.

### 3.2 Token sets

- **H1 set:** stressed vowels (from the existing stress rules), excluding the function-word list
  in `config.py`. Feasibility on v2.0 data: 27,165 stressed tokens; speakers with all 5 stressed
  vowels: Norte 323, Sur 137, México 204.
- **H2 set:** word-final unstressed /a e o/ in polysyllabic words not on the function-word list.
  *Plural* = the vowel is followed by a word-final /s/; *singular* = the vowel ends the word.
  Both followed by a consonant-initial word or a pause (not a vowel), so resyllabified /s/ and
  vowel–vowel coarticulation stay out. A speaker × vowel cell needs ≥ 2 tokens of each type.
  Feasibility (before the following-context restriction): 12,548 singular and 4,794 plural
  tokens; speakers with at least one usable vowel: Norte 305, Sur 135, México 194.

## 4. Analysis (locked before the rerun)

Unit of analysis: the speaker throughout. Speaker × vowel means first, then tests on speakers.
Welch tests, Cohen's *d*, 95 % CIs from 2,000 bootstrap resamples of speakers, Holm within each
family, seed 42.

### 4.1 H1

- Per vowel and formant (5 × 2 = 10 measures), Norte vs. Sur on Lobanov speaker means.
- **Equivalence test (TOST)** with bounds *d* = ±0.30. Reason: 0.30 is the size of the smaller
  Peninsular–Mexican vowel differences that v2.0 detected, so it is a "real dialect difference" in
  this corpus. A measure is called equivalent when the 90 % CI of *d* falls inside ±0.30.
- Same 10 measures for Norte vs. México and Sur vs. México (positive control, ordinary tests).
- Vowel space area (convex hull, Lobanov) as a descriptive summary.

### 4.2 H2

- Per speaker and vowel: shift = mean F1 (plural) − mean F1 (singular), Lobanov.
- **Primary test (one test):** the per-speaker shift averaged over /e/ and /o/, Sur vs. Norte,
  Welch, two-sided.
- Secondary (Holm): each vowel separately for F1; F2 of /a/; México vs. each Peninsular variety.
- Duration shift reported descriptively (no speech-rate normalisation, so no test).

### 4.3 H2a

- Speaker-level CoG means by position (coda vs. onset), Norte vs. Sur, Welch + Holm.
- Dialect × position mixed model on speaker × position cells, random intercept per speaker.

### 4.4 Implementation details (fixed 2026-10-22, before the rerun)

- Function words: the list in `config.FUNCTION_WORDS` (articles, prepositions, conjunctions,
  pronouns, possessives, demonstratives, quantifiers, numerals *dos/tres/seis*, grammatical
  adverbs, forms of *ser/estar/haber*).
- H1 cells: a speaker × vowel cell needs ≥ 2 stressed tokens. Lobanov needs all 5 vowels with
  ≥ 2 tokens each, over all tokens.
- Equivalence decision: the speaker-bootstrap 90 % CI of *d* inside ±0.30. The Welch TOST
  *p*-value is reported next to it.
- Pause: any gap between two phones in the TextGrid (MFA writes silences as empty intervals).
- H2 "plural" means a vowel followed by a word-final /s/ in the same word. Most of these are
  plurals, but words like *lunes* also count.
- /s/: CoG on the original 48 kHz mp3, middle 50 % of the segment with a Hann window, spectrum
  filtered to 1–11 kHz (fixed upper edge so every clip uses the same band).

### 4.5 Deviation after the first run (2026-10-22)

The first v3.0 run (kept in `output/v3_first_run/`) produced impossible Lobanov values, up to
*z* = 14.5. The cause was a measurement failure, not a dialect effect. A few speakers' tracks put
every vowel at almost the same F2 (one Norte speaker: all five vowels at 990–1040 Hz), and the
ceiling search favours such tracks because they have little within-vowel spread. Lobanov over
vowel means then divides by a near-zero spread.

I tried a ratio criterion (within-vowel ÷ between-vowel spread). It failed the synthetic-vowel
check badly (it chose 6400 Hz and gave /i/ F1 errors of +629 %), so I dropped it. The ceiling
search stays as planned. The fix is a dialect-blind speaker check in stage 5, with thresholds
fixed before the rerun:

- F2 /i/ − F2 /u/ ≥ 500 Hz, and F1 /a/ − max(F1 /i/, F1 /u/) ≥ 100 Hz, on the speaker's vowel
  means in Hz. Speakers who fail are left out of every analysis.

**Sensitivity analysis**, also added before the rerun: H1 and the H2 primary test are repeated
with the v2.0 sex-based ceiling (5000 / 5500 / 5250 Hz) and the same QC. It is reported but not
used for the decisions.

**Second deviation, same day.** The 500 / 100 Hz thresholds were a mistake (run kept in
`output/v3_qc500/`). F2 /i/ − /u/ turned out to be a continuous distribution (5th percentile
426 Hz), so the cut removed real speakers with compressed vowel spaces, not just broken tracks.
It also selected speakers on the same F2 values H1 tests, and unevenly: 7.7 % of Norte, 11.9 % of
México, 3.7 % of Sur. The rule is now **thresholds of 0 Hz**, so it only removes physically
impossible vowel spaces (/i/ not in front of /u/, or /a/ not below /i/ and /u/). I fixed this
before seeing the 0 Hz results, because it is the only version that does not cut into the real
distribution.

All three runs (first run, 500 Hz, final) are reported in `RESULTS.md`.

### 4.6 Guardrail

These choices are fixed now. Whatever the rerun gives is what gets reported; nothing is adjusted
afterwards to make an effect appear or disappear.

**Disclosure.** On 2026-10-22, while counting tokens for §3.2, I looked at the descriptive H2
shifts on v2.0 data (no tests). H2 was written before that look. Raw shifts (plural − singular,
F1 Lobanov z): Norte a −0.24, e −0.03, o −0.03; Sur a −0.01, e +0.13, o +0.05; México
a −0.15, e −0.12, o −0.12. These use the v2.0 measurements and token set, so the v3.0 numbers will
differ.

### 4.7 v3.1 — more clips per speaker (declared 2026-10-23, before running)

**Why.** H2 had 34 % power (96 Sur vs 220 Norte speakers). Common Voice 26.0 has no more
Southern speakers (188 is all of them, and no free-text accents add any), but 118 of the 188
recorded more than the 8 clips v3.0 used. More clips per speaker means more *casa*/*casas*
tokens per speaker, so more speakers qualify for H2 and their means are more stable.

**Design.** Same speakers, same seeded clip ranking, cap raised from 8 to **40 clips per
speaker** for all three varieties (so measurement precision stays comparable). Ranks 1–8 are
the v3.0 clips. That gives 14,874 clips (Norte 6,983, Sur 3,366, México 4,525), +9,297. 40 was
picked because the Sur gain flattens after it (cap 60 adds only 603 more Sur clips).
Everything else is v3.0 as it stands: same stages, thresholds, tests and figures. Stages 1–3
are rerun, and alignment is redone for all clips. Output goes to `output_v31/`; `output/`
(v3.0) is not touched.

**Tests.** The same H1, H2 and H2a tests as v3.0, plus one H2 secondary analysis on the new
clips only (ranks 9–40), which v3.0 never measured.

**How to read it.** v3.1 is not an independent replication: the speakers are the same and the
v3.0 clips are included. It answers "does the estimate hold with ~3× more data per speaker?".
v3.0 stays the planned result; v3.1 is reported next to it, whatever it shows.

## 5. Pipeline v3.0

| stage | module | change |
|---|---|---|
| 1 | `s1_sample` | unchanged; manifest reused |
| 2 | `s2_prepare` | unchanged; WAVs reused |
| 3 | `s3_align` | unchanged; TextGrids reused |
| 4 | `s4_extract` | per-speaker ceiling; phone after /s/; drop B1/B2/HNR/SNR |
| 5 | `s5_normalize` | balanced Lobanov, 5 vowels required, vowel-triangle check, no bandwidth filter; also writes the sex-ceiling token file |
| 6 | `s6_vowels` | H1: stressed set, TOST, México control, vowel space area, sex-ceiling sensitivity |
| 7 | `s7_plural` | H2: singular/plural contrast (new), sex-ceiling sensitivity |
| 8 | `s8_sibilants` | H2a: /s/ check (from `s10_codas`, new measurement) |
| 9 | `plots` | figures read from the stage tables, nothing recomputed |

Removed: `s7_cluster`, `s8_classify`, `s9_pseudorep`; from `s6_stats` the bandwidth-vs-noise
check, the naive reproduction and the token-level mixed-model section; from `config.py` the
legacy paths.

Tests: keep the extractor, stress, coda, stats and label tests. Add tests for the ceiling search
(on synthetic vowels), the plural/singular token selection, and the following-context rule.

Figures for the slides: (1) `fig_vowel_space`, stressed vowel space, three varieties;
(2) `fig_equivalence`, *d* with 90 % CIs against ±0.30, and the México contrasts;
(3) `fig_plural`, plural − singular shift by vowel × variety; (4) `fig_sibilants`, /s/ CoG by
position × variety and the weakening index; (5) `fig_ceilings`, the chosen ceilings by sex;
(6) `fig_data_flow`, corpus → tokens. The method diagram is left for the slides.

## 6. Cleanup (to `archive_ssp/`, nothing deleted)

Midterm scripts and plots, `files/`, `files.zip`, `audios_filtrados/`, `results_formants*.csv`,
`metadata_filtrada.csv`, `validated_v25.tsv` (old corpus, 150 MB), `figuras_overleaf*`,
`Speech_Project_Cristina_Gutierrez/`, the SSP report/slides/proposal PDFs and `.tex` files.
Kept in place: `validated.tsv`, `audios_corpus/`, `output/work/` (WAVs + TextGrids), `speech/`.

## 7. Timeline (course dates)

| when | what |
|---|---|
| 22.10.2026 | **done:** stages 4–9 rebuilt and run, 38 tests passing, `RESULTS.md` written |
| by 14.01.2027 | slides (made separately) |
| 14.01.2027 | individual discussion slot with the lecturer |
| 21/28.01.2027 | presentation of the experiment |
| Feb – Mar 2027 | results session |

## 8. Limitations to state up front

- Read speech by volunteers, uncontrolled microphones, MP3 compression (affects high-frequency
  /s/ energy).
- Self-reported, coarse labels: "Sur" mixes Eastern and Western Andalusian, Extremaduran and
  Murcian; the vowel opening is described mainly for Eastern Andalusian, so H2 is diluted.
- Forced alignment places an /s/ boundary even when the /s/ is elided.
- Sentences differ between speakers; word identity is not controlled beyond the token-set rules.
