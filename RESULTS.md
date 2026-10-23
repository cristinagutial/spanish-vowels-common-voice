# Results — v3.0

Every number with the file it came from (all in `output/tables/` unless stated).
Generated on 2026-10-22. Stage 4 comes from the run logged in `output/v3_qc500/run_v3.log`
(stage 4 did not change after that); stages 5–9 are in `output/run_v3.log`. Numbers checked
against the tables on 2026-10-23. Design and locked choices: [PLAN_v3.md](PLAN_v3.md). The v2.0 results
are in `archive_ssp/v2_docs/RESULTS.md`.

---

## 0. Summary

| | result | verdict |
|---|---|---|
| **H1** stressed vowels, Norte vs Sur | 6 of 10 measures equivalent within *d* = ±0.30; none differs after Holm; the other 4 are inconclusive (\|*d*\| ≤ 0.17) | partly supported |
| México control | the only Holm-significant vowel differences are Peninsular–Mexican, but on average they are only 1.2–1.5× the Norte–Sur ones | weak |
| **H2** vowel before word-final /s/ | Sur shift +0.09 vs Norte −0.04, *d* = +0.19, 95 % CI [−0.06, +0.44], *p* = 0.146 | not supported at α = 0.05; direction as predicted |
| **H2a** coda /s/ check | coda CoG *d* = +0.46, *p*_Holm < 0.001; onset *d* = −0.06, n.s.; interaction +596 Hz, *p* < 0.001 | supported |

H1 and H2 were run three times on 2026-10-22, because the first two runs had measurement problems
(§6). The conclusions above hold in all three, and in the sensitivity analysis with the v2.0
sex-based ceiling, except where §6 says otherwise.

---

## 1. Data

| | value | source |
|---|---|---|
| Corpus | Common Voice `es` 26.0 (2026-06-12) | — |
| Clips sampled / aligned | 5,577 / 5,572 | `s3_alignment_report.csv` |
| Speakers with vowel tokens | 956 (Norte 469, Sur 188, México 299) | `s5_speaker_coverage.csv` |
| Vowel tokens extracted | 100,053 | `s5_qc_funnel.csv` |
| Tokens after QC | 97,647 (97.6 %) | `s5_qc_funnel.csv` |
| Speakers with all 5 vowels (≥ 2 tokens each) | 805 | `s5_speaker_coverage.csv` |
| … and a possible vowel triangle | 798 (Norte 388, Sur 161, México 249) | `s5_speaker_coverage.csv` |

QC funnel: 100,053 → measurable 100,052 → F1/F2 in range 99,164 → F2 > F1 99,164 → within-speaker
outliers removed 97,647. v2.0 kept 89,958 tokens; the difference is the bandwidth filter, which
v3.0 no longer applies.

The vowel-triangle check (§6) removed 7 speakers: 3 Norte, 4 México, 0 Sur.

### Formant ceiling per speaker

Source: `s4_ceilings.csv`, `fig_ceilings.png`.

| | value |
|---|---|
| Chosen from stressed tokens / all tokens / sex fallback | 757 / 141 / 58 speakers |
| Median, declared male / female / unknown | 4900 / 5800 / 5250 Hz |
| Median, Norte / Sur / México | 5100 / 5000 / 5200 Hz |
| At the grid edges (4000 / 6500 Hz) | 40 / 26 of 898 searched speakers |

The search does not use the sex field, and it still separates male and female voices by 900 Hz,
which is about what the usual fixed values assume (5000 vs 5500 Hz).

---

## 2. H1 — stressed vowel space

Source: `s6_h1_norte_sur.csv`, `fig_equivalence.png`, `fig_vowel_space.png`.
23,457 stressed tokens in content words → 3,431 speaker × vowel cells (≥ 2 tokens each).
Speakers: Norte 386, Sur 160, México 249. Lobanov z, *d* = Norte − Sur.

| measure | n Norte | n Sur | *d* | 90 % CI | *p*_Holm | *p*_TOST | equivalent |
|---|---|---|---|---|---|---|---|
| F1 /a/ | 375 | 158 | −0.16 | [−0.33, +0.01] | 0.854 | 0.083 | no |
| F1 /e/ | 369 | 154 | −0.05 | [−0.20, +0.11] | 1.000 | 0.003 | **yes** |
| F1 /i/ | 339 | 148 | +0.12 | [−0.04, +0.28] | 1.000 | 0.037 | **yes** |
| F1 /o/ | 339 | 150 | −0.08 | [−0.25, +0.07] | 1.000 | 0.013 | **yes** |
| F1 /u/ | 218 | 103 | −0.10 | [−0.32, +0.11] | 1.000 | 0.065 | no |
| F2 /a/ | 375 | 158 | +0.17 | [+0.03, +0.31] | 0.571 | 0.067 | no |
| F2 /e/ | 369 | 154 | −0.14 | [−0.27, −0.00] | 0.854 | 0.026 | **yes** |
| F2 /i/ | 339 | 148 | −0.10 | [−0.29, +0.02] | 1.000 | 0.003 | **yes** |
| F2 /o/ | 339 | 150 | −0.17 | [−0.32, −0.02] | 0.627 | 0.076 | no |
| F2 /u/ | 218 | 103 | +0.02 | [−0.20, +0.21] | 1.000 | 0.010 | **yes** |

**6 of 10 equivalent, 0 of 10 different.** The bootstrap-CI decision and the TOST *p* agree on
every measure.

The four inconclusive measures all have \|*d*\| ≤ 0.17, and their 90 % CIs reach past the ±0.30
bound. Two of them, F2 /a/ (*p* = 0.057) and F2 /o/ (*p* = 0.070), have 90 % CIs that exclude
zero: small uncorrected differences (Sur /a/ slightly further back, Sur /o/ slightly further
front) that do not survive Holm (*p*_Holm 0.57 and 0.63). F1 /a/ is the most consistent across
runs: *d* between −0.15 and −0.19 every time (Sur /a/ slightly more open), never significant.

**Power** (`s6_power.csv`): at the median n (339 Norte, 150 Sur), 80 % power for *d* ≥ 0.28 and
90 % for *d* ≥ 0.32.

**Vowel space area** (`s6_vowel_space_area_tests.csv`, descriptive): Norte 3.67, Sur 3.76,
México 3.87. Norte vs Sur *d* = −0.11, *p* = 0.43; Norte vs México *d* = −0.25, *p* = 0.045.

### México as positive control

Source: `s6_h1_control.csv`.

| contrast | mean \|*d*\| | max \|*d*\| | significant (Holm) |
|---|---|---|---|
| Norte vs Sur | 0.110 | 0.168 | 0 / 10 |
| Norte vs México | 0.163 | 0.256 | 1 / 10 (F1 /e/ *d* = −0.24) |
| Sur vs México | 0.136 | 0.378 | 2 / 10 (F2 /a/ *d* = −0.38, F2 /o/ *d* = +0.30) |

The only vowel differences that survive correction are Peninsular–Mexican. But the control is
weaker than in v2.0 (2.5× there, 1.2–1.5× here), possibly because v3.0 uses only stressed vowels
and so has fewer tokens per speaker (not tested). Present it as: "the method finds the transatlantic
differences, and they are small too", not as strong calibration.

---

## 3. H2 — vowel before word-final /s/

Source: `s7_h2_*.csv`, `fig_plural.png`. Word-final unstressed /a e o/ in content words,
followed by a consonant or pause. Shift = plural (*casas*) − singular (*casa*), per speaker and
vowel, Lobanov z; positive = more open.

Tokens (`s7_h2_token_counts.csv`): plural Norte 1,743 / Sur 715 / México 1,163; singular 4,362 /
2,041 / 2,680.

### Primary test

| | Sur | Norte | *d* | 95 % CI | *p* |
|---|---|---|---|---|---|
| F1 shift, mean of /e o/ | +0.088 (n = 96) | −0.040 (n = 220) | **+0.19** | [−0.06, +0.44] | **0.146** |

Not significant. The direction is the predicted one, but with 96 Sur and 220 Norte speakers the
test has 34 % power for an effect of this size; 80 % power would need *d* ≥ 0.34.

### Secondary tests (Holm, 6 tests)

| test | n | *d* | *p* | *p*_Holm |
|---|---|---|---|---|
| F1 shift /a/, Sur vs Norte | 48 / 115 | +0.42 | 0.019 | 0.093 |
| F1 shift /e/, Sur vs Norte | 57 / 107 | +0.31 | 0.081 | 0.322 |
| F1 shift /o/, Sur vs Norte | 66 / 176 | +0.06 | 0.668 | 0.668 |
| F2 shift /a/, Sur vs Norte | 48 / 115 | −0.27 | 0.104 | 0.322 |
| F1 shift /e o/, México vs Norte | 137 / 220 | −0.17 | 0.119 | 0.322 |
| F1 shift /e o/, México vs Sur | 137 / 96 | −0.34 | 0.012 | 0.072 |

### Descriptive shifts (F1, Lobanov z)

| | /a/ | /e/ | /o/ |
|---|---|---|---|
| Norte | −0.36 | −0.03 | −0.05 |
| Sur | +0.01 | **+0.16** | −0.00 |
| México | −0.16 | −0.22 | −0.14 |

Read carefully. Only Sur /e/ actually opens before /s/. The Sur–Norte difference for /a/ comes
from Norte and México *closing* /a/ before /s/, while Sur does not. So the pattern is "Sur does
not close where the others do" plus a small opening of /e/. Duration: plural vowels are 18–34 ms
shorter than singular ones in every variety (descriptive, no speech-rate control).

---

## 4. H2a — coda /s/ check

Source: `s8_*.csv`, `fig_sibilants.png`. 21,494 /s/ tokens (coda 10,479, onset 9,370, word-final
before a vowel 1,645, kept out of the test). 1,690 speaker × position cells (≥ 3 tokens) from 880
speakers.

| position | measure | Norte | Sur | *d* | 95 % CI | *p*_Holm |
|---|---|---|---|---|---|---|
| **coda** | CoG (Hz) | 4985 | 4468 | **+0.46** | [+0.27, +0.66] | **< 0.001** |
| coda | duration (s) | 0.102 | 0.094 | +0.26 | [+0.05, +0.47] | 0.026 |
| onset | CoG (Hz) | 5073 | 5140 | −0.06 | [−0.24, +0.13] | 0.496 |
| onset | duration (s) | 0.100 | 0.105 | −0.27 | [−0.45, −0.07] | 0.025 |

**Dialect × position on CoG** (`s8_interaction.csv`, mixed model, 601 speakers): +596 Hz,
SE 66, *p* < 0.001.

**Weakening index**, onset CoG − coda CoG per speaker (`s8_weakening_index.csv`):

| variety | mean | speakers |
|---|---|---|
| Norte | +100 Hz | 367 |
| México | +374 Hz | 245 |
| Sur | +691 Hz | 164 |

Sur vs Norte *d* = +0.85. Same ordering as v2.0 (Norte < México < Sur), with smaller absolute
values because the measurement changed (§5).

---

## 5. Measurement validity

**/s/ measurement.** With the new measurement (48 kHz mp3, middle 50 %, 1–11 kHz band) onset /s/
has a median CoG of 5,231 Hz (5th percentile 3,047 Hz). v2.0 measured on the 16 kHz WAV with the
full spectrum: onset median 3,776 Hz, and 423 onset tokens below 500 Hz, which cannot be an /s/
(`archive_ssp/v2_output/data/sibilant_tokens.csv`).

**Ceiling search on synthetic vowels** (`tests/test_pipeline.py`, `test_search_on_synthetic_vowels`;
male-like vowels, 4 tokens each, 2 % jitter). The search chose 4300 Hz.

| vowel | F1 error, search | F1 error, 5000 Hz | F2 error, search | F2 error, 5000 Hz |
|---|---|---|---|---|
| i | +10.3 % | +13.8 % | **−17.3 %** | −3.4 % |
| e | +5.0 % | +6.7 % | −4.3 % | −1.0 % |
| a | +0.4 % | +0.1 % | +2.9 % | +2.4 % |
| o | +6.0 % | +8.9 % | +8.1 % | +10.1 % |
| u | **+23.0 %** | +30.5 % | +13.4 % | +12.2 % |

The search reduces the F1 overestimate of close vowels but underestimates F2 of /i/. It is not
better everywhere, and that is why the sensitivity analysis with the fixed ceiling exists.

**Track check on 29 vowels** (`src/check_tracks.py` → `output/track_check/`, done 2026-10-23).
20 random stressed vowels (4 per vowel), 3 speakers at each ceiling edge, and one token from
each of the 3 cells with \|*z*\| > 4. Each one is drawn as a spectrogram with the Burg tracks.
**This is a visual check by Claude from the images, not a hand measurement in Praat**; the
CSV has empty `F1_praat` / `F2_praat` columns and `open_in_praat.praat` opens each vowel for
that.

| group | looks right | doubtful | wrong |
|---|---|---|---|
| random (20) | 14 | 5 | 1 |
| ceiling at 4000 Hz (3) | 2 | 1 | 0 |
| ceiling at 6500 Hz (3) | 1 | 1 | 1 bad token |
| extreme H1 cells (3) | 2 | 0 | 1 |

- Both clear errors are /i/ with F2 tracked at ~1050–1080 Hz instead of ~2400 Hz (#9
  *provincia*, #28 *libro*). That matches the F2 /i/ underestimate in the synthetic test, and it
  is one reason F2 /i/ equivalence depends on a single extreme cell (§6).
- "Doubtful" means measured mid-transition, a 30 ms vowel, or a vowel interval that starts in
  silence, not an obviously wrong value.
- At the 4000 Hz floor the F1/F2 values look plausible, but F3 collapses towards F2.
- The bad token is the English loanword *itunes*, which the stressed-vowel filter let through.

None of this changes the analysis (PLAN_v3.md §4.6), but about 1 in 20 random tokens is
mismeasured, and /i/ F2 is the measure to trust least.

---

## 6. Deviations and robustness

Full account in PLAN_v3.md §4.5. Three runs on the same day:

| run | what changed | kept in |
|---|---|---|
| first run | as planned | `output/v3_first_run/` |
| 500 Hz check | speakers dropped if F2 /i/ − /u/ < 500 Hz or F1 /a/ − F1 /i, u/ < 100 Hz | `output/v3_qc500/` |
| **final** | same check with 0 Hz thresholds (only impossible triangles) | `output/tables/` |

Why: the first run had Lobanov values up to *z* = 14.5, from speakers whose track put every vowel
at the same F2 (one: all five between 990 and 1040 Hz). The 500 Hz check removed 66 speakers,
unevenly (7.7 % Norte, 11.9 % México, 3.7 % Sur), on the same F2 values H1 tests, so it was
replaced before its successor was run. I also tried a ratio criterion for the ceiling search; it
failed the synthetic test (it chose 6400 Hz, with +629 % F1 error on /i/) and was never used on
the real data.

**H1 equivalence by measure across runs** (✓ = equivalent; *d* in brackets):

| measure | first run | 500 Hz check | final | final, sex ceiling |
|---|---|---|---|---|
| F1 /a/ | − (−0.17) | − (−0.15) | − (−0.16) | − (−0.19) |
| F1 /e/ | ✓ | ✓ | ✓ | ✓ |
| F1 /i/ | ✓ | ✓ | ✓ | ✓ |
| F1 /o/ | ✓ | ✓ | ✓ | ✓ |
| F1 /u/ | − | − | − | ✓ |
| F2 /a/ | − (+0.17) | − (+0.27, *p*_Holm = 0.035) | − (+0.17) | ✓ (+0.07) |
| F2 /e/ | ✓ | − (−0.20) | ✓ | − (−0.20) |
| F2 /i/ | ✓ | − (−0.19) | ✓ | ✓ |
| F2 /o/ | − (−0.15) | ✓ | − (−0.17) | − (−0.17) |
| F2 /u/ | ✓ | ✓ | ✓ | ✓ |
| **equivalent / different** | 6 / 0 | 5 / 1 | 6 / 0 | 7 / 0 |

(source: `s6_h1_norte_sur.csv` in each run folder, and `s6_h1_norte_sur_sexceil.csv`)

Robust: F1 /e/, F1 /i/, F1 /o/ and F2 /u/ are equivalent in every run. The only significant
difference anywhere (F2 /a/) appears in the run with the flawed exclusion rule and nowhere else.

One more check on the final run: 3 of the 3,431 cells still have \|*z*\| > 4. Without them, F2 /i/
becomes inconclusive (*d* = −0.21) and the count is 5 equivalent / 0 different.

**H2 primary across runs:**

| run | *d* | 95 % CI | *p* |
|---|---|---|---|
| first run | +0.19 | [−0.06, +0.44] | 0.146 |
| 500 Hz check | +0.25 | [−0.00, +0.50] | 0.052 |
| **final** | **+0.19** | **[−0.06, +0.44]** | **0.146** |
| final, sex ceiling (`s7_h2_primary_sexceil.csv`) | +0.29 | [+0.04, +0.55] | 0.026 |

The direction is the same everywhere, and the effect is small. The planned test is not
significant. The fixed-ceiling version is, but it is the sensitivity analysis, so it cannot
replace the planned result.

H2a does not depend on the vowel QC and is identical in all runs.

---

## 7. Claims

**Supported**
- Norte and Sur stressed vowels are equivalent within *d* = ±0.30 on F1 of /e i o/ and F2 of /u/,
  in every run.
- No Norte–Sur vowel measure differs significantly after correction (except one measure in one
  run with a flawed exclusion rule).
- Southern speakers weaken coda /s/, and not onset /s/: *d* = +0.46 in codas, interaction +596 Hz.
  The three varieties line up Norte < México < Sur.

**Not supported, or only suggested**
- Compensatory opening before /s/ (H2): right direction, *d* ≈ +0.2, not significant in the
  planned test. Descriptively, only Sur /e/ opens; for /a/ the difference is that Norte and México
  close and Sur does not.
- Equivalence for F1 /a/, F1 /u/, F2 /a/, F2 /o/: inconclusive. Small differences (\|*d*\| < 0.3)
  remain possible.
- A strong México calibration: the Peninsular–Mexican differences are also small.
- Anything about duration: no speech-rate normalisation.

## 8. Figures

| file | shows |
|---|---|
| `fig_vowel_space.png` | stressed vowel space, Norte/Sur speaker means and ellipses; three-variety means |
| `fig_equivalence.png` | H1: Norte–Sur *d* with 90 % CIs against ±0.30; each Peninsular variety vs México |
| `fig_plural.png` | H2: plural − singular F1 and F2 shift by vowel and variety |
| `fig_sibilants.png` | H2a: /s/ CoG by position and variety; weakening index |
| `fig_ceilings.png` | chosen formant ceiling by declared sex |
| `fig_data_flow.png` | corpus → tokens, and the QC funnel |
