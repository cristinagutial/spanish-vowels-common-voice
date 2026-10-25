# Results

All numbers come from `output/tables/` (log: `output/run_v3.log`). Design in
[PLAN_v3.md](PLAN_v3.md). The longer version with every run is `archive_v3/RESULTS_full.md`.

## Summary

| | result | verdict |
|---|---|---|
| **H1** stressed vowels, Norte vs Sur | 6 of 10 measures equivalent within *d* = ±0.30, none different | partly supported |
| México reference | the only significant vowel differences are Peninsular–Mexican, but they are small too | weak |
| **H2** vowel before word-final /s/ | *d* = +0.19, 95 % CI [−0.06, +0.44], *p* = 0.146 | not supported; right direction |
| **H2a** coda /s/ | coda *d* = +0.46, onset *d* = −0.06, interaction +596 Hz | supported |

## Data

| | |
|---|---|
| Clips sampled / aligned | 5,577 / 5,572 |
| Vowel tokens after QC | 97,647 of 100,053 |
| Usable speakers (Norte / Sur / México) | 388 / 161 / 249 |
| Dropped for an impossible vowel triangle | 7 (3 Norte, 4 México) |

## H1 — stressed vowels

23,457 stressed tokens in content words, one mean per speaker and vowel. Lobanov z,
*d* = Norte − Sur. Source: `s6_h1_norte_sur.csv`, `fig_equivalence.png`, `fig_vowel_space.png`.

| measure | *d* | 90 % CI | *p*_Holm | equivalent |
|---|---|---|---|---|
| F1 /a/ | −0.16 | [−0.33, +0.01] | 0.85 | no |
| F1 /e/ | −0.05 | [−0.20, +0.11] | 1.00 | yes |
| F1 /i/ | +0.12 | [−0.04, +0.28] | 1.00 | yes |
| F1 /o/ | −0.08 | [−0.25, +0.07] | 1.00 | yes |
| F1 /u/ | −0.10 | [−0.32, +0.11] | 1.00 | no |
| F2 /a/ | +0.17 | [+0.03, +0.31] | 0.57 | no |
| F2 /e/ | −0.14 | [−0.27, −0.00] | 0.85 | yes |
| F2 /i/ | −0.10 | [−0.29, +0.02] | 1.00 | yes |
| F2 /o/ | −0.17 | [−0.32, −0.02] | 0.63 | no |
| F2 /u/ | +0.02 | [−0.20, +0.21] | 1.00 | yes |

No measure differs after correction. The four inconclusive ones are small (\|*d*\| ≤ 0.17), but
their intervals reach past ±0.30, so equivalence cannot be claimed for them. With 386 Norte and
160 Sur speakers the design detects *d* ≥ 0.28 with 80 % power (`s6_power.csv`).

**México reference** (`s6_h1_control.csv`): mean \|*d*\| is 0.110 for Norte–Sur, 0.163 for
Norte–México and 0.136 for Sur–México. Only Peninsular–Mexican contrasts reach significance
(Norte–México F1 /e/; Sur–México F2 /a/ and F2 /o/).

## H2 — vowel before word-final /s/

Word-final unstressed /a e o/ in content words, followed by a consonant or a pause. Shift =
*casas*-type minus *casa*-type, per speaker and vowel, Lobanov z (+ = more open). Source:
`s7_h2_primary.csv`, `s7_h2_descriptive.csv`, `fig_plural.png`.

| | Sur | Norte | *d* | 95 % CI | *p* |
|---|---|---|---|---|---|
| F1 shift, mean of /e o/ | +0.088 (n = 96) | −0.040 (n = 220) | +0.19 | [−0.06, +0.44] | 0.146 |

| F1 shift | /a/ | /e/ | /o/ |
|---|---|---|---|
| Norte | −0.36 | −0.03 | −0.05 |
| Sur | +0.01 | +0.16 | −0.00 |
| México | −0.16 | −0.22 | −0.14 |

Not significant: with 96 Sur speakers the test has 34 % power for an effect this size. The
direction is the predicted one, and it comes from /e/: only Sur /e/ opens before /s/. For /a/,
Norte and México close the vowel before /s/ and Sur does not.

## H2a — coda /s/

21,494 /s/ tokens, one mean per speaker and position. Source: `s8_sibilant_norte_sur.csv`,
`s8_interaction.csv`, `s8_weakening_index.csv`, `fig_sibilants.png`.

| position | Norte | Sur | *d* | *p*_Holm |
|---|---|---|---|---|
| coda CoG | 4985 Hz | 4468 Hz | +0.46 | < 0.001 |
| onset CoG | 5073 Hz | 5140 Hz | −0.06 | 0.50 |

Dialect × position: +596 Hz, *p* < 0.001. Weakening index (onset − coda CoG per speaker):
Norte +100 Hz, México +374 Hz, Sur +691 Hz.

## Limitations

- The data are read speech by volunteers with self-reported labels; "Sur" is a broad label.
- Formant tracking is not perfect. In a visual check of 20 random vowels, 1 had a clearly wrong
  F2 (an /i/ tracked at about 1050 Hz), and /i/ F2 is the measure to trust least.
- Differences below *d* ≈ 0.3 cannot be ruled out.

## Figures

| file | shows |
|---|---|
| `fig_vowel_space.png` | stressed vowel space: Norte/Sur speaker means; means of all three varieties |
| `fig_equivalence.png` | H1: Norte–Sur *d* with 90 % CIs against ±0.30; each Peninsular variety vs México |
| `fig_plural.png` | H2: *casas* − *casa* shift in F1 and F2 by vowel and variety |
| `fig_sibilants.png` | H2a: /s/ CoG by position and variety; weakening index |
| `fig_data_flow.png` | corpus → tokens, and the QC steps |
