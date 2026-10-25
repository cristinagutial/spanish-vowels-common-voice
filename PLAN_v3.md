# Plan — Northern vs Southern Peninsular Spanish vowels

Cristina Gutiérrez Alcalá · Experimental Phonetics (Katrin Schneider) · *Topics in Phonetics*, winter 2026/27

## Question

Do Northern and Southern Peninsular Spanish differ in vowel quality, either across the stressed
vowel space or before word-final /s/, where descriptions of Southern Spanish say the vowel
opens when the /s/ weakens?

## Hypotheses

- **H1.** Stressed vowels do not differ between Norte and Sur by more than *d* = ±0.30
  (equivalence test). Mexican Spanish is the reference: the same measures should pick up
  Peninsular–Mexican differences.
- **H2.** In Sur, word-final unstressed /e/ and /o/ are more open before a word-final /s/
  (*casas*) than without it (*casa*), compared with the same contrast in Norte.
- **H2a (check).** Sur weakens coda /s/: lower spectral centre of gravity in codas, no difference
  in onsets. H2 only makes sense if this holds.

## Data

Mozilla Common Voice Spanish 26.0, read speech with self-reported accent labels. Norte 469
speakers, Sur 188 (all that exist), México 300. Up to 8 clips per speaker, 5,577 clips. The
sample, the alignment and most of the code come from my Speech Signal Processing project.

## Method

1. Forced alignment with the Montreal Forced Aligner (`spanish_mfa`).
2. F1 and F2 at the vowel midpoint (Praat's Burg method through Parselmouth). The formant
   ceiling is chosen per speaker from 4000–6500 Hz, as the one with the least spread inside each
   vowel (after Escudero et al. 2009).
3. Quality control: range filters, outliers within speaker and vowel, and speakers with an
   impossible vowel triangle dropped.
4. Lobanov normalisation over each speaker's five vowel means.
5. /s/: spectral centre of gravity on the middle 50 % of the segment, 1–11 kHz, on the original
   48 kHz audio.

## Analysis

The speaker is the unit: one mean per speaker (and vowel) before any test. Welch tests,
Cohen's *d*, bootstrap confidence intervals over speakers, Holm correction within each family.

- **H1:** 10 measures (F1 and F2 of the five vowels), stressed vowels in content words only. A
  measure counts as equivalent when the 90 % CI of *d* lies inside ±0.30.
- **H2:** one test. Per speaker, the F1 shift *casas* − *casa*, averaged over /e o/, Sur vs Norte.
- **H2a:** Norte vs Sur by position, and the dialect × position interaction.

## Changes made during the analysis

- After the first run, a few speakers turned out to have collapsed formant tracks (every vowel
  at about the same F2), so I added the vowel-triangle check. A stricter version of it removed
  real speakers, and more in Norte than in Sur, so I kept only the rule that drops impossible
  triangles (7 speakers).
- The ceiling grid starts at 4000 Hz instead of the planned 4500 Hz, because a pilot put a male
  speaker on the 4500 Hz floor.

The full record of every run is in `archive_v3/PLAN_v3_full.md`.

## Schedule

| when | what |
|---|---|
| Oct 2026 | question, hypotheses, sample and pipeline from the SSP project |
| by 10.12.2026 | analyses and figures ready (course session on analysis and documentation) |
| 14.01.2027 | discussion with the lecturer |
| 21 or 28.01.2027 | presentation |
| Feb–Mar 2027 | results session |

## Limitations

- Read speech by volunteers, uncontrolled microphones, MP3 compression.
- Self-reported, coarse labels: "Sur" mixes Eastern and Western Andalusian, Extremaduran and
  Murcian, and the vowel opening is described mainly for Eastern Andalusian.
- Forced alignment places an /s/ boundary even when the /s/ is gone.
- No speech-rate normalisation, so no claims about duration.
