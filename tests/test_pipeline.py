"""
Tests for the parts of the pipeline I can check against a known answer.

The main one builds a vowel with formants I choose and checks that the extractor
gets them back.

    ./speech/bin/python -m unittest discover -s tests -v
"""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import parselmouth

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.s4_extract import (_measure, best_ceiling, final_s_context, following_context,
                            is_consonant, s_position, stressed_nucleus_index)
from src.s5_normalize import lobanov, plausible_vowel_space
from src.s7_plural import h2_tokens
from src.utils import (bootstrap_d, cohens_d, compare, gini, holm_correction, assign_dialect,
                       normalise_sex, percentile_ci, tost_welch)


def synthesise_vowel(f1, f2, f0=120.0, duration=0.5, fs=16000.0,
                     higher=(2800.0, 3500.0, 4500.0)):
    """
    Build a vowel by exciting a bank of resonators with a pulse train.

    The signal has its peaks exactly at the frequencies I ask for, so the tracker
    can be compared against the truth. F3-F5 are included even though only F1 and
    F2 are tested, because Praat fits a fixed number of poles and the spare ones
    would otherwise drift.
    """
    n = int(duration * fs)
    signal = np.zeros(n)
    period = int(fs / f0)
    tau = np.arange(period) / fs

    # los bandwidths se ensanchan con la frecuencia, como en las voces de verdad
    resonators = [(f1, 60.0, 1.0), (f2, 90.0, 0.5)]
    resonators += [(f, 120.0 + 20.0 * i, 0.25 / (i + 1))
                   for i, f in enumerate(higher)]

    for freq, bandwidth, amplitude in resonators:
        response = np.exp(-np.pi * bandwidth * tau) * np.sin(2 * np.pi * freq * tau)
        for start in range(0, n - period, period):
            signal[start:start + period] += amplitude * response

    signal /= np.max(np.abs(signal))
    return parselmouth.Sound(signal, sampling_frequency=fs)


class TestFormantExtraction(unittest.TestCase):
    """The extractor has to recover the formants it is given."""

    # valores tipicos de las vocales del español (Quilis y Esgueva 1983, hombres)
    TARGETS = {
        "i": (300.0, 2200.0),
        "e": (450.0, 1900.0),
        "a": (700.0, 1400.0),
        "o": (470.0, 1000.0),
        "u": (320.0, 800.0),
    }

    def measure(self, f1, f2):
        sound = synthesise_vowel(f1, f2)
        formant = sound.to_formant_burg(max_number_of_formants=5, maximum_formant=5000.0)
        mid = sound.duration / 2
        return formant.get_value_at_time(1, mid), formant.get_value_at_time(2, mid)

    def test_recovers_open_vowels_accurately(self):
        """Where F1 is well clear of F0, tracking is accurate."""
        for vowel in ("a", "e"):
            with self.subTest(vowel=vowel):
                true_f1, true_f2 = self.TARGETS[vowel]
                got_f1, got_f2 = self.measure(true_f1, true_f2)
                self.assertAlmostEqual(got_f1, true_f1, delta=0.08 * true_f1,
                                       msg=f"/{vowel}/ F1: expected {true_f1}, got {got_f1:.0f}")
                self.assertAlmostEqual(got_f2, true_f2, delta=0.08 * true_f2,
                                       msg=f"/{vowel}/ F2: expected {true_f2}, got {got_f2:.0f}")

    def test_close_vowel_f1_is_overestimated(self):
        """
        Documents a real limitation instead of hiding it.

        Linear prediction overestimates F1 for close vowels, because F1 gets close
        to F0 and in /u/ the two poles interfere. It happens equally in both
        dialects, so it cannot create a difference, but /u/ is the least reliable
        measurement in the study.
        """
        errors = {}
        for vowel in ("i", "u"):
            true_f1, true_f2 = self.TARGETS[vowel]
            got_f1, _ = self.measure(true_f1, true_f2)
            errors[vowel] = (got_f1 - true_f1) / true_f1

        for vowel, error in errors.items():
            self.assertGreater(error, 0.05,
                               f"/{vowel}/ F1 bias was expected to be upward, got {error:+.1%}")
        self.assertGreater(errors["u"], errors["i"],
                           "/u/ should be the worst case, since its F1 and F2 are closest")

    def test_bias_is_dialect_neutral(self):
        """
        Measuring the same target twice gives the same answer.

        Trivial, but it is what lets me compare dialects at all: whatever error
        the tracker has, it has it equally in both groups.
        """
        first = self.measure(*self.TARGETS["a"])
        second = self.measure(*self.TARGETS["a"])
        self.assertAlmostEqual(first[0], second[0], places=6)
        self.assertAlmostEqual(first[1], second[1], places=6)

    def test_ceiling_affects_estimates(self):
        """A mismatched ceiling biases the estimate, which is why it is chosen per speaker."""
        sound = synthesise_vowel(700.0, 1400.0)
        mid = sound.duration / 2
        low = sound.to_formant_burg(max_number_of_formants=5,
                                    maximum_formant=3000.0).get_value_at_time(1, mid)
        correct = sound.to_formant_burg(max_number_of_formants=5,
                                        maximum_formant=5000.0).get_value_at_time(1, mid)
        self.assertNotAlmostEqual(low, correct, delta=1.0)


class TestCeilingSearch(unittest.TestCase):
    """The per-speaker ceiling is the one with the least spread inside each vowel."""

    def test_picks_the_least_spread(self):
        rng = np.random.default_rng(1)
        ceilings = np.array([4500.0, 5000.0, 5500.0])
        vowels = np.repeat(["a", "e", "i"], 6)
        base_f1 = np.repeat([700.0, 450.0, 300.0], 6)
        base_f2 = np.repeat([1400.0, 1900.0, 2200.0], 6)
        # el techo del medio es el estable; los otros tienen mucho ruido
        noise = np.array([0.20, 0.01, 0.20])
        f1 = base_f1[:, None] * (1 + rng.normal(0, 1, (18, 3)) * noise)
        f2 = base_f2[:, None] * (1 + rng.normal(0, 1, (18, 3)) * noise)
        self.assertEqual(best_ceiling(np.abs(f1), np.abs(f2), vowels, ceilings), 5000.0)

    def test_search_on_synthetic_vowels(self):
        """
        Documents what the search does on vowels with a known answer.

        It lowers the F1 overestimate for /u/, but it costs accuracy on F2 of /i/,
        so it is not better everywhere. RESULTS.md reports the full table.
        """
        rng = np.random.default_rng(0)
        grid = np.array(config.CEILING_GRID)
        tokens, labels, truth = [], [], []
        for vowel, (f1, f2) in TestFormantExtraction.TARGETS.items():
            for _ in range(4):
                jitter = 1 + rng.normal(0, 0.02, 2)
                sound = synthesise_vowel(f1 * jitter[0], f2 * jitter[1])
                tokens.append(np.array([_measure(sound, [sound.duration / 2], c) for c in grid])[:, :, 0])
                labels.append(vowel)
                truth.append((f1 * jitter[0], f2 * jitter[1]))
        values, labels, truth = np.array(tokens), np.array(labels), np.array(truth)
        chosen = best_ceiling(values[:, :, 0], values[:, :, 1], labels, grid)
        k, k5000 = int(np.where(grid == chosen)[0][0]), int(np.where(grid == 5000.0)[0][0])

        def error(col, formant, vowel):
            mask = labels == vowel
            return (values[mask, col, formant] / truth[mask, formant] - 1).mean()

        self.assertLess(error(k, 0, "u"), error(k5000, 0, "u"))
        self.assertLess(error(k, 1, "i"), error(k5000, 1, "i"))   # F2 de /i/ sale mas baja

    def test_declines_with_too_few_vowels(self):
        """With fewer than MIN_VOWELS_FOR_CEILING usable vowels it returns None."""
        f = np.full((4, 2), 500.0)
        self.assertIsNone(best_ceiling(f, f * 3, np.array(["a"] * 4), [4500.0, 5000.0]))

    def test_ignores_tokens_undefined_at_some_ceiling(self):
        """Every ceiling is judged on the same tokens."""
        f1 = np.full((9, 2), 500.0) + np.arange(9)[:, None]
        f2 = f1 * 3
        f1[0, 1] = np.nan
        vowels = np.repeat(["a", "e", "o"], 3)
        self.assertIsNone(best_ceiling(f1, f2, vowels, [4500.0, 5000.0]))


class TestStressAssignment(unittest.TestCase):
    """Spanish stress rules, applied to a known syllable count."""

    def test_penultimate_default(self):
        # las que acaban en vocal, -n o -s son llanas
        self.assertEqual(stressed_nucleus_index("casa", 2), 0)
        self.assertEqual(stressed_nucleus_index("hablan", 2), 0)
        self.assertEqual(stressed_nucleus_index("libros", 2), 0)
        self.assertEqual(stressed_nucleus_index("ventana", 3), 1)

    def test_final_default(self):
        # las demas son agudas
        self.assertEqual(stressed_nucleus_index("comer", 2), 1)
        self.assertEqual(stressed_nucleus_index("papel", 2), 1)
        self.assertEqual(stressed_nucleus_index("feliz", 2), 1)

    def test_written_accent_overrides(self):
        self.assertEqual(stressed_nucleus_index("árbol", 2), 0)
        self.assertEqual(stressed_nucleus_index("canción", 2), 1)
        self.assertEqual(stressed_nucleus_index("médico", 3), 0)

    def test_monosyllable(self):
        self.assertEqual(stressed_nucleus_index("pan", 1), 0)

    def test_declines_when_counts_disagree(self):
        """
        When the spelling and the aligner disagree about how many syllables there
        are, the word is left unlabelled rather than guessed.
        """
        self.assertIsNone(stressed_nucleus_index("día", 3))


class TestContext(unittest.TestCase):
    """What follows a segment, which H2 and H2a depend on."""

    def test_gap_is_a_pause(self):
        # el silencio se quita al leer el TextGrid y deja un hueco
        phones = [(0.0, 0.1, "a"), (0.3, 0.4, "k")]
        self.assertEqual(following_context(phones, 0), "pause")

    def test_last_phone_is_followed_by_a_pause(self):
        self.assertEqual(following_context([(0.0, 0.1, "a")], 0), "pause")

    def test_contiguous_phones(self):
        phones = [(0.0, 0.1, "s"), (0.1, 0.2, "a"), (0.2, 0.3, "t̪"), (0.3, 0.4, "j")]
        self.assertEqual(following_context(phones, 0), "vowel")
        self.assertEqual(following_context(phones, 1), "consonant")
        self.assertEqual(following_context(phones, 2), "vowel")   # semiconsonante = vocalico

    def test_unknown_word_is_other(self):
        phones = [(0.0, 0.1, "a"), (0.1, 0.2, "spn")]
        self.assertEqual(following_context(phones, 0), "other")

    def test_final_s_context(self):
        # 'casas tienen' -> ... a s | t̪ ... : vocal + /s/ final + consonante
        phones = [(0.0, 0.1, "k"), (0.1, 0.2, "a"), (0.2, 0.3, "s"), (0.3, 0.4, "a"),
                  (0.4, 0.5, "s"), (0.5, 0.6, "t̪")]
        self.assertEqual(final_s_context(phones, 3, word_end_index=4), "consonant")
        # la primera /a/ va seguida de /s/, pero esa /s/ no es final de palabra
        self.assertIsNone(final_s_context(phones, 1, word_end_index=4))

    def test_word_final_vowel_has_no_final_s(self):
        phones = [(0.0, 0.1, "k"), (0.1, 0.2, "a"), (0.2, 0.3, "s"), (0.3, 0.4, "a")]
        self.assertIsNone(final_s_context(phones, 3, word_end_index=3))

    def test_s_positions(self):
        # 'este' -> la /s/ cierra la silaba
        este = [(0.0, 0.1, "e"), (0.1, 0.2, "s"), (0.2, 0.3, "t̪"), (0.3, 0.4, "e")]
        self.assertEqual(s_position(este, 1, word_final=False), "coda")
        # 'casa' -> la /s/ abre la silaba
        casa = [(0.0, 0.1, "k"), (0.1, 0.2, "a"), (0.2, 0.3, "s"), (0.3, 0.4, "a")]
        self.assertEqual(s_position(casa, 2, word_final=False), "onset")
        # 'los amigos' -> /s/ final ante vocal, se resilabifica
        los_amigos = [(0.0, 0.1, "l"), (0.1, 0.2, "o"), (0.2, 0.3, "s"), (0.3, 0.4, "a")]
        self.assertEqual(s_position(los_amigos, 2, word_final=True), "prevocalic_final")
        # 'los' + pausa
        los = [(0.0, 0.1, "l"), (0.1, 0.2, "o"), (0.2, 0.3, "s"), (0.5, 0.6, "a")]
        self.assertEqual(s_position(los, 2, word_final=True), "coda")

    def test_consonant_classification(self):
        self.assertTrue(is_consonant("s"))
        self.assertTrue(is_consonant("t̪"))
        self.assertFalse(is_consonant("a"))
        self.assertFalse(is_consonant("j"))   # semiconsonante, no es nucleo
        self.assertFalse(is_consonant("spn"))  # asi marca el alineador el ruido


class TestStatistics(unittest.TestCase):

    def test_cohens_d_sign_and_scale(self):
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertAlmostEqual(cohens_d(a, a), 0.0)
        self.assertGreater(cohens_d(a + 10, a), 0)
        self.assertLess(cohens_d(a, a + 10), 0)

    def test_cohens_d_known_value(self):
        rng = np.random.default_rng(0)
        a = rng.normal(0, 1, 20000)
        b = rng.normal(1, 1, 20000)
        self.assertAlmostEqual(cohens_d(a, b), -1.0, delta=0.05)

    def test_holm_is_monotone_and_conservative(self):
        raw = [0.01, 0.02, 0.03, 0.04]
        adjusted = holm_correction(raw)
        self.assertTrue(np.all(adjusted >= np.array(raw)))
        self.assertTrue(np.all(np.diff(adjusted) >= -1e-12))
        self.assertAlmostEqual(adjusted[0], 0.04)

    def test_holm_caps_at_one(self):
        self.assertTrue(np.all(holm_correction([0.6, 0.7, 0.8]) <= 1.0))

    def test_gini(self):
        self.assertAlmostEqual(gini([5, 5, 5, 5]), 0.0, places=6)
        # si uno solo aporta todo, la desigualdad es maxima
        self.assertGreater(gini([0, 0, 0, 100]), 0.7)

    def test_tost_accepts_equal_groups(self):
        rng = np.random.default_rng(2)
        a, b = rng.normal(0, 1, 800), rng.normal(0, 1, 800)
        self.assertLess(tost_welch(a, b, 0.3), 0.05)

    def test_tost_rejects_different_groups(self):
        rng = np.random.default_rng(3)
        a, b = rng.normal(0, 1, 800), rng.normal(1, 1, 800)
        self.assertGreater(tost_welch(a, b, 0.3), 0.5)

    def test_bootstrap_interval_covers_the_estimate(self):
        rng = np.random.default_rng(4)
        a, b = rng.normal(0.5, 1, 200), rng.normal(0, 1, 200)
        lo, hi = percentile_ci(bootstrap_d(a, b), 0.95)
        self.assertLess(lo, cohens_d(a, b))
        self.assertGreater(hi, cohens_d(a, b))
        lo90, hi90 = percentile_ci(bootstrap_d(a, b), 0.90)
        self.assertTrue(lo <= lo90 and hi90 <= hi)

    def test_compare_equivalence_flag(self):
        rng = np.random.default_rng(5)
        same = compare(rng.normal(0, 1, 600), rng.normal(0, 1, 600), ("x", "y"), bound_d=0.3)
        apart = compare(rng.normal(1, 1, 600), rng.normal(0, 1, 600), ("x", "y"), bound_d=0.3)
        self.assertTrue(same["equivalent"])
        self.assertFalse(apart["equivalent"])


class TestLobanov(unittest.TestCase):
    """Lobanov on vowel means: every vowel weighs the same."""

    def frame(self, n_a):
        rows = [("s1", "a", 800.0)] * n_a
        rows += [("s1", v, f) for v, f in [("e", 500.0), ("i", 300.0), ("o", 520.0), ("u", 340.0)] for _ in range(2)]
        return pd.DataFrame(rows, columns=["client_id", "vowel", "F1_50"])

    def test_token_frequency_does_not_move_the_scale(self):
        few, _ = lobanov(self.frame(2), "F1_50")
        many, _ = lobanov(self.frame(40), "F1_50")
        self.assertAlmostEqual(few.iloc[0], many.iloc[0], places=9)

    def test_speaker_missing_a_vowel_is_not_normalised(self):
        df = self.frame(2)
        df = df[df["vowel"] != "u"]
        z, eligible = lobanov(df, "F1_50")
        self.assertTrue(z.isna().all())
        self.assertEqual(len(eligible), 0)

    def test_collapsed_vowel_space_is_dropped(self):
        """A track with every vowel at F2 = 1000 Hz is a tracker failure, not an accent."""
        values = {"a": (700, 1400), "e": (450, 1900), "i": (300, 2200), "o": (470, 1000), "u": (320, 800)}
        rows = [("good", v, f1, f2) for v, (f1, f2) in values.items()]
        rows += [("collapsed", v, f1, 1000.0 + i) for i, (v, (f1, _)) in enumerate(values.items())]
        df = pd.DataFrame(rows, columns=["client_id", "vowel", "F1_50", "F2_50"])
        self.assertEqual(plausible_vowel_space(df), {"good"})


class TestH2Tokens(unittest.TestCase):
    """The 'casa' / 'casas' token sets."""

    def test_selection(self):
        base = dict(client_id="s1", dialecto="sur", lobanov_ok=True, stressed=False,
                    n_syllables=2, F1_50_lob=0.0, F2_50_lob=0.0, duration=0.06)
        rows = [
            dict(base, word="casas", vowel="a", final_s=True, after_s="consonant",
                 word_final_vowel=False, following="consonant"),
            dict(base, word="casa", vowel="a", final_s=False, after_s=None,
                 word_final_vowel=True, following="pause"),
            # /s/ final ante vocal: se resilabifica, fuera
            dict(base, word="casas", vowel="a", final_s=True, after_s="vowel",
                 word_final_vowel=False, following="consonant"),
            # palabra gramatical, fuera
            dict(base, word="estas", vowel="a", final_s=True, after_s="pause",
                 word_final_vowel=False, following="consonant"),
            # tonica, fuera
            dict(base, word="está", vowel="a", stressed=True, final_s=False, after_s=None,
                 word_final_vowel=True, following="pause"),
        ]
        selected = h2_tokens(pd.DataFrame(rows))
        self.assertEqual(list(selected["type"]), ["plural", "singular"])


class TestMetadata(unittest.TestCase):

    def test_single_dialect_label(self):
        self.assertEqual(assign_dialect("España: Norte peninsular (Asturias)"), "norte")
        self.assertEqual(assign_dialect("España: Sur peninsular (Andalucia)"), "sur")

    def test_ambiguous_label_is_rejected(self):
        """Speakers who tick two regions cannot be assigned to either."""
        both = "España: Norte peninsular (Asturias)|España: Sur peninsular (Andalucia)"
        self.assertIsNone(assign_dialect(both))

    def test_missing_label(self):
        self.assertIsNone(assign_dialect(None))
        self.assertIsNone(assign_dialect("Rioplatense: Argentina"))

    def test_sex_normalisation(self):
        self.assertEqual(normalise_sex("male_masculine"), "male")
        self.assertEqual(normalise_sex("female_feminine"), "female")
        self.assertEqual(normalise_sex(None), "unknown")
        self.assertEqual(normalise_sex("other"), "unknown")


if __name__ == "__main__":
    unittest.main(verbosity=2)
