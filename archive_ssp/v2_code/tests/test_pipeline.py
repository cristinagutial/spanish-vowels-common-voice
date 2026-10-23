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
import parselmouth

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.s4_extract import coda_s_context, is_consonant, stressed_nucleus_index
from src.s5_normalize import bark
from src.utils import cohens_d, gini, holm_correction, assign_dialect, normalise_sex


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
        """A mismatched ceiling biases the estimate, which is why it depends on sex."""
        sound = synthesise_vowel(700.0, 1400.0)
        mid = sound.duration / 2
        low = sound.to_formant_burg(max_number_of_formants=5,
                                    maximum_formant=3000.0).get_value_at_time(1, mid)
        correct = sound.to_formant_burg(max_number_of_formants=5,
                                        maximum_formant=5000.0).get_value_at_time(1, mid)
        self.assertNotAlmostEqual(low, correct, delta=1.0)


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
    """Coda /s/ detection, which the aspiration analysis depends on."""

    def test_word_final_s_is_coda(self):
        phones = [(0.0, 0.1, "a"), (0.1, 0.2, "s")]
        self.assertTrue(coda_s_context(phones, 0, word_end_index=1))

    def test_s_before_consonant_is_coda(self):
        # 'este' -> e s t̪ e : la /s/ cierra la primera silaba
        phones = [(0.0, 0.1, "e"), (0.1, 0.2, "s"), (0.2, 0.3, "t̪"), (0.3, 0.4, "e")]
        self.assertTrue(coda_s_context(phones, 0, word_end_index=3))

    def test_s_before_vowel_is_onset(self):
        # 'casa' -> k a s a : la /s/ abre la segunda silaba, no es coda
        phones = [(0.0, 0.1, "k"), (0.1, 0.2, "a"), (0.2, 0.3, "s"), (0.3, 0.4, "a")]
        self.assertFalse(coda_s_context(phones, 1, word_end_index=3))

    def test_non_s_following(self):
        phones = [(0.0, 0.1, "a"), (0.1, 0.2, "n")]
        self.assertFalse(coda_s_context(phones, 0, word_end_index=1))

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

    def test_bark_is_monotone(self):
        values = bark(np.array([100.0, 500.0, 1000.0, 2000.0, 4000.0]))
        self.assertTrue(np.all(np.diff(values) > 0))


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
