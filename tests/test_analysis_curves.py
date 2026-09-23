from __future__ import annotations

import unittest
from math import isclose, log

from epiaudio.analysis import curves

NATS_PER_BIT = log(2.0)


class CeilingTests(unittest.TestCase):
    def test_ceiling_is_ln_vocab(self):
        self.assertAlmostEqual(curves.ceiling_loss_nats(1024), log(1024))
        self.assertAlmostEqual(curves.ceiling_loss_nats(117_649), log(117_649))

    def test_degenerate_vocabulary_rejected(self):
        with self.assertRaises(ValueError):
            curves.ceiling_loss_nats(1)


class WindowTests(unittest.TestCase):
    def test_tail_fraction_matches_the_sweep_eval_cadence(self):
        """num_evals=20 across epiaudio/sweeps/, so the estimator window is 1/20."""
        self.assertAlmostEqual(curves.tail_fraction_for(20), 0.05)
        with self.assertRaises(ValueError):
            curves.tail_fraction_for(0)

    def test_estimator_window_averages_more_tokens_than_the_eval_set(self):
        """The as-implemented term is lower-variance than a held-out estimate."""
        train_tokens = 31_250_000
        window_tokens = train_tokens * curves.tail_fraction_for(20)
        self.assertAlmostEqual(window_tokens, 1_562_500)
        self.assertGreater(window_tokens / 131_072, 11.0)

    def test_tail_window_is_at_least_one_point(self):
        self.assertEqual(list(curves.tail_window([1.0, 2.0], tail_fraction=0.01)), [2.0])

    def test_tail_fraction_bounds_enforced(self):
        with self.assertRaises(ValueError):
            curves.tail_window([1.0], tail_fraction=0.0)
        with self.assertRaises(ValueError):
            curves.tail_window([1.0], tail_fraction=1.5)

    def test_empty_curve_rejected(self):
        with self.assertRaises(ValueError):
            curves.trajectory_mean([])


class MonotonicityTests(unittest.TestCase):
    def test_clean_descent_is_non_increasing(self):
        self.assertTrue(curves.is_non_increasing([6.0, 5.0, 4.0, 3.0]))

    def test_small_noise_absorbed_by_tolerance(self):
        self.assertTrue(
            curves.is_non_increasing([6.0, 5.0, 5.02, 4.0], tolerance=0.01)
        )

    def test_sustained_rise_not_absorbed(self):
        self.assertFalse(
            curves.is_non_increasing([3.0, 4.0, 5.0, 6.0], tolerance=0.01)
        )


class ShapeTests(unittest.TestCase):
    def _shape(self, losses: list[float], vocab: int = 1024) -> curves.LossCurveShape:
        stats = curves.summarize_curve(losses, vocab_size=vocab)
        return curves.classify_shape(losses, stats)

    def test_decreasing(self):
        self.assertIs(
            self._shape([6.9, 5.0, 4.0, 3.5, 3.2, 3.1, 3.05, 3.0, 2.98, 2.97]),
            curves.LossCurveShape.DECREASING,
        )

    def test_plateau(self):
        losses = [6.9, 5.0, 4.0, 3.5, 3.4, 3.41, 3.39, 3.40, 3.38, 3.39]
        self.assertIn(
            self._shape(losses),
            (curves.LossCurveShape.DECREASING, curves.LossCurveShape.PLATEAU),
        )

    def test_late_rise(self):
        losses = [6.9, 5.0, 4.0, 3.4, 3.3, 3.6, 4.2, 5.0, 5.8, 6.4]
        self.assertIs(self._shape(losses), curves.LossCurveShape.LATE_RISE)

    def test_minimum_near_ceiling_takes_priority(self):
        """A curve whose best point never leaves ln(V) says nothing."""
        ceiling = log(1024)
        losses = [ceiling, ceiling - 0.001, ceiling, ceiling - 0.002, ceiling]
        self.assertIs(self._shape(losses), curves.LossCurveShape.MINIMUM_NEAR_CEILING)

    def test_minimum_near_ceiling_does_not_require_a_flat_shape(self):
        """The check only inspects the minimum, so a noisy curve qualifies too."""
        ceiling = log(1024)
        losses = [ceiling + 0.02, ceiling - 0.01, ceiling + 0.03, ceiling - 0.019,
                   ceiling + 0.01, ceiling, ceiling + 0.025, ceiling - 0.005]
        self.assertGreater(max(losses) - min(losses), 0.02)  # genuinely not flat
        self.assertIs(self._shape(losses), curves.LossCurveShape.MINIMUM_NEAR_CEILING)

    def test_plateau_requires_settling_near_the_minimum(self):
        """A real, non-monotone dip that recovers back near its minimum."""
        losses = [6.5, 5.0, 4.0, 3.0, 3.5, 3.1, 3.0, 3.05, 3.0, 3.02]
        self.assertFalse(curves.is_non_increasing(losses))  # 3.0 -> 3.5 breaks it
        self.assertIs(self._shape(losses), curves.LossCurveShape.PLATEAU)

    def test_irregular_curve_gets_other_not_a_silent_plateau(self):
        """A curve matching none of the positive checks must not default to PLATEAU.

        Regression test for the bug this module's PLATEAU used to have: it was
        an unconditional fallback, so an oscillating curve that never settles
        was mislabeled "descends early, then flat" purely because it also
        wasn't ceiling-bound, late-rising, or monotone.
        """
        losses = [6.5, 3.0, 6.0, 3.2, 5.8, 3.4, 5.5, 3.6, 5.0, 4.5]
        self.assertFalse(curves.is_non_increasing(losses))
        self.assertIs(self._shape(losses), curves.LossCurveShape.OTHER)


class DiagnosisTests(unittest.TestCase):
    def test_positive_model_bits_need_no_diagnosis(self):
        result = curves.diagnose(
            [6.9, 5.0, 4.0, 3.0], model_bits=1.0e6, vocab_size=1024
        )
        self.assertIs(result.mechanism, curves.NegativityMechanism.NOT_NEGATIVE)
        self.assertFalse(result.is_negative)
        self.assertFalse(result.is_data_property)

    def test_minimum_near_ceiling_is_the_only_data_property_verdict(self):
        ceiling = log(1024)
        result = curves.diagnose(
            [ceiling, ceiling - 0.001, ceiling, ceiling],
            model_bits=-500.0,
            vocab_size=1024,
        )
        self.assertIs(
            result.mechanism, curves.NegativityMechanism.NO_LEARNABLE_STRUCTURE
        )
        self.assertTrue(result.is_data_property)

    def test_late_rise_is_a_training_failure_not_a_data_property(self):
        result = curves.diagnose(
            [6.9, 5.0, 4.0, 3.4, 3.3, 3.6, 4.2, 5.0, 5.8, 6.4],
            model_bits=-2.0e6,
            vocab_size=1024,
        )
        self.assertIs(result.mechanism, curves.NegativityMechanism.LATE_DIVERGENCE)
        self.assertFalse(result.is_data_property)

    def test_tiny_negative_against_a_large_online_code_is_noise(self):
        losses = [6.9, 5.0, 4.0, 3.5, 3.3, 3.31, 3.29, 3.30, 3.31, 3.32]
        result = curves.diagnose(
            losses,
            model_bits=-1.0e3,
            vocab_size=1024,
            online_code_bits=2.7e8,
        )
        self.assertIs(
            result.mechanism, curves.NegativityMechanism.SIGN_UNSTABLE_NEAR_ZERO
        )
        self.assertFalse(result.is_data_property)

    def test_large_negative_without_a_matching_mechanism_is_flagged(self):
        losses = [6.9, 5.0, 4.0, 3.5, 3.3, 3.31, 3.29, 3.30, 3.31, 3.32]
        result = curves.diagnose(
            losses,
            model_bits=-5.0e7,
            vocab_size=1024,
            online_code_bits=2.7e8,
        )
        self.assertIs(result.mechanism, curves.NegativityMechanism.UNEXPLAINED)

    def test_missing_online_code_falls_through_rather_than_assuming_noise(self):
        losses = [6.9, 5.0, 4.0, 3.5, 3.3, 3.31, 3.29, 3.30, 3.31, 3.32]
        result = curves.diagnose(losses, model_bits=-1.0e3, vocab_size=1024)
        self.assertIs(result.mechanism, curves.NegativityMechanism.UNEXPLAINED)

    def test_omitting_train_tokens_skips_the_scope_check_entirely(self):
        """Backward compatibility: no train_tokens means no reconstruction."""
        losses = [6.0, 5.0, 4.0, 3.5, 3.4, 3.5, 4.0, 5.0, 6.5, 7.5]
        result = curves.diagnose(losses, model_bits=-1200.0, vocab_size=1024)
        self.assertIsNone(result.reconstructed_model_bits)
        self.assertIs(result.mechanism, curves.NegativityMechanism.LATE_DIVERGENCE)

    def test_consistent_train_tokens_does_not_trigger_a_false_scope_mismatch(self):
        """A model_bits that genuinely matches its own losses must not be flagged."""
        losses = [6.0, 5.0, 4.0, 3.5, 3.4, 3.5, 4.0, 5.0, 6.5, 7.5]
        train_tokens = 1_000_000
        consistent = curves.implied_model_bits(losses, train_tokens)
        result = curves.diagnose(
            losses, model_bits=consistent, vocab_size=1024,
            train_tokens=train_tokens,
        )
        self.assertIs(result.mechanism, curves.NegativityMechanism.LATE_DIVERGENCE)
        self.assertAlmostEqual(result.reconstructed_model_bits, consistent)

    def test_scope_mismatch_when_losses_do_not_reconstruct_model_bits(self):
        """The exact failure mode from review: a run's whole trajectory passed
        in to explain an earlier checkpoint's K(M).

        `losses` here is a full run that genuinely diverges late, so
        classify_shape correctly calls it LATE_RISE -- but the `model_bits`
        supplied belongs to a different, earlier checkpoint that never saw
        that divergence. Blaming LATE_DIVERGENCE for it would be wrong; the
        guard must refuse to name a shape-based mechanism at all.
        """
        whole_run_losses = [6.0, 5.0, 4.0, 3.5, 3.4, 3.5, 4.0, 5.0, 6.5, 7.5]
        earlier_checkpoint_model_bits = -1200.0
        result = curves.diagnose(
            whole_run_losses,
            model_bits=earlier_checkpoint_model_bits,
            vocab_size=1024,
            train_tokens=1_000_000,
        )
        self.assertIs(result.mechanism, curves.NegativityMechanism.SCOPE_MISMATCH)
        self.assertFalse(result.is_data_property)
        self.assertIsNotNone(result.reconstructed_model_bits)

    def test_scope_mismatch_takes_priority_over_shape_based_mechanisms(self):
        """A mismatch is reported even when the shape would say MINIMUM_NEAR_CEILING."""
        ceiling = log(1024)
        losses = [ceiling, ceiling - 0.001, ceiling, ceiling - 0.002, ceiling]
        result = curves.diagnose(
            losses, model_bits=-500.0, vocab_size=1024, train_tokens=1.0,
        )
        self.assertIs(result.mechanism, curves.NegativityMechanism.SCOPE_MISMATCH)

    def test_diagnose_forwards_plateau_tolerance_to_classify_shape(self):
        """classify_shape gained plateau_tolerance in the prior fix; diagnose
        must expose it too, or a caller has no way to override it."""
        losses = [6.5, 5.0, 4.0, 3.0, 3.5, 3.1, 3.0, 3.05, 3.0, 3.02]
        loose = curves.diagnose(
            losses, model_bits=1.0, vocab_size=1024, plateau_tolerance=0.9,
        )
        strict = curves.diagnose(
            losses, model_bits=1.0, vocab_size=1024, plateau_tolerance=1e-9,
        )
        self.assertIs(loose.shape, curves.LossCurveShape.PLATEAU)
        self.assertIs(strict.shape, curves.LossCurveShape.OTHER)

    def test_statistics_are_retained_for_audit(self):
        losses = [6.9, 5.0, 4.0, 3.0]
        result = curves.diagnose(losses, model_bits=1.0, vocab_size=1024)
        self.assertEqual(result.statistics.steps, 4)
        self.assertAlmostEqual(result.statistics.first_loss, 6.9)
        self.assertAlmostEqual(result.statistics.min_loss, 3.0)
        self.assertAlmostEqual(result.statistics.total_descent, 3.9)
        self.assertGreater(result.statistics.ceiling_gap, 0.0)


class EstimatorIdentityTests(unittest.TestCase):
    """Tie the algebra in curves.py back to what train_torch.py computes."""

    def test_model_bits_equal_tokens_times_trajectory_minus_tail_gap(self):
        losses = [6.0, 5.0, 4.5, 4.2, 4.1, 4.05, 4.0, 4.0, 3.99, 3.98]
        tokens = 31_250_000
        expected = (
            (sum(losses) / len(losses)) - losses[-1]
        ) * tokens / NATS_PER_BIT
        actual = curves.implied_model_bits(losses, tokens, tail_fraction=0.1)
        self.assertTrue(isclose(actual, expected, rel_tol=1e-12))

    def test_monotone_decreasing_curves_cannot_be_negative(self):
        losses = [6.0, 5.5, 5.0, 4.8, 4.7, 4.65, 4.6, 4.58, 4.57, 4.56]
        self.assertGreater(curves.implied_model_bits(losses, 1e6), 0.0)

    def test_a_late_rise_drives_the_estimate_below_zero(self):
        losses = [6.0, 5.0, 4.0, 3.5, 3.4, 3.5, 4.0, 5.0, 6.5, 7.5]
        self.assertLess(curves.implied_model_bits(losses, 1e6), 0.0)

if __name__ == "__main__":
    unittest.main()
