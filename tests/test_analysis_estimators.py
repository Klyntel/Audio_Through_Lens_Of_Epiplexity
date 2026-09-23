from __future__ import annotations

import unittest
from math import log

from epiaudio.analysis import estimators

NATS_PER_BIT = log(2.0)


class UnitConversionTests(unittest.TestCase):
    def test_code_length_round_trips_through_loss(self):
        bits = estimators.code_bits_from_loss(6.0, 1e6)
        self.assertAlmostEqual(bits, 6.0 * 1e6 / NATS_PER_BIT)
        self.assertAlmostEqual(estimators.loss_from_code_bits(bits, 1e6), 6.0)

    def test_negative_loss_rejected(self):
        with self.assertRaises(ValueError):
            estimators.code_bits_from_loss(-1.0, 10.0)


class RestandardizationTests(unittest.TestCase):
    def test_data_term_is_linear_in_the_inference_set_size(self):
        stored = estimators.code_bits_from_loss(6.0, 1_000_000)
        rescaled = estimators.restandardize_data_bits(stored, 1_000_000, 4_000_000)
        self.assertAlmostEqual(rescaled, stored * 4.0)

    def test_restandardizing_preserves_the_implied_loss(self):
        stored = estimators.code_bits_from_loss(5.25, 800_000)
        rescaled = estimators.restandardize_data_bits(stored, 800_000, 6_480_000)
        self.assertAlmostEqual(
            estimators.loss_from_code_bits(rescaled, 6_480_000), 5.25, places=9
        )

    def test_non_positive_sizes_rejected(self):
        with self.assertRaises(ValueError):
            estimators.restandardize_data_bits(1.0, 0, 10)


class RestandardizationByClipsTests(unittest.TestCase):
    def test_matches_the_token_based_result_for_the_same_tokenizer(self):
        # 800,000 clips at DAC's 3000 tokens/clip is 2.4B tokens either way.
        stored = estimators.code_bits_from_loss(6.0, 2_400_000_000)
        by_tokens = estimators.restandardize_data_bits(
            stored, 2_400_000_000, 4_800_000_000
        )
        by_clips = estimators.restandardize_data_bits_by_clips(
            stored, stored_clips=800_000, target_clips=1_600_000, tokenizer="dac"
        )
        self.assertAlmostEqual(by_tokens, by_clips)

    def test_clip_count_is_comparable_across_tokenizers(self):
        # Restandardizing to "the same 10,000 clips" should not depend on
        # which tokenizer produced the stored value, unlike a raw token target.
        stored = estimators.code_bits_from_loss(6.0, 1_000_000)
        dac = estimators.restandardize_data_bits_by_clips(
            stored, stored_clips=1000, target_clips=10_000, tokenizer="dac"
        )
        encodec = estimators.restandardize_data_bits_by_clips(
            stored, stored_clips=1000, target_clips=10_000, tokenizer="encodec"
        )
        # Both restandardize by the same 10x clip ratio, so both scale the
        # stored value by exactly 10x, regardless of the tokenizer's rate.
        self.assertAlmostEqual(dac, stored * 10.0)
        self.assertAlmostEqual(encodec, stored * 10.0)

    def test_non_positive_clip_counts_rejected(self):
        with self.assertRaises(ValueError):
            estimators.restandardize_data_bits_by_clips(
                1.0, stored_clips=0, target_clips=10, tokenizer="dac"
            )


class EvalLossVariantTests(unittest.TestCase):
    def test_variant_subtracts_the_held_out_code_from_the_online_code(self):
        online = estimators.code_bits_from_loss(6.0, 31_250_000)
        model_bits = estimators.epiplexity_from_eval_loss(online, 5.5, 31_250_000)
        self.assertAlmostEqual(
            model_bits,
            online - estimators.code_bits_from_loss(5.5, 31_250_000),
        )
        self.assertGreater(model_bits, 0.0)

    def test_variant_goes_negative_when_held_out_loss_exceeds_the_online_mean(self):
        online = estimators.code_bits_from_loss(6.0, 1e6)
        self.assertLess(estimators.epiplexity_from_eval_loss(online, 6.5, 1e6), 0.0)


class PrecisionTests(unittest.TestCase):
    def test_required_precision_is_the_signal_to_baseline_ratio(self):
        self.assertAlmostEqual(
            estimators.relative_precision_required(1e6, 1e8), 0.01
        )

    def test_low_structure_data_demands_sub_percent_accuracy(self):
        """The core reason prequential estimation fails in this regime."""
        online = estimators.code_bits_from_loss(6.0, 31_250_000)
        model_bits = 0.01 * online
        required = estimators.relative_precision_required(model_bits, online)
        assert required is not None
        self.assertAlmostEqual(required, 0.01)
        self.assertLess(required / 10.0, 0.002)

    def test_precision_is_undefined_without_an_online_code(self):
        self.assertIsNone(estimators.relative_precision_required(1.0, 0.0))


class StructuralFractionTests(unittest.TestCase):
    def test_fraction_of_total_information(self):
        self.assertAlmostEqual(estimators.structural_fraction(1.0, 99.0), 0.01)

    def test_no_reading_for_negative_or_empty_totals(self):
        self.assertIsNone(estimators.structural_fraction(-1.0, 99.0))
        self.assertIsNone(estimators.structural_fraction(0.0, 0.0))

if __name__ == "__main__":
    unittest.main()
