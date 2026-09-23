from __future__ import annotations

import unittest
from math import isclose, log2

from epiaudio.analysis import units


class TokenDurationTests(unittest.TestCase):
    def test_token_rates_match_the_five_second_specs(self):
        self.assertEqual(units.tokens_per_second("encodec"), 75.0)
        self.assertEqual(units.tokens_per_second("dac"), 600.0)
        self.assertEqual(units.tokens_per_second("xcodec"), 400.0)
        self.assertAlmostEqual(units.tokens_per_second("sqcodec"), 355.6)

    def test_fixed_token_budget_buys_very_different_audio(self):
        """The confound the whole package exists to remove."""
        budget = 31_250_000
        hours = {
            tokenizer: units.tokens_to_seconds(budget, tokenizer) / 3600.0
            for tokenizer in ("encodec", "sqcodec", "xcodec", "dac")
        }
        self.assertAlmostEqual(hours["encodec"], 115.74, places=2)
        self.assertAlmostEqual(hours["sqcodec"], 24.41, places=2)
        self.assertAlmostEqual(hours["xcodec"], 21.70, places=2)
        self.assertAlmostEqual(hours["dac"], 14.47, places=2)
        self.assertAlmostEqual(hours["encodec"] / hours["dac"], 8.0, places=6)

    def test_seconds_and_tokens_round_trip(self):
        for tokenizer in ("encodec", "dac", "xcodec"):
            tokens = units.seconds_to_tokens(3600.0, tokenizer)
            self.assertAlmostEqual(
                units.tokens_to_seconds(tokens, tokenizer), 3600.0, places=6
            )

    def test_sqcodec_rounds_to_whole_tokens(self):
        tokens = units.seconds_to_tokens(3600.0, "sqcodec")
        self.assertEqual(tokens, round(3600.0 * 355.6))
        self.assertLess(
            abs(units.tokens_to_seconds(tokens, "sqcodec") - 3600.0) / 3600.0, 1e-6
        )

    def test_negative_inputs_rejected(self):
        with self.assertRaises(ValueError):
            units.tokens_to_seconds(-1.0, "encodec")
        with self.assertRaises(ValueError):
            units.seconds_to_tokens(-1.0, "encodec")

    def test_unknown_tokenizer_names_the_known_ones(self):
        with self.assertRaises(KeyError) as ctx:
            units.tokens_per_second("mp3")
        self.assertIn("encodec", str(ctx.exception))


class BitrateTests(unittest.TestCase):
    def test_max_bits_per_token(self):
        self.assertAlmostEqual(units.max_bits_per_token("encodec"), 10.0)
        self.assertAlmostEqual(units.max_bits_per_token("sqcodec"), log2(117_649))

    def test_uniform_code_bitrates_span_eight_times(self):
        rates = {
            tokenizer: units.nominal_bitrate_bits_per_second(tokenizer)
            for tokenizer in ("encodec", "sqcodec", "xcodec", "dac")
        }
        self.assertAlmostEqual(rates["encodec"], 750.0)
        self.assertAlmostEqual(rates["dac"], 6000.0)
        self.assertAlmostEqual(rates["dac"] / rates["encodec"], 8.0)

    def test_continuous_extractor_has_no_code_length(self):
        with self.assertRaises(ValueError):
            units.max_bits_per_token("whisper")


class ModelSizeTests(unittest.TestCase):
    def test_d_model_matches_train_py_formula(self):
        """train.py:143 -- round(sqrt(P*1e6/N/12)/64)*64."""
        for params_millions, depth in ((160, 12), (40, 6), (5, 3)):
            expected = round(
                ((params_millions * 1e6 / depth / 12) ** 0.5) / 64
            ) * 64
            self.assertEqual(units.d_model_for(params_millions, depth), expected)

    def test_grid_points_below_the_width_floor_are_rejected(self):
        with self.assertRaises(ValueError):
            units.d_model_for(0.01, 24)

    def test_non_embedding_params_approximate_the_budget(self):
        """Rounding d_model to a multiple of 64 costs a few percent of the budget.

        P=160 at N=12 implies a width of 1054, which rounds down to 1024 and so
        realises 151.0M block parameters rather than 160M.
        """
        d_model = units.d_model_for(160, 12)
        self.assertEqual(d_model, 1024)
        actual = units.non_embedding_params(12, d_model)
        self.assertTrue(isclose(actual, 160e6, rel_tol=0.10))
        self.assertAlmostEqual(actual / 1e6, 151.0, places=1)

    def test_embedding_fraction_explains_the_cross_tokenizer_gap(self):
        """The readout is untied, so a large vocabulary is charged twice.

        Counts verified against an instantiated `TransformerDecoder`:
        EnCodec 150,994,944 block + 1,048,576 embed + 384,000 pos + 1,048,576
        readout = 153,476,096. SQ-Codec 150,994,944 + 120,472,576 + 1,820,672 +
        120,472,576 = 393,760,768.
        """
        encodec = units.embedding_parameter_fraction(160, 12, 1024, 375)
        sqcodec = units.embedding_parameter_fraction(160, 12, 117_649, 1778)
        self.assertAlmostEqual(encodec, 0.0162, places=3)
        self.assertAlmostEqual(sqcodec, 0.6166, places=3)

        encodec_total = units.analytic_total_params(160, 12, 1024, 375)
        sqcodec_total = units.analytic_total_params(160, 12, 117_649, 1778)
        self.assertAlmostEqual(sqcodec_total / encodec_total, 2.566, places=3)

    def test_flop_bearing_params_exclude_the_gather_tables(self):
        """6ND overcounts compute by 45% for SQ-Codec, 0.9% for EnCodec."""
        for vocab, expected in ((1024, 1.009), (117_649, 1.450)):
            total = units.analytic_total_params(160, 12, vocab, 375)
            flops = units.flop_bearing_params(160, 12, vocab)
            self.assertAlmostEqual(total / flops, expected, places=2)
        self.assertEqual(
            units.flop_bearing_params(160, 12, 1024),
            units.non_embedding_params(12, 1024) + 1024 * 1024,
        )

    def test_untied_readout_adds_a_second_vocab_matrix(self):
        tied = units.embedding_params(1024, 375, 512, tie_readout=True)
        untied = units.embedding_params(1024, 375, 512, tie_readout=False)
        self.assertEqual(untied - tied, 1024 * 512)


class RepeatFactorTests(unittest.TestCase):
    def test_mls_headroom_matches_the_recorded_value(self):
        """The team sheet records MLS clearing T with 5.6x headroom."""
        factor = units.repeat_factor(469_942, "encodec", 31_250_000)
        self.assertAlmostEqual(1.0 / factor, 5.64, places=2)
        self.assertLess(factor, 1.0)

    def test_small_corpora_replay_tokens(self):
        self.assertAlmostEqual(
            units.repeat_factor(6_394, "encodec", 31_250_000), 13.03, places=2
        )
        self.assertAlmostEqual(
            units.repeat_factor(9_988, "encodec", 31_250_000), 8.34, places=2
        )

    def test_repeat_factor_depends_on_the_tokenizer(self):
        """A corpus can repeat 13x under EnCodec and under 2x under DAC."""
        encodec = units.repeat_factor(6_394, "encodec", 31_250_000)
        dac = units.repeat_factor(6_394, "dac", 31_250_000)
        self.assertGreater(encodec, 13.0)
        self.assertLess(dac, 2.0)

    def test_clips_required(self):
        self.assertEqual(units.clips_required(31_250_000, "encodec"), 83_334)
        self.assertEqual(units.clips_required(31_250_000, "dac"), 10_417)


if __name__ == "__main__":
    unittest.main()
