import math
import unittest

from epiaudio.conditional_epiplexity.prequential import (
    ConditionalPrequentialEstimator,
)


class ConditionalPrequentialEstimatorTest(unittest.TestCase):
    def test_matches_a_hand_calculated_prequential_code(self) -> None:
        estimator = ConditionalPrequentialEstimator()
        estimator.update(label_nll_sum_nats=5.0, label_count=2)
        estimator.update(label_nll_sum_nats=4.0, label_count=3)

        estimate = estimator.estimate(
            reference_label_nll_nats_per_label=1.2,
        )

        self.assertEqual(estimate.label_count, 5)
        self.assertEqual(estimate.online_label_code_nats, 9.0)
        self.assertEqual(estimate.online_label_nll_nats_per_label, 1.8)
        self.assertEqual(estimate.reference_label_code_nats, 6.0)
        self.assertEqual(estimate.conditional_epiplexity_nats, 3.0)
        self.assertAlmostEqual(
            estimate.online_label_nll_bits_per_label,
            1.8 / math.log(2.0),
        )
        self.assertAlmostEqual(
            estimate.reference_label_nll_bits_per_label,
            1.2 / math.log(2.0),
        )
        self.assertAlmostEqual(
            estimate.online_label_code_bits,
            9.0 / math.log(2.0),
        )
        self.assertAlmostEqual(
            estimate.reference_label_code_bits,
            6.0 / math.log(2.0),
        )
        self.assertAlmostEqual(
            estimate.conditional_epiplexity_bits,
            3.0 / math.log(2.0),
        )
        self.assertAlmostEqual(
            estimate.conditional_epiplexity_bits_per_label,
            3.0 / (5 * math.log(2.0)),
        )

    def test_does_not_clamp_a_negative_finite_sample_estimate(self) -> None:
        estimator = ConditionalPrequentialEstimator()
        estimator.update(label_nll_sum_nats=1.0, label_count=2)

        estimate = estimator.estimate(
            reference_label_nll_nats_per_label=1.0,
        )

        self.assertEqual(estimate.conditional_epiplexity_nats, -1.0)
        self.assertLess(estimate.conditional_epiplexity_bits, 0.0)

    def test_resume_state_matches_uninterrupted_accounting(self) -> None:
        uninterrupted = ConditionalPrequentialEstimator()
        uninterrupted.update(label_nll_sum_nats=3.25, label_count=2)
        uninterrupted.update(label_nll_sum_nats=4.75, label_count=3)

        partial = ConditionalPrequentialEstimator()
        partial.update(label_nll_sum_nats=3.25, label_count=2)
        restored = ConditionalPrequentialEstimator.from_state_dict(partial.state_dict())
        restored.update(label_nll_sum_nats=4.75, label_count=3)

        self.assertEqual(restored.state_dict(), uninterrupted.state_dict())
        self.assertEqual(
            restored.estimate(0.8),
            uninterrupted.estimate(0.8),
        )

    def test_requires_valid_observations_and_reference_loss(self) -> None:
        estimator = ConditionalPrequentialEstimator()
        with self.assertRaisesRegex(ValueError, "At least one label"):
            estimator.estimate(1.0)

        invalid_updates = (
            (-1.0, 1, ValueError),
            (math.inf, 1, ValueError),
            (math.nan, 1, ValueError),
            (1.0, 0, ValueError),
            (1.0, 1.5, TypeError),
            (True, 1, TypeError),
        )
        for nll_sum, count, error in invalid_updates:
            with self.subTest(nll_sum=nll_sum, count=count):
                with self.assertRaises(error):
                    estimator.update(nll_sum, count)  # type: ignore[arg-type]

        estimator.update(1.0, 1)
        for reference in (-1.0, math.inf, math.nan, True):
            with self.subTest(reference=reference):
                with self.assertRaises((TypeError, ValueError)):
                    estimator.estimate(reference)  # type: ignore[arg-type]

    def test_rejects_malformed_or_unsupported_state(self) -> None:
        valid_state: dict[str, object] = {
            "version": 1,
            "online_label_nll_nats": 2.0,
            "label_count": 3,
        }
        malformed_states = (
            {"version": 1, "label_count": 3},
            {**valid_state, "extra": 1},
            {**valid_state, "version": 2},
            {**valid_state, "online_label_nll_nats": -1.0},
            {**valid_state, "label_count": -1},
        )
        for state in malformed_states:
            with self.subTest(state=state):
                with self.assertRaises((TypeError, ValueError)):
                    ConditionalPrequentialEstimator.from_state_dict(state)


if __name__ == "__main__":
    unittest.main()
