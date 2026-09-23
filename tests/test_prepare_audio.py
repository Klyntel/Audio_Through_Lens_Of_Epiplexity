from __future__ import annotations

import unittest

import torch

from epiaudio.dataset.prepare_audio import _build_clip_plan, load_clip


class _FakeSplit:
    """Minimal stand-in for an HF Dataset split: len(), column access by name,
    and row access by index -- everything _build_clip_plan needs."""

    def __init__(self, rows: list[dict]):
        self._rows = rows
        self.column_names = list(rows[0].keys()) if rows else []

    def __len__(self):
        return len(self._rows)

    def __getitem__(self, key):
        if isinstance(key, str):
            return [row[key] for row in self._rows]
        return self._rows[key]


class BuildClipPlanTests(unittest.TestCase):
    def test_rows_without_clip_duration_get_exactly_one_window_each(self):
        """FSD50K/LOCATA/STARSS23-style rows carry no clip_duration column at all,
        so the plan must fall back to today's one-window-per-row behavior untouched."""
        split = _FakeSplit([{"audio": "a"}, {"audio": "b"}, {"audio": "c"}])
        self.assertEqual(_build_clip_plan(split), [(0, 0), (1, 0), (2, 0)])

    def test_short_rows_get_exactly_one_window(self):
        """DEMAND-style rows at or under seconds_fix stay at one window, matching
        the existing padding path: the no-op-for-short-rows guarantee."""
        split = _FakeSplit([{"clip_duration": 3.0}, {"clip_duration": 5.0}])
        self.assertEqual(_build_clip_plan(split, seconds_fix=5), [(0, 0), (1, 0)])

    def test_long_row_gets_multiple_sequential_windows(self):
        """A 12-second row yields floor(12/5) = 2 windows, not one sample from
        somewhere inside it."""
        split = _FakeSplit([{"clip_duration": 12.0}])
        self.assertEqual(_build_clip_plan(split, seconds_fix=5), [(0, 0), (0, 1)])

    def test_mixed_short_and_long_rows(self):
        split = _FakeSplit([
            {"clip_duration": 2.0},   # 1 window (padded)
            {"clip_duration": 17.0},  # floor(17/5) = 3 windows
            {"clip_duration": 5.0},   # exactly one window's worth -> 1 window
        ])
        self.assertEqual(
            _build_clip_plan(split, seconds_fix=5),
            [(0, 0), (1, 0), (1, 1), (1, 2), (2, 0)],
        )


class LoadClipTests(unittest.TestCase):
    @staticmethod
    def _decode(row):
        return row["tensor"], row["sr"]

    def test_short_row_is_padded_regardless_of_window_index(self):
        """A row shorter than seconds_fix always hits the padding path;
        _build_clip_plan never emits more than one window for such a row, so
        window_index is always 0 in practice, but the padding itself must not
        depend on which window_index it's handed."""
        sr = 10
        waveform = torch.ones(1, 3 * sr)  # 3 seconds, shorter than seconds_fix=5
        split = [{"tensor": waveform, "sr": sr}]

        y, out_sr = load_clip((split, (0, 0), self._decode), seconds_fix=5)

        self.assertEqual(out_sr, sr)
        self.assertEqual(y.shape, (1, 5 * sr))
        self.assertTrue(torch.equal(y[:, : 3 * sr], waveform))
        self.assertTrue(torch.equal(y[:, 3 * sr :], torch.zeros(1, 2 * sr)))

    def test_long_row_windows_are_sequential_and_non_overlapping(self):
        """A 15-second row split at seconds_fix=5 gives three distinct,
        back-to-back 5-second windows rather than one random sample."""
        sr = 10
        # Each second's samples are filled with that second's index, so a window's
        # content directly reveals which seconds it actually covers.
        waveform = torch.cat(
            [torch.full((1, sr), float(second)) for second in range(15)], dim=1
        )
        split = [{"tensor": waveform, "sr": sr}]

        window_0, _ = load_clip((split, (0, 0), self._decode), seconds_fix=5)
        window_1, _ = load_clip((split, (0, 1), self._decode), seconds_fix=5)
        window_2, _ = load_clip((split, (0, 2), self._decode), seconds_fix=5)

        self.assertTrue(torch.equal(window_0, waveform[:, 0:50]))
        self.assertTrue(torch.equal(window_1, waveform[:, 50:100]))
        self.assertTrue(torch.equal(window_2, waveform[:, 100:150]))
        self.assertFalse(torch.equal(window_0, window_1))
        self.assertFalse(torch.equal(window_1, window_2))


if __name__ == "__main__":
    unittest.main()
