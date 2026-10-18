import unittest

import numpy as np
import pandas as pd

from research_engine import desired_signal


class ShortSignalTests(unittest.TestCase):
    def setUp(self):
        index = pd.date_range("2026-01-01", periods=240, freq="5min", tz="UTC")
        close = np.linspace(150.0, 80.0, len(index))
        self.frame = pd.DataFrame({
            "open": close,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": np.full(len(index), 10.0),
        }, index=index)
        self.params = {
            "fast": 8, "slow": 30, "momentum": 6, "breakout": 12,
            "entry_momentum": 0.0, "exit_momentum": -0.005,
            "rsi_entry": 20, "rsi_exit": 50, "stop_pct": 0.06,
        }

    def test_short_signal_only_when_enabled(self):
        long_flat = desired_signal(self.frame, "trend", self.params, allow_short=False)
        long_short = desired_signal(self.frame, "trend", self.params, allow_short=True)
        self.assertEqual(float(long_flat.iloc[-1]), 0.0)
        self.assertEqual(float(long_short.iloc[-1]), -1.0)

    def test_short_signal_is_delayed_one_bar(self):
        signals = desired_signal(self.frame, "trend", self.params, allow_short=True)
        self.assertIn(float(signals.iloc[-1]), (-1.0, 0.0))


if __name__ == "__main__":
    unittest.main()
