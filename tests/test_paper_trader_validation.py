import unittest
from datetime import datetime, timedelta, timezone

from paper_trader import champion_validation_error


def valid_champion(created_at=None):
    return {
        "symbol": "ETHUSDT",
        "interval": "30m",
        "strategy": "mean_reversion",
        "params": {"fast": 8, "slow": 48, "momentum": 12, "breakout": 12,
                   "rsi_entry": 20, "rsi_exit": 50, "stop_pct": 0.06},
        "champion_version": "champion-test1234567890",
        "validation_metadata": {
            "created_at": (created_at or datetime.now(timezone.utc)).isoformat(),
            "status": "validated",
            "test_used_for_selection": False,
            "validation_metrics": {"total_return_pct": 1.0},
            "validation_stress_metrics": {"total_return_pct": 0.5},
            "test_metrics": {"total_return_pct": 0.2},
            "test_stress_metrics": {"total_return_pct": 0.1},
            "cost_convention": {"fee_basis": "executed notional"},
        },
    }


class PaperTraderChampionGateTests(unittest.TestCase):
    def test_accepts_fresh_versioned_validated_champion(self):
        self.assertIsNone(champion_validation_error(valid_champion()))

    def test_rejects_legacy_champion_without_metadata(self):
        self.assertIsNotNone(champion_validation_error({
            "symbol": "BTCUSDT", "interval": "5m", "strategy": "trend",
            "params": {"fast": 8, "slow": 30, "momentum": 6, "breakout": 12},
        }))

    def test_rejects_stale_champion(self):
        old = datetime.now(timezone.utc) - timedelta(hours=25)
        self.assertIn("stale", champion_validation_error(valid_champion(old)))

    def test_rejects_champion_if_final_test_was_used_for_selection(self):
        champion = valid_champion()
        champion["validation_metadata"]["test_used_for_selection"] = True
        self.assertIn("final-test isolation", champion_validation_error(champion))

    def test_rejects_incomplete_validation_metrics(self):
        champion = valid_champion()
        del champion["validation_metadata"]["validation_stress_metrics"]
        self.assertIn("validation_stress_metrics", champion_validation_error(champion))


if __name__ == "__main__":
    unittest.main()
