import unittest

from scripts.benchmark import run


class BenchmarkTests(unittest.TestCase):
    def test_v2_metrics_are_recorded(self):
        row = run("rule", episodes=1, max_ticks=10, campaign_level=3)[0]
        for key in ("win", "hp_remaining", "environment_kills", "mean_branch_factor",
                    "deadlock_rate", "unique_action_rate", "skill_efficiency"):
            self.assertIn(key, row)
        self.assertGreaterEqual(row["mean_branch_factor"], 1)
        self.assertGreaterEqual(row["unique_action_rate"], 0)
        self.assertLessEqual(row["unique_action_rate"], 1)
