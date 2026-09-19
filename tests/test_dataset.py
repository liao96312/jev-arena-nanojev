import tempfile
import unittest
from pathlib import Path

from arena.dataset import split_for_seed, validate_dataset
from arena import ArenaConfig, ArenaEnv
from scripts.generate_dataset import generate, rollout_probabilities


class DatasetTests(unittest.TestCase):
    def test_split_is_group_stable(self):
        self.assertEqual(split_for_seed(7), split_for_seed(7))
        self.assertEqual(split_for_seed(19), "ood")

    def test_generated_dataset_validates(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.jsonl"
            summary = generate(path, records=40, per_seed=2)
            self.assertEqual(summary["records"], 40)
            self.assertEqual(summary, validate_dataset(path))
            self.assertEqual(set(summary["splits"]), {"train", "dev", "calibration", "test", "ood"})

    def test_rollout_soft_targets_penalize_fire(self):
        env = ArenaEnv(ArenaConfig(width=5, height=5, walls=0, enemies=0, gems=0, fires=0, medkits=0))
        env.player.position = (2, 2)
        env.fires = {(2, 3)}
        candidates = [action.value for action in env.legal_actions()]
        probabilities, returns = rollout_probabilities(env, candidates, horizon=1, temperature=1)
        self.assertAlmostEqual(sum(probabilities.values()), 1)
        self.assertLess(returns["move_s"], returns["move_n"])
        self.assertLess(probabilities["move_s"], probabilities["move_n"])

    def test_rollout_targets_reward_gem_progress(self):
        env = ArenaEnv(ArenaConfig(width=5, height=5, walls=0, enemies=0, gems=0, fires=0, medkits=0))
        env.player.position = (2, 2)
        env.gems = {(4, 2)}
        probabilities, returns = rollout_probabilities(env, ["move_w", "move_e"], horizon=1, temperature=1)
        self.assertGreater(returns["move_e"], returns["move_w"])
        self.assertGreater(probabilities["move_e"], probabilities["move_w"])

    def test_rollout_dataset_is_not_one_hot(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "soft.jsonl"
            generate(path, records=20, per_seed=1, targets="rollout")
            import json
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            self.assertTrue(any(0 < value < 1 for row in rows for value in row["gold_probs"]["action"].values()))

    def test_v2_dataset_contains_tactical_features(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "v2.jsonl"
            generate(path, records=100, per_seed=10, targets="rollout", rollout_horizon=1, v2=True)
            import json
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            self.assertTrue(all(row["family_id"] == "arena_v2" for row in rows))
            self.assertTrue(any("W bow=" in row["state"] for row in rows))
            self.assertTrue(any("barrel=" in row["state"] for row in rows))
            self.assertTrue(any(any(action.startswith("shoot_") for action in row["questions"]["action"]["criteria"])
                                for row in rows))


if __name__ == "__main__":
    unittest.main()
