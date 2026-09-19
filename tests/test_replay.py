import tempfile
import unittest
from pathlib import Path

from agents import RuleAgent
from arena import ArenaConfig, ArenaEnv
from arena.replay import ReplayLogger
from arena.entities import PlayerLoadout
from scripts.replay import verify


class ReplayTests(unittest.TestCase):
    def test_replay_reproduces_transitions(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "episode.jsonl"
            env = ArenaEnv(ArenaConfig(max_ticks=10))
            agent, logger = RuleAgent(), ReplayLogger(path)
            while not env.done:
                state = env.observation()
                candidates = [action.value for action in env.legal_actions()]
                action = agent.act(env)
                result = env.step(action)
                logger.log(state, candidates, env, action.value, result)
            summary = verify(path)
            self.assertTrue(summary["verified"])
            self.assertEqual(summary["steps"], env.tick)

    def test_replay_reproduces_ranged_weapon_state(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weapon.jsonl"
            config = ArenaConfig(width=7, height=1, max_ticks=1, walls=0, enemies=1,
                                 gems=0, fires=0, medkits=0)
            env = ArenaEnv(config, PlayerLoadout(bow=True, arrows=2))
            for seed in range(20):
                env.reset(seed)
                shots = [action for action in env.legal_actions() if action.value.startswith("shoot_bow_")]
                if shots:
                    break
            self.assertTrue(shots)
            state = env.observation()
            candidates = [action.value for action in env.legal_actions()]
            result = env.step(shots[0])
            ReplayLogger(path).log(state, candidates, env, shots[0].value, result)
            self.assertTrue(verify(path)["verified"])


if __name__ == "__main__":
    unittest.main()
