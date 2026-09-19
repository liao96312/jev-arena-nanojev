import tempfile
import unittest
from pathlib import Path

from agents import RuleAgent
from arena import ArenaConfig, ArenaEnv
from arena.replay import ReplayLogger
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


if __name__ == "__main__":
    unittest.main()
