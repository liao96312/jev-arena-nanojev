import math
import unittest

from agents import MCTSAgent
from arena import ArenaConfig, ArenaEnv
from arena.entities import Action, Enemy
from search import Node, mcts_search


class MCTSTests(unittest.TestCase):
    def test_search_exports_visit_policy_and_is_deterministic(self):
        env = ArenaEnv(ArenaConfig(width=5, height=5, walls=0, enemies=0, gems=0,
                                   fires=0, medkits=0))
        env.player.position, env.gems = (2, 2), {(4, 2)}
        first = mcts_search(env, iterations=32, rollout_depth=2, seed=7)
        second = mcts_search(env, iterations=32, rollout_depth=2, seed=7)
        self.assertEqual(first, second)
        self.assertEqual(sum(first["visits"].values()), 32)
        self.assertAlmostEqual(math.fsum(first["action_probs"].values()), 1)
        self.assertGreater(first["tree_nodes"], 1)

    def test_mcts_prefers_instant_pit_kill(self):
        env = ArenaEnv(ArenaConfig(width=5, height=5, walls=0, enemies=0, gems=0,
                                   fires=0, pits=0, medkits=0))
        env.player.position = (1, 2)
        env.enemies, env.pits = [Enemy((2, 2))], {(3, 2)}
        env._plan_enemy_intents()
        self.assertEqual(MCTSAgent(iterations=96, rollout_depth=2).act(env), Action.SHOVE_E)

    def test_node_starts_unexpanded(self):
        env = ArenaEnv(ArenaConfig(width=3, height=3, walls=0, enemies=0, gems=0,
                                   fires=0, medkits=0))
        node = Node(env)
        self.assertEqual(set(node.untried), set(env.legal_actions()))
