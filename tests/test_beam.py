import math
import unittest

from agents import BeamSearchAgent
from arena import ArenaConfig, ArenaEnv
from arena.entities import Action, Enemy
from search import beam_search, evaluate_state


class BeamSearchTests(unittest.TestCase):
    def test_search_returns_values_visits_and_soft_distribution(self):
        env = ArenaEnv(ArenaConfig(width=5, height=5, walls=0, enemies=0, gems=0,
                                   fires=0, medkits=0, action_points=2))
        env.player.position, env.gems = (2, 2), {(4, 2)}
        result = beam_search(env, depth=3, width=8)
        offered = {action.value for action in env.legal_actions()}
        self.assertEqual(set(result["action_probs"]), offered)
        self.assertEqual(set(result["action_values"]), offered)
        self.assertAlmostEqual(math.fsum(result["action_probs"].values()), 1)
        self.assertGreater(result["expanded"], len(offered))

    def test_beam_uses_pit_environment_kill(self):
        env = ArenaEnv(ArenaConfig(width=5, height=5, walls=0, enemies=0, gems=0,
                                   fires=0, pits=0, medkits=0))
        env.player.position = (1, 2)
        env.enemies, env.pits = [Enemy((2, 2))], {(3, 2)}
        env._plan_enemy_intents()
        self.assertEqual(BeamSearchAgent(depth=2, width=8).act(env), Action.SHOVE_E)

    def test_terminal_evaluator_orders_win_over_death(self):
        env = ArenaEnv(ArenaConfig(width=3, height=3, walls=0, enemies=0, gems=0,
                                   fires=0, medkits=0, finish_on_all_gems=True))
        won, dead = env.clone(), env.clone()
        won.done, won.gems = True, set()
        dead.player.hp = 0
        self.assertGreater(evaluate_state(won), evaluate_state(dead))
