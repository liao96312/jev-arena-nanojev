import unittest

from arena import ArenaConfig, ArenaEnv
from arena.entities import Enemy
from scripts.analyze_game_complexity import tactical_metrics


class ComplexityTests(unittest.TestCase):
    def test_detects_immediate_forced_death(self):
        env = ArenaEnv(ArenaConfig(width=1, height=2, walls=0, enemies=0, gems=0, fires=0,
                                   medkits=0, enemy_damage=5, charger_ratio=0, bomber_ratio=0))
        env.player.position = (0, 0)
        env.player.hp = 5
        env.player.cooldowns["emp"] = 99
        env.enemies = [Enemy((0, 1))]
        env._plan_enemy_intents()
        immediate, two_step, gap, entropy = tactical_metrics(env)
        self.assertTrue(immediate)
        self.assertTrue(two_step)
        self.assertGreaterEqual(gap, 0)
        self.assertGreaterEqual(entropy, 0)


if __name__ == "__main__":
    unittest.main()
