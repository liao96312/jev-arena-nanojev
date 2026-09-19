import unittest

from agents import RandomAgent, RuleAgent
from arena import ArenaConfig, ArenaEnv, campaign_config
from arena.candidates import build_candidates
from arena.entities import Action, Enemy


class ArenaTests(unittest.TestCase):
    def test_seed_is_deterministic(self):
        first, second = ArenaEnv(), ArenaEnv()
        self.assertEqual(first.reset(42), second.reset(42))
        agents = RandomAgent(7), RandomAgent(7)
        for _ in range(20):
            self.assertEqual(first.step(agents[0].act(first)), second.step(agents[1].act(second)))
            self.assertEqual(first.observation(), second.observation())

    def test_candidates_filter_wall_but_keep_fire(self):
        env = ArenaEnv(ArenaConfig(width=5, height=5, walls=0, enemies=0, gems=0, fires=0, medkits=0))
        env.player.position = (2, 2)
        env.walls = {(2, 1)}
        env.fires = {(2, 3)}
        candidates = build_candidates(env)
        self.assertNotIn("move_n", candidates)
        self.assertIn("move_s", candidates)
        self.assertIn("fire", candidates["move_s"])
        self.assertIn("wait", candidates)

    def test_attack_only_when_adjacent(self):
        env = ArenaEnv(ArenaConfig(width=5, height=5, walls=0, enemies=0, gems=0, fires=0, medkits=0))
        env.player.position = (2, 2)
        env.enemies = [Enemy((3, 2))]
        self.assertIn(Action.ATTACK_E, env.legal_actions())
        self.assertNotIn(Action.MOVE_E, env.legal_actions())
        result = env.step(Action.ATTACK_E)
        self.assertIn("attack", result.events)

    def test_clone_is_independent(self):
        env = ArenaEnv()
        clone = env.clone()
        clone.step(clone.legal_actions()[0])
        self.assertNotEqual(env.observation(), clone.observation())

    def test_candidate_marks_backtracking(self):
        env = ArenaEnv(ArenaConfig(width=5, height=5, walls=0, enemies=0, gems=0, fires=0, medkits=0))
        env.player.position = (2, 2)
        env.step(Action.MOVE_E)
        from arena.candidates import build_candidates
        self.assertIn("previous cell", build_candidates(env)["move_w"])
        self.assertEqual(env.last_action, "move_e")

    def test_candidate_describes_gem_progress(self):
        env = ArenaEnv(ArenaConfig(width=5, height=5, walls=0, enemies=0, gems=0, fires=0, medkits=0))
        env.player.position = (2, 2)
        env.gems = {(4, 2)}
        from arena.candidates import build_candidates
        self.assertIn("distance 2 to 1", build_candidates(env)["move_e"])

    def test_agents_run_to_terminal(self):
        for agent in (RandomAgent(1), RuleAgent()):
            env = ArenaEnv(ArenaConfig(max_ticks=30))
            while not env.done:
                env.step(agent.act(env))
            self.assertLessEqual(env.tick, 30)

    def test_campaign_gets_harder(self):
        first, fifth = campaign_config(1), campaign_config(5)
        self.assertLess(first.enemies, fifth.enemies)
        self.assertLess(first.fires, fifth.fires)
        self.assertGreater(first.medkits, fifth.medkits)

    def test_collecting_last_campaign_gem_completes_level(self):
        env = ArenaEnv(ArenaConfig(width=3, height=3, walls=0, enemies=0, gems=1, fires=0,
                                   medkits=0, finish_on_all_gems=True))
        env.player.position, env.gems = (1, 1), {(2, 1)}
        result = env.step(Action.MOVE_E)
        self.assertTrue(result.done)
        self.assertIn("level_complete", result.events)

    def test_slow_enemy_damages_on_contact(self):
        env = ArenaEnv(ArenaConfig(width=5, height=5, walls=0, enemies=0, gems=0, fires=0,
                                   medkits=0, enemy_move_interval=2))
        env.player.position = (2, 2)
        env.enemies = [Enemy((2, 4))]
        env.step(Action.WAIT)
        self.assertEqual(env.enemies[0].position, (2, 4))
        result = env.step(Action.WAIT)
        self.assertEqual(env.enemies[0].position, (2, 3))
        self.assertEqual(env.player.hp, 95)
        self.assertIn("damage:enemy:5", result.events)


if __name__ == "__main__":
    unittest.main()
