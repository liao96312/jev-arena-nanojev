import unittest

from agents import RuleAgent
from arena import ArenaEnv, campaign_config
from arena.candidates import build_candidates
from arena.entities import Action
from arena.observation import encode_state


class PrismBossTests(unittest.TestCase):
    def test_fixed_room_and_model_context(self):
        env = ArenaEnv(campaign_config(10))
        self.assertEqual(env.player.position, (10, 15))
        self.assertEqual(env.reflectors, {(9, 12), (10, 12), (11, 12)})
        self.assertIn((6, 10), env.walls)
        self.assertNotIn((4, 4), env.walls)
        self.assertIn((3, 1), env.walls)
        self.assertIn((1, 8), env.walls)
        self.assertNotIn((2, 8), env.walls)
        self.assertEqual(env.pistol_pickups, {(10, 14)})
        self.assertEqual(env.energy_cells, {(7, 15), (13, 15)})
        self.assertEqual(env.medkits, {(7, 14), (13, 14)})
        self.assertTrue((env.pistol_pickups | env.energy_cells | env.medkits) <= env._reachable_cells())
        self.assertIn("Boss prism", encode_state(env))
        self.assertIn("prism_warden", build_candidates(env)[Action.SHOOT_PISTOL_N.value])
        self.assertEqual(env.player.loadout.energy, 12)

    def test_warning_matches_damage_and_mirror_blocks(self):
        env = ArenaEnv(campaign_config(10))
        env.round = 3
        env.boss.target = env.player.position
        self.assertEqual(env.boss_ray()[-1], (10, 12))
        env.step(Action.WAIT)
        reflected = env.step(Action.WAIT)
        self.assertIn("boss_reflect:1", reflected.events)
        self.assertEqual(env.player.hp, 100)
        env.boss.position = (9, 8)
        env.boss.target = (8, 15)
        env.player.position = (8, 14)
        self.assertIn(env.player.position, env.boss_ray())
        env.step(Action.WAIT)
        hit = env.step(Action.WAIT)
        self.assertIn("damage:boss_prism:14", hit.events)

    def test_rule_agent_can_finish_boss_without_damage(self):
        env, agent = ArenaEnv(campaign_config(10)), RuleAgent()
        events = []
        for _ in range(100):
            if env.done:
                break
            events.extend(env.step(agent.act(env)).events)
        self.assertTrue(env.done)
        self.assertIsNone(env.boss)
        self.assertEqual(env.player.hp, 100)
        self.assertIn("boss_shield_break", events)
        self.assertTrue(any(event.startswith("boss_lunge_aim:") for event in events))
        self.assertIn("boss_lunge", events)
        self.assertIn("boss_defeated", events)

    def test_lunge_warning_matches_damage_zone_and_can_be_dodged(self):
        env = ArenaEnv(campaign_config(10))
        env.round = 4
        env.boss.reflections = 2
        env.boss.lunge_target = (10, 14)
        self.assertIn(("prism_warden/lunge", 24), env.imminent_threats())
        env.step(Action.MOVE_W)
        result = env.step(Action.WAIT)
        self.assertIn("boss_lunge", result.events)
        self.assertEqual(env.player.hp, 100)
        self.assertEqual(env.boss.position, (10, 14))

        exposed = ArenaEnv(campaign_config(10))
        exposed.round = 4
        exposed.boss.lunge_target = (10, 14)
        exposed.step(Action.WAIT)
        hit = exposed.step(Action.WAIT)
        self.assertIn("damage:boss_lunge:24", hit.events)
        self.assertEqual(exposed.player.hp, 76)


if __name__ == "__main__":
    unittest.main()
