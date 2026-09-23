import unittest
from pathlib import Path
from types import SimpleNamespace

import pygame

from arena import ArenaConfig, ArenaEnv
from scripts.play_gui import keyboard_command, parse_level_selection, restart_level, switch_agent


class GuiControlTests(unittest.TestCase):
    def test_all_planned_boss_assets_are_loadable(self):
        root = Path(__file__).resolve().parents[1] / "assets" / "sprites"
        assets = list(root.glob("boss_*.png")) + list(root.glob("effect_boss_*.png"))
        self.assertEqual((len(list(root.glob("boss_*.png"))),
                          len(list(root.glob("effect_boss_*.png")))), (10, 10))
        self.assertTrue(all(pygame.image.load(asset).get_size() == (1254, 1254) for asset in assets))

    def test_restart_and_keyboard_alternatives(self):
        self.assertEqual(keyboard_command(pygame, SimpleNamespace(key=pygame.K_r, unicode="r")), "restart")
        self.assertEqual(keyboard_command(pygame, SimpleNamespace(key=pygame.K_F5, unicode="")), "restart")
        self.assertEqual(keyboard_command(
            pygame, SimpleNamespace(key=0, unicode="", scancode=pygame.KSCAN_R)), "restart")
        self.assertEqual(keyboard_command(pygame, SimpleNamespace(key=pygame.K_KP3, unicode="")), "agent_3")
        self.assertEqual(keyboard_command(pygame, SimpleNamespace(key=pygame.K_4, unicode="4")), "agent_4")
        self.assertEqual(keyboard_command(pygame, SimpleNamespace(key=pygame.K_RIGHT, unicode="")), "faster")
        self.assertEqual(keyboard_command(pygame, SimpleNamespace(key=pygame.K_l, unicode="l")), "level_select")

    def test_level_selection_accepts_only_campaign_range(self):
        self.assertEqual(parse_level_selection("1"), 1)
        self.assertEqual(parse_level_selection("100"), 100)
        self.assertIsNone(parse_level_selection("0"))
        self.assertIsNone(parse_level_selection("101"))
        self.assertIsNone(parse_level_selection("boss"))

    def test_restart_resets_finished_level_and_cancels_pending_work(self):
        env = ArenaEnv(ArenaConfig(width=5, height=5, walls=0, enemies=0, gems=0, fires=0, medkits=0))
        env.tick, env.done, env.player.hp = 20, True, 0
        pending = SimpleNamespace(cancelled=False)
        pending.cancel = lambda: setattr(pending, "cancelled", True)
        self.assertIsNone(restart_level(env, 7, pending))
        self.assertTrue(pending.cancelled)
        self.assertEqual((env.tick, env.done, env.player.hp, env.seed), (0, False, 100, 7))

    def test_switching_to_local_agent_cancels_pending_decision(self):
        pending = SimpleNamespace(cancelled=False)
        pending.cancel = lambda: setattr(pending, "cancelled", True)
        remote, pending_after = switch_agent("jev", 7, pending)
        self.assertTrue(pending.cancelled)
        self.assertEqual(remote.name, "jev")
        self.assertIsNone(pending_after)
        agent, pending_after = switch_agent("rule", 7, pending_after)
        self.assertEqual(agent.name, "rule")
        self.assertIsNone(pending_after)


if __name__ == "__main__":
    unittest.main()
