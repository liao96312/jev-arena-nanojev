import unittest
from types import SimpleNamespace

import pygame

from arena import ArenaConfig, ArenaEnv
from scripts.play_gui import keyboard_command, restart_level


class GuiControlTests(unittest.TestCase):
    def test_restart_and_keyboard_alternatives(self):
        self.assertEqual(keyboard_command(pygame, SimpleNamespace(key=pygame.K_r, unicode="r")), "restart")
        self.assertEqual(keyboard_command(pygame, SimpleNamespace(key=pygame.K_F5, unicode="")), "restart")
        self.assertEqual(keyboard_command(
            pygame, SimpleNamespace(key=0, unicode="", scancode=pygame.KSCAN_R)), "restart")
        self.assertEqual(keyboard_command(pygame, SimpleNamespace(key=pygame.K_KP3, unicode="")), "agent_3")
        self.assertEqual(keyboard_command(pygame, SimpleNamespace(key=pygame.K_RIGHT, unicode="")), "faster")

    def test_restart_resets_finished_level_and_cancels_pending_work(self):
        env = ArenaEnv(ArenaConfig(width=5, height=5, walls=0, enemies=0, gems=0, fires=0, medkits=0))
        env.tick, env.done, env.player.hp = 20, True, 0
        pending = SimpleNamespace(cancelled=False)
        pending.cancel = lambda: setattr(pending, "cancelled", True)
        self.assertIsNone(restart_level(env, 7, pending))
        self.assertTrue(pending.cancelled)
        self.assertEqual((env.tick, env.done, env.player.hp, env.seed), (0, False, 100, 7))


if __name__ == "__main__":
    unittest.main()
