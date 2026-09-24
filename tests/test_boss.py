import unittest

from agents import RuleAgent
from arena import ArenaEnv, campaign_config
from arena.candidates import build_candidates
from arena.entities import Action
from arena.observation import encode_state
from nanojev_adapter.policy import select_action


class PrismBossTests(unittest.TestCase):
    def test_fixed_room_and_model_context(self):
        env = ArenaEnv(campaign_config(10))
        self.assertEqual(env.player.position, (10, 15))
        self.assertEqual(env.reflectors, {(9, 12), (11, 12), (8, 13), (12, 13)})
        self.assertEqual({y for _, y in env.reflectors}, {12, 13})
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
        env.player.position = (9, 15)
        env.boss.target = env.player.position
        self.assertEqual(env.boss_ray()[-1], (9, 12))
        self.assertFalse(env.imminent_threats())
        env.step(Action.WAIT)
        reflected = env.step(Action.WAIT)
        self.assertIn("boss_reflect:1", reflected.events)
        self.assertEqual(env.player.hp, 100)
        self.assertEqual(env.boss.used_reflectors, {(9, 12)})
        env.boss.position = (10, 8)
        env.boss.target = (9, 15)
        self.assertEqual(env.boss_ray()[-1], (9, 15))
        env.step(Action.WAIT)
        used = env.step(Action.WAIT)
        self.assertNotIn("boss_reflect:2", used.events)
        self.assertIn("damage:boss_prism:14", used.events)

        env.boss.position = (10, 8)
        env.boss.target = (10, 15)
        env.player.position = (10, 14)
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

    def test_default_hybrid_policy_pursues_boss_kill_with_flat_model_scores(self):
        env = ArenaEnv(campaign_config(10))
        events = []
        for _ in range(100):
            if env.done:
                break
            scores = {action.value: 1.0 for action in env.legal_actions()}
            choice, _ = select_action(scores, env, "hybrid")
            events.extend(env.step(choice).events)
        self.assertTrue(env.done)
        self.assertEqual(env.player.hp, 100)
        self.assertGreaterEqual(sum(event.startswith("boss_reflect:") for event in events), 9)
        self.assertIn("boss_prism_phase_two", events)
        self.assertTrue(any(event.startswith("boss_cover_break:") for event in events))
        self.assertLess(len(env.breakable_walls), 4)
        self.assertEqual(len(env.reflectors), 4)
        self.assertIn("boss_defeated", events)

    def test_prism_sweep_warning_matches_damage_and_cover_breaks(self):
        env = ArenaEnv(campaign_config(10))
        env.round = 3
        env.boss.hp = 100
        env.boss.sweep = True
        env.boss.target = (10, 15)
        env.player.position = (11, 15)
        self.assertIn(("prism_warden/beam", 14), env.imminent_threats())
        env.step(Action.WAIT)
        hit = env.step(Action.WAIT)
        self.assertIn("boss_prism_sweep", hit.events)
        self.assertIn("damage:boss_prism:14", hit.events)

        env = ArenaEnv(campaign_config(10))
        env.round = 3
        env.boss.target = (6, 10)
        self.assertEqual(env.boss_ray()[-1], (6, 10))
        self.assertNotIn(("prism_warden/beam", 14), env.imminent_threats())
        env.step(Action.WAIT)
        broken = env.step(Action.WAIT)
        self.assertIn("boss_cover_break:6:10", broken.events)
        self.assertNotIn((6, 10), env.walls | env.breakable_walls)
        self.assertEqual(len(env.reflectors), 4)

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

    def test_furnace_room_and_two_distinct_attacks(self):
        env = ArenaEnv(campaign_config(20))
        self.assertEqual(env.coolant_valves, {(7, 12), (10, 12), (13, 12)})
        self.assertEqual(env.player.position, (10, 16))
        self.assertTrue(env.player.loadout.bow)
        self.assertFalse(env.player.loadout.pistol)
        self.assertEqual(env.bow_pickups, {(10, 15)})
        self.assertEqual(env.arrow_bundles, {(6, 16), (14, 16)})
        self.assertGreater(len(env.forge_floor), 220)
        self.assertTrue((env.coolant_valves | env.medkits | env.arrow_bundles) <= env._reachable_cells())
        self.assertIn("Boss furnace", encode_state(env))
        env.player.position = (10, 12)
        self.assertIn("furnace_hydra", build_candidates(env)[Action.SHOOT_BOW_N.value])
        env.player.position = (10, 16)

        env.round = 3
        env.boss.attack_kind = "wave"
        env.boss.head_x = 10
        env.boss.target = (10, 16)
        self.assertIn(("furnace_hydra/wave", 22), env.imminent_threats())
        env.step(Action.WAIT)
        wave = env.step(Action.WAIT)
        self.assertIn("damage:furnace_wave:22", wave.events)

        env.player.position = (10, 12)
        env.boss.attack_kind = "wave"
        env.boss.head_x = 10
        env.boss.target = (10, 16)
        env.step(Action.WAIT)
        valve = env.step(Action.WAIT)
        self.assertIn("furnace_valve:10", valve.events)
        self.assertNotIn("damage:furnace_wave:22", valve.events)

        env.boss.attack_kind = "fireball"
        env.boss.target = env.player.position
        self.assertIn(("furnace_hydra/fireball", 16), env.imminent_threats())
        env.step(Action.WAIT)
        fireball = env.step(Action.WAIT)
        self.assertIn("damage:furnace_fireball:16", fireball.events)

    def test_rule_agent_can_finish_furnace_without_damage(self):
        env, agent = ArenaEnv(campaign_config(20)), RuleAgent()
        events = []
        for _ in range(100):
            if env.done:
                break
            events.extend(env.step(agent.act(env)).events)
        self.assertTrue(env.done)
        self.assertEqual(env.player.hp, 100)
        self.assertEqual(events.count("boss_shield_break"), 2)
        self.assertIn("furnace_phase_two", events)
        self.assertTrue(any(event.startswith("furnace_ignite:") for event in events))
        self.assertIn("boss_defeated", events)

    def test_default_hybrid_policy_pursues_furnace_kill_with_flat_model_scores(self):
        env = ArenaEnv(campaign_config(20))
        events = []
        for _ in range(100):
            if env.done:
                break
            scores = {action.value: 1.0 for action in env.legal_actions()}
            choice, _ = select_action(scores, env, "hybrid")
            events.extend(env.step(choice).events)
        self.assertTrue(env.done)
        self.assertEqual(env.player.hp, 100)
        self.assertEqual(events.count("boss_shield_break"), 2)
        self.assertIn("furnace_phase_two", events)
        self.assertTrue(any(event.startswith("furnace_ignite:") for event in events))
        self.assertIn("boss_defeated", events)

    def test_furnace_phase_two_wide_wave_and_temporary_fire(self):
        env = ArenaEnv(campaign_config(20))
        env.round = 3
        env.boss.hp = 50
        env.boss.attack_kind = "wave"
        env.boss.head_x = 10
        env.boss.target = (10, 16)
        env.player.position = (11, 14)
        self.assertIn(("furnace_hydra/wave", 12), env.imminent_threats())
        env.step(Action.WAIT)
        wave = env.step(Action.WAIT)
        self.assertIn("damage:furnace_wave:12", wave.events)

        env = ArenaEnv(campaign_config(20))
        env.round = 3
        env.boss.hp = 50
        env.boss.attack_kind = "fireball"
        env.boss.target = (10, 12)
        env.player.position = (16, 16)
        env.step(Action.WAIT)
        fireball = env.step(Action.WAIT)
        self.assertIn("furnace_ignite:11:12", fireball.events)
        self.assertIn((11, 12), env.fires)
        self.assertNotIn((10, 12), env.fires)
        for _ in range(6):
            env.step(Action.WAIT)
        self.assertNotIn((11, 12), env.fires)


class StormBossTests(unittest.TestCase):
    def test_staggered_room_chain_warning_and_grounding(self):
        env = ArenaEnv(campaign_config(30))
        self.assertFalse(env.gems)
        self.assertEqual(env.player.position, (10, 16))
        self.assertEqual(env.grounding_pylons, {(8, 9), (12, 9), (11, 13), (7, 13)})
        self.assertEqual(env.relay_pads, {(8, 8), (10, 8), (12, 8)})
        self.assertTrue((env.relay_pads | env.grounding_pylons | env.medkits) <= env._reachable_cells())
        env.round = 3
        env.player.position = (10, 8)
        env.boss.target = env.player.position
        self.assertEqual(len(env.storm_chain()), 6)
        self.assertIn("Boss storm", encode_state(env))
        self.assertIn("storm_choir", build_candidates(env)[Action.SHOOT_PISTOL_N.value])
        self.assertFalse(env.imminent_threats())
        env.step(Action.WAIT)
        result = env.step(Action.WAIT)
        self.assertIn("storm_grounded", result.events)
        self.assertIn("boss_shield_break", result.events)
        self.assertEqual(env.player.hp, 100)

    def test_surge_damage_matches_warning(self):
        env = ArenaEnv(campaign_config(30))
        env.round = 3
        env.boss.attack_kind = "surge"
        env.boss.target = env.player.position
        self.assertIn(("storm_choir/surge", 26), env.imminent_threats())
        env.step(Action.WAIT)
        hit = env.step(Action.WAIT)
        self.assertIn("damage:storm_surge:26", hit.events)

    def test_default_hybrid_policy_finishes_storm_without_damage(self):
        env = ArenaEnv(campaign_config(30))
        events = []
        for _ in range(100):
            if env.done:
                break
            scores = {action.value: 1.0 for action in env.legal_actions()}
            choice, _ = select_action(scores, env, "hybrid")
            events.extend(env.step(choice).events)
        self.assertTrue(env.done)
        self.assertEqual(env.player.hp, 100)
        self.assertEqual(events.count("storm_grounded"), 2)
        self.assertIn("storm_phase_two", events)
        self.assertIn("storm_relay_shift", events)
        self.assertIn("boss_defeated", events)

    def test_storm_phase_two_surge_and_shifted_relays(self):
        env = ArenaEnv(campaign_config(30))
        env.round = 3
        env.boss.hp = 80
        env.boss.attack_kind = "surge"
        env.boss.target = env.player.position
        self.assertIn(("storm_choir/surge", 30), env.imminent_threats())
        env.step(Action.WAIT)
        surge = env.step(Action.WAIT)
        self.assertIn("damage:storm_surge:30", surge.events)

        env = ArenaEnv(campaign_config(30))
        env.round = 3
        env.boss.hp = 80
        env.boss.exposed_rounds = 1
        env.step(Action.WAIT)
        shifted = env.step(Action.WAIT)
        self.assertIn("storm_relay_shift", shifted.events)
        self.assertEqual(env.relay_pads, {(7, 8), (13, 8)})
        self.assertTrue(env.relay_pads <= env._reachable_cells())
        for pad in env.relay_pads:
            env.boss.target = pad
            self.assertEqual(len(env.storm_chain()), 6)


class ChronoBossTests(unittest.TestCase):
    def test_flank_changes_side_with_player_without_teleporting(self):
        right = ArenaEnv(campaign_config(40))
        right.round = 3
        right.player.position = (14, 11)
        right.step(Action.WAIT)
        right.step(Action.WAIT)
        self.assertEqual(right.boss.position, (11, 7))
        self.assertEqual(right.boss.phase, "slash")
        right.step(Action.WAIT)
        right.step(Action.WAIT)
        self.assertEqual(right.boss.leap_target, (14, 7))

        left = ArenaEnv(campaign_config(40))
        left.round = 3
        left.player.position = (6, 11)
        left.boss.position = (9, 7)
        left.step(Action.WAIT)
        left.step(Action.WAIT)
        self.assertEqual(left.boss.position, (11, 7))
        self.assertEqual(left.boss.phase, "flank")
        left.step(Action.WAIT)
        left.step(Action.WAIT)
        self.assertEqual(left.boss.position, (12, 7))
        self.assertEqual(left.boss.phase, "slash")

    def test_room_leap_warning_and_anchor_counter(self):
        env = ArenaEnv(campaign_config(40))
        self.assertEqual(env.time_anchors, {(9, 11), (14, 11)})
        self.assertTrue((env.time_anchors | env.medkits | env.energy_cells) <= env._reachable_cells())
        self.assertIn("Boss chrono", encode_state(env))
        env.round = 3
        env.player.position = (9, 11)
        env.step(Action.WAIT)
        flank = env.step(Action.WAIT)
        self.assertIn("boss_move:10:7:12:7", flank.events)
        self.assertEqual(env.boss.slash_target, (9, 11))
        self.assertIn(("chrono_mantis/slash", 24), env.imminent_threats())
        env.step(Action.MOVE_S)
        slash = env.step(Action.MOVE_S)
        self.assertIn("chrono_leap_aim:9:7", slash.events)
        self.assertEqual(env.boss.leap_countdown, 2)
        self.assertEqual(env.boss.position, (12, 7))
        self.assertEqual(env._distance(env.boss.position, env.boss.leap_target), 3)
        env.step(Action.DASH_N)
        env.step(Action.WAIT)
        self.assertEqual(env.boss.position, (12, 8))
        self.assertEqual(env._distance(env.boss.position, env.boss.leap_target), 4)
        env.step(Action.WAIT)
        leap = env.step(Action.WAIT)
        self.assertIn("chrono_anchor", leap.events)
        self.assertIn("boss_shield_break", leap.events)
        self.assertEqual(env.boss.position, (9, 7))
        self.assertEqual(env.player.hp, 100)


    def test_occupied_leap_landing_is_cancelled_not_retargeted(self):
        env = ArenaEnv(campaign_config(40))
        env.round = 3
        env.player.position = (9, 7)
        env.boss.position = (12, 7)
        env.boss.phase = "leap"
        env.boss.leap_target = (9, 7)
        env.boss.leap_countdown = 1
        env.boss.slash_target = (10, 11)
        env.step(Action.WAIT)
        result = env.step(Action.WAIT)
        self.assertIn("chrono_leap_cancel", result.events)
        self.assertEqual(env.boss.position, (12, 7))
        self.assertEqual(env.player.hp, 100)

    def test_slash_and_delayed_echo_match_warning(self):
        env = ArenaEnv(campaign_config(40))
        env.round = 3
        env.boss.phase = "slash"
        env.boss.slash_target = env.player.position
        self.assertIn(("chrono_mantis/slash", 24), env.imminent_threats())
        env.step(Action.WAIT)
        slash = env.step(Action.WAIT)
        self.assertIn("damage:chrono_slash:24", slash.events)
        self.assertIn(("chrono_mantis/echo", 18), env.imminent_threats())
        for _ in range(4):
            echo = env.step(Action.WAIT)
        self.assertIn("damage:chrono_echo:18", echo.events)
        self.assertEqual(env.player.hp, 58)

    def test_default_hybrid_policy_finishes_mantis_without_damage(self):
        env = ArenaEnv(campaign_config(40))
        events = []
        for _ in range(100):
            if env.done:
                break
            scores = {action.value: 1.0 for action in env.legal_actions()}
            choice, _ = select_action(scores, env, "hybrid")
            events.extend(env.step(choice).events)
        self.assertTrue(env.done)
        self.assertEqual(env.player.hp, 100)
        self.assertEqual(events.count("chrono_anchor"), 3)
        self.assertIn("chrono_phase_two", events)
        self.assertIn("chrono_anchor_shift", events)
        self.assertIn("boss_defeated", events)

    def test_phase_two_moves_anchor_and_melee_can_finish_core(self):
        env = ArenaEnv(campaign_config(40))
        env.round = 3
        env.boss.hp = 100
        env.boss.position = (9, 7)
        env.boss.exposed_rounds = 3
        for _ in range(6):
            result = env.step(Action.WAIT)
        self.assertIn("chrono_anchor_shift", result.events)
        self.assertEqual(env.time_anchors, {(9, 13), (14, 13)})
        self.assertEqual(env.boss.position, (10, 7))
        self.assertTrue(env.time_anchors <= env._reachable_cells())

        env.boss.phase = "leap"
        env.boss.leap_target = (9, 7)
        env.boss.leap_countdown = 1
        env.player.position = (9, 13)
        env.step(Action.WAIT)
        leap = env.step(Action.WAIT)
        self.assertIn("chrono_anchor", leap.events)

        env.boss.position = (9, 7)
        env.boss.hp = 20
        env.boss.exposed_rounds = 2
        env.player.position = (9, 8)
        env.player.loadout.energy = 0
        scores = {action.value: 1.0 for action in env.legal_actions()}
        choice, _ = select_action(scores, env, "hybrid")
        self.assertEqual(choice, "attack_n")
        self.assertIn("boss_defeated", env.step(choice).events)

    def test_rule_agent_can_finish_mantis_without_damage(self):
        env, agent = ArenaEnv(campaign_config(40)), RuleAgent()
        for _ in range(100):
            if env.done:
                break
            env.step(agent.act(env))
        self.assertTrue(env.done)
        self.assertEqual(env.player.hp, 100)


class VoidBossTests(unittest.TestCase):
    def test_room_and_gravity_counter(self):
        env = ArenaEnv(campaign_config(50))
        self.assertIsNotNone(env.boss)
        self.assertFalse(env.gems)
        self.assertEqual(env.gravity_nodes, {(7, 11), (10, 12), (13, 11)})
        self.assertTrue((env.gravity_nodes | env.medkits | env.energy_cells) <= env._reachable_cells())
        self.assertIn("Boss void", encode_state(env))
        env.round = 3
        env.player.position = (10, 12)
        env.boss.target = env.player.position
        env.boss.attack_kind = "mine"
        env.step(Action.WAIT)
        pulse = env.step(Action.WAIT)
        self.assertIn("void_drain:1", pulse.events)
        self.assertEqual(env.player.hp, 100)
        self.assertEqual(env.boss.drained_nodes, {(10, 12)})

    def test_beam_warning_and_pull_never_enters_pit(self):
        env = ArenaEnv(campaign_config(50))
        env.round = 3
        env.boss.attack_kind = "beam"
        env.boss.target = env.player.position
        self.assertIn(("void_angler/beam", 22), env.imminent_threats())
        env.step(Action.WAIT)
        beam = env.step(Action.WAIT)
        self.assertIn("damage:void_beam:22", beam.events)

        env = ArenaEnv(campaign_config(50))
        env.round = 3
        env.player.position = (4, 10)
        env.boss.attack_kind = "mine"
        env.boss.target = (3, 10)
        env.step(Action.WAIT)
        pulse = env.step(Action.WAIT)
        self.assertNotIn("void_pull:4:10:3:10", pulse.events)
        self.assertNotIn(env.player.position, env.pits)

    def test_default_hybrid_and_rule_finish_without_damage(self):
        for rule in (False, True):
            env, agent = ArenaEnv(campaign_config(50)), RuleAgent()
            events = []
            for _ in range(120):
                if env.done:
                    break
                action = (agent.act(env) if rule else
                          select_action({action.value: 1.0 for action in env.legal_actions()}, env, "hybrid")[0])
                events.extend(env.step(action).events)
            self.assertTrue(env.done)
            self.assertEqual(env.player.hp, 100)
            self.assertEqual(events.count("boss_shield_break"), 2)
            self.assertIn("boss_defeated", events)


if __name__ == "__main__":
    unittest.main()
