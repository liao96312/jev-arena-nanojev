import random
import unittest

from agents import RuleAgent
from arena import ArenaEnv, campaign_config
from arena.candidates import build_candidates
from arena.entities import Action
from arena.observation import encode_state
from nanojev_adapter.policy import select_action


class PrismBossTests(unittest.TestCase):
    def test_reflector_can_be_shoved_to_change_ray_geometry(self):
        env = ArenaEnv(campaign_config(10))
        env.reflectors = {(9, 12)}
        env.player.position = (8, 12)
        self.assertIn(Action.SHOVE_E, env.legal_actions())
        result = env.step(Action.SHOVE_E)
        self.assertIn("prism_reflector_shove:9:12:10:12", result.events)
        self.assertEqual(env.reflectors, {(10, 12)})

    def test_reflector_layout_varies_by_seed_and_recharges_after_delay(self):
        first, second = ArenaEnv(campaign_config(10)), ArenaEnv(campaign_config(10))
        second.reset(1)
        self.assertNotEqual(first.reflectors, second.reflectors)
        boss = first.boss
        boss.used_reflectors.add(next(iter(first.reflectors)))
        boss.reflector_regen = 2
        first.round = 3
        first._resolve_boss([])
        self.assertTrue(boss.used_reflectors)
        first._resolve_boss([])
        self.assertFalse(boss.used_reflectors)

    def test_reflected_shot_can_chain_an_unwarned_followup(self):
        env = ArenaEnv(campaign_config(10))
        env.round = 3
        env.player.position = (9, 15)
        env.boss.target = env.player.position
        env.rng.random = lambda: 0.0
        events = []
        env._resolve_boss(events)
        self.assertIn("boss_reflect:1", events)
        self.assertIn("boss_prism_followup:9:12:9:15", events)
        self.assertIn("damage:boss_prism_followup:7", events)

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
        for _ in range(140):
            if env.done:
                break
            scores = {action.value: 1.0 for action in env.legal_actions()}
            choice, _ = select_action(scores, env, "hybrid")
            events.extend(env.step(choice).events)
        self.assertTrue(env.done)
        self.assertGreater(env.player.hp, 0)
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

    def test_rapid_wave_combo_makes_valve_unsafe_but_warns_columns(self):
        env = ArenaEnv(campaign_config(20))
        env.round = 3
        env.boss.hp = 50
        env.boss.attack_kind = "wave"
        env.boss.target = (10, 16)
        env.boss.head_x = 10
        env.boss.wave_columns = (10,)
        env.boss.combo_queued = True
        env.player.position = (10, 12)
        first = []
        env._resolve_furnace(env.boss, first)
        self.assertIn("furnace_valve:10", first)
        self.assertIn("furnace_rapid_combo", first)
        self.assertTrue(env.boss.wave_rapid)
        env.player.position = (env.boss.head_x, 12)
        self.assertIn(("furnace_hydra/wave", 22), env.imminent_threats())
        second = []
        env._resolve_furnace(env.boss, second)
        self.assertIn("damage:furnace_wave:22", second)
        self.assertFalse(any(event.startswith("furnace_valve:") for event in second))

    def test_rule_agent_can_finish_furnace_without_damage(self):
        env, agent = ArenaEnv(campaign_config(20)), RuleAgent()
        events = []
        for _ in range(100):
            if env.done:
                break
            events.extend(env.step(agent.act(env)).events)
        self.assertTrue(env.done)
        self.assertEqual(env.player.hp, 100)
        self.assertGreaterEqual(events.count("boss_shield_break"), 2)
        self.assertIn("furnace_phase_two", events)
        self.assertTrue(any(event.startswith("furnace_valve_strike:") for event in events))
        self.assertIn("boss_defeated", events)

    def test_furnace_valve_can_be_opened_from_the_side(self):
        env = ArenaEnv(campaign_config(20))
        env.player.position = (6, 12)
        env.boss.head_x = 7
        env.boss.attack_kind = "wave"
        env.boss.target = (7, 16)
        self.assertIn(Action.ATTACK_E, env.legal_actions())
        result = env.step(Action.ATTACK_E)
        self.assertIn("furnace_valve_strike:7", result.events)
        self.assertEqual(env.player.position, (6, 12))
        self.assertIn(7, env.boss.valves_opened)

    def test_default_hybrid_policy_pursues_furnace_kill_with_flat_model_scores(self):
        env = ArenaEnv(campaign_config(20))
        events = []
        for _ in range(160):
            if env.done:
                break
            scores = {action.value: 1.0 for action in env.legal_actions()}
            choice, _ = select_action(scores, env, "hybrid")
            events.extend(env.step(choice).events)
        self.assertTrue(env.done)
        self.assertGreater(env.player.hp, 0)
        self.assertGreaterEqual(events.count("boss_shield_break"), 2)
        self.assertIn("furnace_phase_two", events)
        self.assertIn("boss_defeated", events)

    def test_furnace_summons_one_warned_ember_minion(self):
        env = ArenaEnv(campaign_config(20))
        env.round = 4
        events = []
        env._summon_boss_minion(env.boss, events)
        self.assertIn("boss_summon:furnace:6:7", events)
        self.assertEqual(env.enemies[0].summoned_by, "furnace")
        self.assertEqual(env._enemy_move_interval(env.enemies[0]), 1)
        self.assertIsNotNone(env.enemies[0].intent)
        env.round = 5
        env._summon_boss_minion(env.boss, events)
        self.assertEqual(len(env.enemies), 1)
        env.boss.exposed_rounds = 1
        env.boss.hp = 1
        env._damage_entity(env.boss, 1, events, "pistol")
        self.assertEqual(env.enemies, [])

    def test_furnace_can_summon_again_after_minion_is_gone(self):
        env = ArenaEnv(campaign_config(20))
        env.round = 4
        events = []
        env._summon_boss_minion(env.boss, events)
        self.assertEqual(env.boss.summons, 1)
        env.round = 8
        env._summon_boss_minion(env.boss, events)
        self.assertEqual(env.boss.summons, 2)
        self.assertEqual(len(env.enemies), 2)
        self.assertTrue(all(env._enemy_move_interval(enemy) == 1 for enemy in env.enemies))

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
    def test_same_relay_pad_is_not_a_permanent_safe_spot(self):
        env = ArenaEnv(campaign_config(30))
        env.player.position = (8, 8)
        env.boss.target = (8, 8)
        env.boss.attack_kind = "chain"
        env.boss.last_ground_pad = (8, 8)
        self.assertIn(("storm_choir/arc", env.boss.arc_damage), env.imminent_threats())
        events = []
        env._resolve_storm(env.boss, events)
        self.assertIn("damage:storm_arc:14", events)
        self.assertNotIn("boss_shield_break", events)

    def test_emp_short_circuits_warned_chain_without_standing_on_pad(self):
        env = ArenaEnv(campaign_config(30))
        env.round = 3
        env.player.position = (8, 10)
        env.boss.attack_kind = "chain"
        env.boss.target = (8, 10)
        self.assertIn(Action.EMP, env.legal_actions())
        result = env.step(Action.EMP)
        self.assertIn("storm_emp_ground", result.events)
        self.assertIn("boss_shield_break", result.events)
        self.assertNotIn(env.player.position, env.relay_pads)

    def test_rule_agent_can_finish_storm_boss(self):
        env, agent = ArenaEnv(campaign_config(30)), RuleAgent()
        events = []
        for _ in range(100):
            if env.done:
                break
            events.extend(env.step(agent.act(env)).events)
        self.assertIn("boss_defeated", events)
        self.assertGreater(env.player.hp, 0)

    def test_unannounced_net_is_visible_before_next_round_damage(self):
        env = ArenaEnv(campaign_config(30))
        env.round = 3
        env.player.position = (10, 12)
        env.boss.attack_kind = "surge"
        env.boss.target = (8, 8)
        env.rng.random = lambda: 0.0
        placed = []
        env._resolve_storm(env.boss, placed)
        self.assertIn("storm_net_place:10:12", placed)
        self.assertIn(("storm_choir/net", 12), env.imminent_threats())
        env.player.position = (10, 14)
        fired = []
        env._resolve_storm(env.boss, fired)
        self.assertIn("storm_net_fire:10:12", fired)
        self.assertFalse(any(event.startswith("damage:storm_net:") for event in fired))

    def test_moves_to_distinct_chain_and_surge_positions_before_aim(self):
        env = ArenaEnv(campaign_config(30))
        env.player.position = (10, 10)
        events = []
        env._resolve_storm(env.boss, events)
        self.assertIn("boss_move:10:5:8:5", events)
        self.assertEqual(env.boss.target, (10, 10))
        events.clear()
        env._resolve_storm(env.boss, events)
        self.assertEqual(env.boss.position, (8, 6))
        self.assertIn("boss_move:8:5:8:6", events)
        self.assertTrue(any(event.startswith("storm_chain:8:5:") for event in events))
        events.clear()
        env._resolve_storm(env.boss, events)
        self.assertIn("boss_move:8:6:11:6", events)
        self.assertIn("storm_aim:surge:10:10", events)

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
    def test_adjacent_anchor_can_be_primed_then_evaded(self):
        env = ArenaEnv(campaign_config(40))
        env.round = 3
        env.player.position = (8, 11)
        env.boss.phase = "leap"
        env.boss.leap_target = (9, 11)
        env.boss.slash_target = (8, 11)
        env.boss.leap_countdown = 1
        self.assertIn(Action.ATTACK_E, env.legal_actions())
        prime = env.step(Action.ATTACK_E)
        self.assertIn("chrono_anchor_prime:9:11", prime.events)
        leap = env.step(Action.WAIT)
        self.assertIn("boss_shield_break", leap.events)
        self.assertEqual(env.player.position, (8, 11))
        self.assertIsNone(env.boss.primed_anchor)

    def test_opening_stalks_without_attacking_during_spawn_protection(self):
        env = ArenaEnv(campaign_config(40))
        env.step(Action.WAIT)
        opening = env.step(Action.WAIT)
        self.assertIn("boss_move:10:7:10:9", opening.events)
        self.assertFalse(any(event.startswith(("chrono_slash:", "chrono_leap:", "damage:"))
                             for event in opening.events))

    def test_some_leaps_chain_into_next_warned_slash(self):
        env, agent = ArenaEnv(campaign_config(40)), RuleAgent()
        env.reset(1)
        chained = False
        for _ in range(100):
            if env.done:
                break
            events = env.step(agent.act(env)).events
            if (any(event.startswith("chrono_leap:") for event in events) and
                    any(event.startswith("chrono_slash_aim:") for event in events)):
                chained = True
                self.assertEqual(env.boss.phase, "slash")
                expected = (env.boss.slash_damage // 2 if env.boss.slash_target in env.time_anchors
                            else env.boss.slash_damage)
                self.assertIn(("chrono_mantis/slash", expected),
                              env.imminent_threats(env.boss.slash_target))
                break
        self.assertTrue(chained)

    def test_anchor_counter_temporarily_slows_flank(self):
        env = ArenaEnv(campaign_config(40))
        env.player.position = (6, 12)
        env.boss.position = (9, 7)
        env.boss.slow_rounds = 4
        events = []
        env._resolve_chrono(env.boss, events)
        self.assertEqual(env.boss.position[0], 10)
        self.assertEqual(env.boss.slow_rounds, 3)

    def test_anchor_softens_slash_but_is_not_invulnerable(self):
        env = ArenaEnv(campaign_config(40))
        env.player.position = (9, 11)
        env.boss.phase = "slash"
        env.boss.slash_target = (9, 11)
        self.assertIn(("chrono_mantis/slash", 12), env.imminent_threats())
        events = []
        env._resolve_chrono(env.boss, events)
        self.assertIn("chrono_anchor_guard", events)
        self.assertEqual(env.player.hp, 88)
        self.assertIn("damage:chrono_slash:12", events)
        self.assertEqual(env.boss.leap_target, (9, 11))

        env = ArenaEnv(campaign_config(40))
        rng = random.Random(16)
        hits = 0
        for _ in range(160):
            if env.done:
                break
            probabilities = {action.value: rng.random() + .01 for action in env.legal_actions()}
            action, _ = select_action(probabilities, env, "hybrid")
            hits += sum(event.startswith("boss_hit:") for event in env.step(action).events)
        self.assertTrue(env.done)
        self.assertGreater(hits, 0)
        self.assertGreater(env.player.hp, 0)

    def test_flank_changes_side_with_player_without_teleporting(self):
        right = ArenaEnv(campaign_config(40))
        right.round = 3
        right.player.position = (14, 11)
        right.step(Action.WAIT)
        right.step(Action.WAIT)
        self.assertEqual(right.boss.position, (11, 8))
        self.assertEqual(right.boss.phase, "slash")
        right.step(Action.WAIT)
        right.step(Action.WAIT)
        self.assertEqual(right.boss.leap_target, (14, 11))

        left = ArenaEnv(campaign_config(40))
        left.round = 3
        left.player.position = (6, 11)
        left.boss.position = (9, 7)
        left.step(Action.WAIT)
        left.step(Action.WAIT)
        self.assertEqual(left.boss.position, (12, 8))
        self.assertEqual(left.boss.phase, "slash")
        left.step(Action.WAIT)
        left.step(Action.WAIT)
        self.assertEqual(left.boss.position, (10, 10))
        self.assertEqual(left.boss.phase, "leap")

    def test_room_leap_warning_and_anchor_counter(self):
        env = ArenaEnv(campaign_config(40))
        self.assertEqual(env.time_anchors, {(9, 11), (14, 11)})
        self.assertTrue((env.time_anchors | env.medkits | env.energy_cells) <= env._reachable_cells())
        self.assertIn("Boss chrono", encode_state(env))
        env.round = 3
        env.player.position = (9, 13)
        env.step(Action.WAIT)
        flank = env.step(Action.WAIT)
        self.assertIn("boss_move:10:7:12:9", flank.events)
        self.assertEqual(env.boss.slash_target, (9, 13))
        self.assertIn(("chrono_mantis/slash", 24), env.imminent_threats())
        env.step(Action.DASH_N)
        slash = env.step(Action.WAIT)
        self.assertIn("chrono_leap_aim:9:11", slash.events)
        self.assertIn(env.boss.leap_countdown, (1, 2))
        self.assertEqual(env.boss.position, (10, 11))
        leap = None
        for _ in range(4):
            result = env.step(Action.WAIT)
            if "boss_shield_break" in result.events:
                leap = result
                break
        self.assertIsNotNone(leap)
        self.assertIn("chrono_anchor", leap.events)
        self.assertIn("boss_shield_break", leap.events)
        self.assertEqual(env.boss.slow_rounds, 1)
        self.assertNotEqual(env.boss.position, env.player.position)
        self.assertEqual(env.player.hp, 100)


    def test_occupied_leap_landing_hits_player_without_overlap(self):
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
        self.assertIn("damage:chrono_leap:30", result.events)
        self.assertNotEqual(env.boss.position, env.player.position)
        self.assertEqual(env.player.hp, 70)

    def test_jump_targets_player_cell_then_lands_there_if_dodged(self):
        env = ArenaEnv(campaign_config(40))
        env.boss.phase = "slash"
        env.boss.slash_target = (9, 11)
        env.player.position = (10, 13)
        events = []
        env._resolve_chrono(env.boss, events)
        self.assertEqual(env.boss.leap_target, (10, 13))
        self.assertIn("chrono_leap_aim:10:13", events)
        env.player.position = (11, 13)
        events = []
        for _ in range(2):
            env._resolve_chrono(env.boss, events)
            if any(event.startswith("chrono_leap:") for event in events):
                break
        self.assertEqual(env.boss.position, (10, 13))
        self.assertEqual(env.player.hp, 100)
        self.assertIn("chrono_leap:10:13", events)

    def test_slash_and_delayed_jump_match_warning(self):
        env = ArenaEnv(campaign_config(40))
        env.round = 3
        env.boss.phase = "slash"
        env.boss.slash_target = env.player.position
        self.assertIn(("chrono_mantis/slash", 24), env.imminent_threats())
        env.step(Action.WAIT)
        slash = env.step(Action.WAIT)
        self.assertIn("damage:chrono_slash:24", slash.events)
        self.assertIn(("chrono_mantis/leap", 30), env.imminent_threats())
        for _ in range(4):
            echo = env.step(Action.WAIT)
            if "damage:chrono_leap:30" in echo.events:
                break
        self.assertIn("damage:chrono_leap:30", echo.events)
        self.assertEqual(env.player.hp, 46)

    def test_default_hybrid_policy_finishes_mantis(self):
        env = ArenaEnv(campaign_config(40))
        events = []
        for _ in range(100):
            if env.done:
                break
            scores = {action.value: 1.0 for action in env.legal_actions()}
            choice, _ = select_action(scores, env, "hybrid")
            events.extend(env.step(choice).events)
        self.assertTrue(env.done)
        self.assertGreater(env.player.hp, 0)
        self.assertGreaterEqual(events.count("chrono_anchor"), 2)
        self.assertIn("chrono_phase_two", events)
        self.assertIn("chrono_anchor_shift", events)
        self.assertIn("boss_defeated", events)

    def test_leap_counter_allows_melee_then_retreat_wave_can_be_dodged(self):
        env = ArenaEnv(campaign_config(40))
        env.round = 3
        env.player.position = (9, 11)
        env.boss.position = (10, 11)
        env.boss.exposed_rounds = 4
        hp = env.boss.hp
        env.step(Action.ATTACK_E)
        env.step(Action.WAIT)
        self.assertEqual(env.boss.hp, hp - env.config.attack_damage)
        self.assertEqual(env.boss.position, (11, 9))
        env.step(Action.WAIT)
        retreat = env.step(Action.WAIT)
        self.assertEqual(env.boss.position, (13, 7))
        self.assertIn("chrono_retreat", retreat.events)
        self.assertEqual(env.boss.retreat_target, (9, 11))
        self.assertIn(("chrono_mantis/retreat", env.boss.echo_damage), env.imminent_threats())
        hit = env.clone()
        hit.step(Action.WAIT)
        self.assertIn("damage:chrono_retreat:18", hit.step(Action.WAIT).events)
        env.step(Action.MOVE_N)
        burst = env.step(Action.WAIT)
        self.assertIn("chrono_retreat_burst:9:11", burst.events)
        self.assertEqual(env.player.hp, 100)
        self.assertIsNone(env.boss.retreat_target)

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
        self.assertEqual(env.boss.position, (7, 7))
        self.assertTrue(env.time_anchors <= env._reachable_cells())

        env.boss.phase = "leap"
        env.boss.leap_target = (9, 13)
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

    def test_rule_agent_can_finish_mantis(self):
        env, agent = ArenaEnv(campaign_config(40)), RuleAgent()
        for _ in range(100):
            if env.done:
                break
            env.step(agent.act(env))
        self.assertTrue(env.done)
        self.assertGreater(env.player.hp, 0)


class VoidBossTests(unittest.TestCase):
    def test_aimed_mine_can_be_cut_from_adjacent_tile(self):
        env = ArenaEnv(campaign_config(50))
        env.player.position = (9, 12)
        env.boss.attack_kind = "mine"
        env.boss.target = (10, 12)
        self.assertIn(Action.ATTACK_E, env.legal_actions())
        result = env.step(Action.ATTACK_E)
        self.assertIn("void_node_cut:10:12", result.events)
        self.assertIn((10, 12), env.boss.drained_nodes)

    def test_third_node_opens_shorter_core_window(self):
        env = ArenaEnv(campaign_config(50))
        env.player.position = (13, 11)
        env.boss.attack_kind = "mine"
        env.boss.target = (13, 11)
        env.boss.drained_nodes = {(7, 11), (10, 12)}
        events = []
        env._resolve_void(env.boss, events)
        self.assertEqual(env.boss.exposed_rounds, 3)
        self.assertIn("boss_shield_break", events)

    def test_hook_pulls_and_temporarily_disables_move_without_empty_actions(self):
        env = ArenaEnv(campaign_config(50))
        env.round = 3
        env.player.position = (10, 12)
        env.boss.attack_kind = "hook"
        env.boss.target = env.player.position
        events = []
        env._resolve_void(env.boss, events)
        self.assertTrue(any(event.startswith("void_hook:") for event in events))
        self.assertIn("damage:void_hook:18", events)
        self.assertEqual(env.hooked_actions, 1)
        self.assertEqual(env._distance(env.player.position, env.boss.position), 1)
        legal = env.legal_actions()
        self.assertIn(Action.WAIT, legal)
        self.assertFalse(any(action.value.startswith(("move_", "dash_")) for action in legal))
        for action in (RuleAgent().act(env),
                       select_action({candidate.value: 1.0 for candidate in legal}, env, "hybrid")[0]):
            self.assertIn(action, legal)
            env.clone().step(action)
        env.step(Action.WAIT)
        self.assertEqual(env.hooked_actions, 0)
        self.assertTrue(any(action.value.startswith("move_") for action in env.legal_actions()))

    def test_angler_never_occupies_counter_nodes_while_approaching(self):
        env = ArenaEnv(campaign_config(50))
        env.boss.position = (7, 10)
        env.player.position = (11, 11)
        events = []
        env._void_approach(env.boss, (7, 11), events)
        self.assertEqual(env.boss.position, (7, 10))
        self.assertFalse(events)

    def test_casting_movement_keeps_locked_warning_and_exposed_window(self):
        env = ArenaEnv(campaign_config(50))
        env.round = 3
        env.player.position = (13, 11)
        boss = env.boss
        boss.attacks = 0
        boss.target = None
        env.step(Action.WAIT)
        aim = env.step(Action.WAIT)
        self.assertEqual((boss.position, boss.target, boss.attack_kind), ((12, 8), (13, 11), "mine"))
        self.assertIn("boss_move:10:6:12:8", aim.events)
        self.assertIn("void_aim:mine:13:11", aim.events)
        env.step(Action.WAIT)
        cast = env.step(Action.WAIT)
        self.assertIn("void_mine:13:11", cast.events)
        self.assertIn("void_drain:1", cast.events)
        self.assertEqual(boss.position, (13, 10))

        boss.target = None
        boss.exposed_rounds = 4
        env.step(Action.WAIT)
        stagger = env.step(Action.WAIT)
        self.assertTrue(any(event.startswith("boss_move:") for event in stagger.events))
        env.step(Action.WAIT)
        recovery = env.step(Action.WAIT)
        self.assertTrue(any(event.startswith("boss_move:") for event in recovery.events))
        self.assertEqual(boss.exposed_rounds, 2)

    def test_beam_origin_stays_fixed_until_firing(self):
        env = ArenaEnv(campaign_config(50))
        env.round = 3
        env.player.position = (10, 12)
        boss = env.boss
        boss.attacks = 1
        env.step(Action.WAIT)
        aim = env.step(Action.WAIT)
        self.assertIn("void_aim:beam:10:12", aim.events)
        self.assertEqual(boss.position, (7, 6))
        self.assertIn("void_warp", aim.events)
        self.assertIn("boss_move:10:6:7:6", aim.events)
        env.step(Action.WAIT)
        fire = env.step(Action.WAIT)
        self.assertIn("void_beam:7:6:10:12", fire.events)
        self.assertFalse(any(event.startswith("boss_move:") for event in fire.events))

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
            for seed in range(3):
                with self.subTest(rule=rule, seed=seed):
                    env, agent = ArenaEnv(campaign_config(50)), RuleAgent()
                    env.reset(seed)
                    events = []
                    for _ in range(120):
                        if env.done:
                            break
                        action = (agent.act(env) if rule else
                                  select_action({action.value: 1.0 for action in env.legal_actions()}, env, "hybrid")[0])
                        events.extend(env.step(action).events)
                    self.assertTrue(env.done)
                    self.assertEqual(env.player.hp, 100)
                    self.assertGreaterEqual(events.count("boss_shield_break"), 1)
                    self.assertIn("boss_defeated", events)


class IronBossTests(unittest.TestCase):
    def test_aimed_mature_vine_can_be_cut_without_standing_on_root(self):
        env = ArenaEnv(campaign_config(60))
        env.player.position = (9, 11)
        env.vine_walls.add((9, 10))
        env.walls.add((9, 10))
        env.boss.attack_kind = "flame"
        env.boss.target = (9, 12)
        self.assertIn(Action.ATTACK_N, env.legal_actions())
        result = env.step(Action.ATTACK_N)
        self.assertIn("iron_vine_cut:9:10", result.events)
        self.assertNotIn((9, 10), env.walls)
        self.assertIn(9, env.boss.refluxed_roots)

    def test_vine_reseeds_next_enemy_round_and_flame_is_frequent(self):
        env = ArenaEnv(campaign_config(60))
        env.vine_seeds.clear()
        env.vine_walls.clear()
        env.player.position = (10, 15)
        env.boss.attack_kind = "flame"
        env.boss.target = (9, 12)
        events = []
        env._resolve_iron(env.boss, events)
        self.assertEqual(env.vine_seeds[(9, 10)], 1)
        env.boss.attacks = 2
        env._resolve_iron(env.boss, events)
        self.assertIn("iron_vine_grow:9:10", events)
        self.assertEqual(env.boss.attack_kind, "flame")

    def test_iron_summons_one_vine_hunter(self):
        env = ArenaEnv(campaign_config(60))
        env.round = 6
        events = []
        env._summon_boss_minion(env.boss, events)
        self.assertIn("boss_summon:iron:6:8", events)
        self.assertEqual(env.enemies[0].summoned_by, "iron")
        env.enemies.clear()
        env._summon_boss_minion(env.boss, events)
        self.assertEqual(env.enemies, [])

    def test_seed_growth_flame_counter_and_safe_routes(self):
        env = ArenaEnv(campaign_config(60))
        self.assertFalse(env.gems)
        self.assertEqual(env.root_plates, {(6, 12), (9, 12), (12, 12), (15, 12)})
        self.assertEqual(len(env.vine_seeds), 4)
        self.assertTrue((env.root_plates | env.medkits | env.energy_cells) <= env._reachable_cells())
        self.assertIn("Boss iron", encode_state(env))
        env.round = 3
        env.player.position = (9, 12)
        env.step(Action.WAIT)
        env.step(Action.WAIT)
        self.assertEqual(env.vine_seeds[(9, 10)], 1)
        self.assertNotIn(("iron_gardener/flame", 22), env.imminent_threats())
        env.step(Action.WAIT)
        flame = env.step(Action.WAIT)
        self.assertIn("iron_vine_grow:9:10", flame.events)
        self.assertIn("iron_vine_burn:9:10", flame.events)
        self.assertIn("iron_reflux:1", flame.events)
        self.assertNotIn((9, 10), env.walls)
        self.assertTrue(env.root_plates <= env._reachable_cells())

    def test_thorn_warning_matches_damage(self):
        env = ArenaEnv(campaign_config(60))
        env.round = 3
        env.boss.attack_kind = "thorn"
        env.boss.target = env.player.position
        self.assertIn(("iron_gardener/thorn", 18), env.imminent_threats())
        env.step(Action.WAIT)
        result = env.step(Action.WAIT)
        self.assertIn("damage:iron_thorn:18", result.events)

    def test_default_hybrid_and_rule_finish_without_damage(self):
        for rule in (False, True):
            env, agent = ArenaEnv(campaign_config(60)), RuleAgent()
            events = []
            for _ in range(120):
                if env.done:
                    break
                action = (agent.act(env) if rule else
                          select_action({action.value: 1.0 for action in env.legal_actions()}, env, "hybrid")[0])
                events.extend(env.step(action).events)
            self.assertTrue(env.done)
            self.assertEqual(env.player.hp, 100)
            self.assertGreaterEqual(events.count("boss_shield_break"), 3)
            self.assertIn("boss_defeated", events)


class MirrorBossTests(unittest.TestCase):
    def test_exposed_mirror_can_dodge_one_shot_then_warn_counter(self):
        env = ArenaEnv(campaign_config(70))
        env.round = 3
        env.boss.exposed_rounds = 4
        env.boss.evade_ready = True
        env.player.position = (10, 11)
        hp, energy = env.boss.hp, env.player.loadout.energy
        shot = env.step(Action.SHOOT_PISTOL_N)
        self.assertIn("mirror_evade", shot.events)
        self.assertEqual(env.boss.hp, hp)
        self.assertEqual(env.player.loadout.energy, energy - 1)
        self.assertNotEqual(env.boss.position, (10, 6))
        warning = env.step(Action.WAIT)
        self.assertIn("mirror_echo_aim:10:11", warning.events)
        self.assertIn(("mirror_seraph/echo", env.boss.echo_damage), env.imminent_threats())
        env.step(Action.DASH_S)
        fired = env.step(Action.MOVE_S)
        self.assertIn("mirror_echo:10:11", fired.events)
        self.assertFalse(any(event.startswith("damage:mirror_echo:") for event in fired.events))

    def test_emp_blocks_mirror_evade_during_exposure(self):
        env = ArenaEnv(campaign_config(70))
        env.round = 3
        env.boss.exposed_rounds = 4
        env.boss.evade_ready = True
        env.player.position = (10, 8)
        self.assertIn(Action.EMP, env.legal_actions())
        jam = env.step(Action.EMP)
        self.assertIn("mirror_emp_jam", jam.events)
        hp = env.boss.hp
        shot = env.step(Action.SHOOT_PISTOL_N)
        self.assertNotIn("mirror_evade", shot.events)
        self.assertLess(env.boss.hp, hp)

    def test_other_bosses_reposition_during_exposure(self):
        for level in (30, 60, 80, 90, 100):
            with self.subTest(level=level):
                env = ArenaEnv(campaign_config(level))
                env.round = 3
                env.player.position = (12, 12)
                env.boss.exposed_rounds = 2 if level == 90 else 4
                if level == 100:
                    env.boss.seals = 4
                start = env.boss.position
                events = []
                env._resolve_boss(events)
                self.assertNotEqual(env.boss.position, start)
                self.assertTrue(any(event.startswith("boss_move:") for event in events))
                self.assertNotIn(env.boss.position, env.walls | env.pits)
                self.assertNotEqual(env.boss.position, env.player.position)

    def test_echo_hits_even_when_mirror_ray_breaks_a_lock(self):
        env = ArenaEnv(campaign_config(70))
        env.player.position = (10, 14)
        env.last_non_wait_action = "move_e"
        events = []
        env._resolve_mirror(env.boss, events)
        self.assertEqual(env.boss.echo_target, (10, 14))
        self.assertIn("mirror_echo_aim:10:14", events)
        self.assertIn(("mirror_seraph/echo", 14), env.imminent_threats((11, 14)))
        events.clear()
        env._resolve_mirror(env.boss, events)
        self.assertIn("mirror_lock_break:1", events)
        self.assertIn("damage:mirror_echo:14", events)
        self.assertEqual(env.player.hp, 86)
        self.assertIsNone(env.boss.echo_target)

    def test_echo_aim_is_locked_and_can_be_dodged(self):
        env = ArenaEnv(campaign_config(70))
        env.player.position = (10, 14)
        env.last_non_wait_action = "move_e"
        events = []
        env._resolve_mirror(env.boss, events)
        env.player.position = (12, 14)
        self.assertEqual(env.boss.echo_target, (10, 14))
        self.assertNotIn(("mirror_seraph/echo", 14), env.imminent_threats())
        events.clear()
        env._resolve_mirror(env.boss, events)
        self.assertIn("mirror_echo:10:14", events)
        self.assertFalse(any(event.startswith("damage:mirror_echo:") for event in events))
        self.assertEqual(env.player.hp, 100)

    def test_delayed_mirror_warning_and_lock(self):
        env = ArenaEnv(campaign_config(70))
        self.assertEqual(env.mirror_locks, {(6, 6), (14, 6), (10, 11)})
        self.assertTrue((env.mirror_locks | env.medkits | env.energy_cells) <= env._reachable_cells())
        self.assertIn("Boss mirror", encode_state(env))
        env.round = 3
        env.step("wait")
        result = env.step("move_e")
        self.assertIn("mirror_aim:move_e:w", result.events)
        self.assertIn("boss_move:10:6:11:6", result.events)
        self.assertEqual(env.mirror_ray(), ((10, 6), (9, 6), (8, 6), (7, 6), (6, 6)))
        self.assertNotIn(("mirror_seraph/shard", 18), env.imminent_threats((8, 6)))
        env.step("wait")
        result = env.step("wait")
        self.assertIn("mirror_lock_break:1", result.events)
        self.assertEqual(env.boss.broken_locks, {(6, 6)})

    def test_sidestep_does_not_shift_locked_mirror_ray(self):
        env = ArenaEnv(campaign_config(70))
        env.last_non_wait_action = "move_w"
        events = []
        env._resolve_mirror(env.boss, events)
        self.assertIn("boss_move:10:6:9:6", events)
        warned = env.mirror_ray()
        events.clear()
        env._resolve_mirror(env.boss, events)
        self.assertEqual(env.boss.position, (9, 6))
        self.assertEqual(warned[-1], (14, 6))
        self.assertFalse(any(event.startswith("boss_move:") for event in events))

    def test_used_lock_allows_danger_and_emp_echo_is_local(self):
        env = ArenaEnv(campaign_config(70))
        env.round = 3
        env.boss.broken_locks.add((6, 6))
        env.boss.copied_action = "dash_e"
        env.boss.mirrored_direction = "w"
        env.player.position = (8, 6)
        self.assertIn(("mirror_seraph/shard", 26), env.imminent_threats())
        env.step("wait")
        result = env.step("wait")
        self.assertIn("damage:mirror_shard:26", result.events)
        env.player.position = (10, 9)
        self.assertIn(Action.SHOOT_PISTOL_N, env.legal_actions())
        env.boss.copied_action = "emp"
        env.boss.mirrored_direction = None
        env.step("wait")
        result = env.step("wait")
        self.assertIn("mirror_silence", result.events)
        self.assertEqual(env.boss.silence_rounds, 1)
        self.assertNotIn(Action.SHOOT_PISTOL_N, env.legal_actions())
        env.player.position = (10, 14)
        env.player.hp = 50
        env.player.medkits = 1
        env.boss.copied_action = "heal"
        env.boss.mirrored_direction = None
        env.step("wait")
        result = env.step("wait")
        self.assertIn("mirror_heal_echo", result.events)
        self.assertEqual(env.player.medkits, 1)

    def test_clones_reflect_movement_and_share_damage(self):
        env = ArenaEnv(campaign_config(70))
        env.player.position = (9, 14)
        env.boss.hp = 100
        env.boss.exposed_rounds = 1
        events = []
        env._resolve_mirror(env.boss, events)
        clones = [enemy for enemy in env.enemies if enemy.summoned_by and
                  enemy.summoned_by.startswith("mirror_")]
        self.assertEqual(len(clones), 1)
        self.assertEqual(clones[0].position, (11, 14))
        env.player.position = (8, 14)
        env._move_mirror_clones(events)
        first = next(enemy for enemy in clones if enemy.summoned_by == "mirror_0")
        self.assertEqual(first.position, (12, 14))
        env._damage_entity(first, 10, events, "attack")
        self.assertIn("mirror_link:3", events)
        self.assertEqual(env.player.hp, 97)

    def test_mirror_wider_echo_and_clone_threats_match_damage(self):
        env = ArenaEnv(campaign_config(70))
        env.boss.echo_target = (10, 14)
        self.assertIn(("mirror_seraph/echo", 14), env.imminent_threats((11, 14)))
        self.assertIn(("mirror_seraph/echo", 14), env.imminent_threats((14, 14)))
        self.assertNotIn(("mirror_seraph/echo", 14), env.imminent_threats((12, 14)))
        env.player.position = (14, 14)
        env.boss.copied_action = "move_e"
        env.boss.mirrored_direction = "w"
        events = []
        env._resolve_mirror(env.boss, events)
        self.assertIn("damage:mirror_echo:14", events)
        env.boss.echo_target = None
        env.player.position = (8, 14)
        env.boss.hp, env.boss.exposed_rounds = 100, 1
        env._resolve_mirror(env.boss, [])
        self.assertIn(("mirror_seraph/clone", 4), env.imminent_threats((10, 14)))


class SiegeBossTests(unittest.TestCase):
    def test_charge_recoils_when_it_hits_moved_cover(self):
        env = ArenaEnv(campaign_config(80))
        env.boss.position = (8, 8)
        env.boss.charge_target = (8, 15)
        env.player.position = (9, 15)
        env.rail_covers[(7, 11)] = (8, 11)
        events = []
        env._resolve_siege(env.boss, events)
        self.assertIn("siege_charge_recoil:8", events)
        self.assertIn((8, 7), env.boss.broken_locks)

    def test_warned_diagonal_charge_hits_path_and_breaks_cover(self):
        env = ArenaEnv(campaign_config(80))
        env.player.position = (11, 14)
        env.boss.shots = 3
        events = []
        env._resolve_siege(env.boss, events)
        path = env.siege_charge_path()
        self.assertIn((11, 14), path)
        self.assertIn(("siege_leviathan/charge", 28), env.imminent_threats())
        events.clear()
        env._resolve_siege(env.boss, events)
        self.assertIn("damage:siege_charge:28", events)
        self.assertIn("boss_move:10:5:11:13", events)
        self.assertEqual(env.boss.position, (11, 13))

    def test_blast_breaks_cover_and_temporary_pit_restores(self):
        env = ArenaEnv(campaign_config(80))
        env.player.position = (10, 12)
        env.boss.blast_target = (10, 12)
        env.boss.blast_kind = "cross"
        self.assertIn(("siege_leviathan/blast", 22), env.imminent_threats((10, 14)))
        events = []
        env._resolve_siege(env.boss, events)
        self.assertIn("rail_cover_break:10:13", events)
        self.assertIn("damage:siege_blast:22", events)
        env.boss.blast_target = (9, 12)
        env.boss.blast_kind = "pit"
        env._resolve_siege(env.boss, events)
        self.assertIn((9, 12), env.siege_pits)
        self.assertIn((9, 12), env.pits)
        for _ in range(3):
            env._resolve_siege(env.boss, events)
        self.assertNotIn((9, 12), env.pits)

    def test_cover_shove_is_a_valid_model_candidate(self):
        env = ArenaEnv(campaign_config(80))
        env.step("dash_n")
        self.assertEqual(env.player.position, (10, 14))
        self.assertIn("rail cover", build_candidates(env)["shove_n"])

    def test_heavy_reposition_only_before_rail_lock(self):
        env = ArenaEnv(campaign_config(80))
        env.player.position = (7, 12)
        events = []
        env._resolve_siege(env.boss, events)
        self.assertIn("boss_move:10:5:8:7", events)
        self.assertEqual(env.boss.rail_target, 7)
        for _ in range(2):
            events.clear()
            env._resolve_siege(env.boss, events)
            self.assertEqual(env.boss.position, (8, 7))
            self.assertFalse(any(event.startswith("boss_move:") for event in events))

    def test_boss_rooms_and_supplies_are_distinct(self):
        rooms, supplies = [], []
        for level in range(10, 101, 10):
            env = ArenaEnv(campaign_config(level))
            rooms.append(frozenset(env.walls))
            supplies.append((len(env.medkits), len(env.energy_cells),
                             len(env.arrow_bundles), len(env.bow_pickups),
                             len(env.pistol_pickups)))
            items = (env.medkits | env.energy_cells | env.arrow_bundles |
                     env.bow_pickups | env.pistol_pickups)
            self.assertTrue(items <= env._reachable_cells(), level)
        self.assertEqual(len(set(rooms)), 10)
        self.assertEqual(len(set(supplies)), 10)

    def test_two_round_rail_warning_cover_and_rebuild(self):
        env = ArenaEnv(campaign_config(80))
        self.assertEqual(env.rail_locks, {(7, 7), (8, 7), (12, 7), (13, 7)})
        self.assertEqual(len(env.rail_covers), 3)
        self.assertIn("Boss siege", encode_state(env))
        env.round = 3
        env.player.position = (7, 12)
        env.step("wait")
        aim = env.step("wait")
        self.assertIn("rail_aim:v:7:2", aim.events)
        self.assertEqual(env.boss.charge, 2)
        self.assertNotIn(("siege_leviathan/railgun", 36), env.imminent_threats())
        env.step("wait")
        charge = env.step("wait")
        self.assertIn("rail_charge:v:7:1", charge.events)
        self.assertNotIn(("siege_leviathan/railgun", 36), env.imminent_threats())
        env.step("wait")
        fire = env.step("wait")
        self.assertIn("rail_lock_break:1", fire.events)
        self.assertIn((7, 11), env.rail_rebuilds)
        self.assertEqual(env.player.hp, 100)
        for _ in range(4):
            rebuild = env.step("wait")
        self.assertIn("rail_cover_rebuild:7:11", rebuild.events)
        self.assertEqual(env.rail_covers[(7, 11)], (7, 11))

    def test_uncovered_line_damages_and_cover_can_be_shoved(self):
        env = ArenaEnv(campaign_config(80))
        env.round = 3
        env.player.position = (9, 12)
        env.boss.rail_axis = "v"
        env.boss.rail_target = 9
        env.boss.charge = 1
        self.assertIn(("siege_leviathan/railgun", 36), env.imminent_threats())
        env.step("wait")
        fire = env.step("wait")
        self.assertIn("damage:railgun:36", fire.events)
        env.player.position = (7, 11)
        self.assertIn(Action.SHOVE_E, env.legal_actions())
        shove = env.step("shove_e")
        self.assertIn("rail_cover_shove:8:11:e", shove.events)
        self.assertEqual(env.rail_covers[(7, 11)], (9, 11))

    def test_default_hybrid_and_rule_finish_without_damage(self):
        for rule in (False, True):
            env, agent = ArenaEnv(campaign_config(80)), RuleAgent()
            events = []
            for _ in range(200):
                if env.done:
                    break
                action = (agent.act(env) if rule else
                          select_action({action.value: 1.0 for action in env.legal_actions()}, env, "hybrid")[0])
                events.extend(env.step(action).events)
            self.assertTrue(env.done)
            self.assertEqual(env.player.hp, 100)
            self.assertGreaterEqual(events.count("boss_shield_break"), 1)
            self.assertIn("boss_defeated", events)


class NullBossTests(unittest.TestCase):
    def test_node_can_be_activated_from_adjacent_tile(self):
        env = ArenaEnv(campaign_config(90))
        env.player.position = (5, 10)
        self.assertIn(Action.ATTACK_S, env.legal_actions())
        result = env.step(Action.ATTACK_S)
        self.assertIn("null_node_strike:5:11", result.events)
        self.assertEqual(env.boss.node_index, 1)
        self.assertEqual(env.player.position, (5, 10))

    def test_warp_approaches_player_in_both_axes(self):
        env = ArenaEnv(campaign_config(90))
        env.player.position = (11, 14)
        env._resolve_null(env.boss, [])
        self.assertEqual(env.boss.warp_target, (12, 10))
        env._resolve_null(env.boss, [])
        self.assertEqual(env.boss.position, (12, 10))

    def test_wider_fracture_and_faster_warp_cycle(self):
        env = ArenaEnv(campaign_config(90))
        env.player.position = (10, 15)
        events = []
        env._resolve_null(env.boss, events)
        self.assertEqual(env.boss.erase_targets,
                         {(8, 15), (9, 15), (10, 15), (11, 15), (12, 15),
                          (10, 16)})
        first = env.boss.warp_target
        env._resolve_null(env.boss, events)
        self.assertEqual(env.boss.position, first)
        env._resolve_null(env.boss, events)
        self.assertIsNotNone(env.boss.warp_target)

    def test_warp_warns_destination_and_changes_real_position(self):
        env = ArenaEnv(campaign_config(90))
        env.player.position = (10, 15)
        events = []
        env._resolve_null(env.boss, events)
        destination = env.boss.warp_target
        self.assertIn(f"null_warp_aim:{destination[0]}:{destination[1]}", events)
        self.assertEqual(env.boss.position, (10, 5))
        self.assertIn(f"warp={destination}", encode_state(env))
        events.clear()
        env._resolve_null(env.boss, events)
        self.assertEqual(env.boss.position, destination)
        self.assertIn("null_warp", events)
        self.assertNotIn(destination, env.walls | env.pits)

    def test_warp_cancels_when_player_occupies_warned_cell(self):
        env = ArenaEnv(campaign_config(90))
        env._resolve_null(env.boss, [])
        env.player.position = env.boss.warp_target
        events = []
        env._resolve_null(env.boss, events)
        self.assertEqual(env.boss.position, (10, 5))
        self.assertNotIn("null_warp", events)
        self.assertIsNone(env.boss.warp_target)

    def test_numbered_nodes_wrong_order_and_reverse_write(self):
        env = ArenaEnv(campaign_config(90))
        self.assertEqual(len(env.null_nodes), 4)
        env.player.position = env.null_nodes[0]
        events = []
        env._collect(events)
        self.assertEqual(env.boss.node_index, 1)
        env.player.position = env.null_nodes[2]
        env._collect(events)
        self.assertIn("null_node_reset", events)
        self.assertEqual(env.boss.node_index, 0)
        for node in env.null_nodes:
            env.player.position = node
            env._collect(events)
        self.assertIn("null_reverse_write", events)
        self.assertEqual(env.boss.exposed_rounds, 5)
        self.assertFalse(env.null_void)

    def test_action_lock_and_two_round_floor_warning(self):
        env = ArenaEnv(campaign_config(90))
        env.player.position = (10, 15)
        events = []
        env._resolve_null(env.boss, events)
        self.assertEqual(env.boss.blocked_kind, "move")
        self.assertEqual(env.boss.erase_countdown, 2)
        self.assertNotIn(Action.MOVE_W, env.legal_actions())
        self.assertIn(Action.DASH_W, env.legal_actions())
        self.assertFalse(env.null_void)
        env._resolve_null(env.boss, events)
        self.assertEqual(env.boss.erase_countdown, 1)
        self.assertFalse(env.null_void)
        env._resolve_null(env.boss, events)
        self.assertTrue(env.null_void)
        self.assertIn("null_fracture:6", events)
        env._resolve_null(env.boss, events)
        self.assertFalse(env.null_void)
        self.assertIn("null_floor_restore", events)

    def test_default_hybrid_and_rule_survive_null_boss(self):
        for rule in (False, True):
            env, agent = ArenaEnv(campaign_config(90)), RuleAgent()
            events = []
            for _ in range(200):
                if env.done:
                    break
                action = (agent.act(env) if rule else
                          select_action({action.value: 1.0 for action in env.legal_actions()}, env, "hybrid")[0])
                events.extend(env.step(action).events)
            self.assertTrue(env.done)
            self.assertGreater(env.player.hp, 0)
            self.assertGreaterEqual(events.count("boss_shield_break"), 1)
            self.assertIn("boss_defeated", events)


class ApexBossTests(unittest.TestCase):
    def test_shooting_verdict_caster_marks_active_appeal(self):
        env = ArenaEnv(campaign_config(100))
        env.round = 3
        env.boss.seals = 4
        env.boss.kind = "verdict"
        env.boss.countdown = 1
        env.boss.target = (10, 12)
        env.boss.appeal = (10, 11)
        env.boss.danger = {(10, 12)}
        env.player.position = (10, 12)
        self.assertIn(Action.SHOOT_PISTOL_N, env.legal_actions())
        mark = env.step(Action.SHOOT_PISTOL_N)
        self.assertIn("apex_appeal_mark", mark.events)
        fired = env.step(Action.WAIT)
        self.assertIn("apex_appeal", fired.events)
        self.assertIn("boss_shield_break", fired.events)
        self.assertEqual(env.player.hp, 100)

    def test_flank_and_real_charge_keep_warning_locked(self):
        env = ArenaEnv(campaign_config(100))
        events = []
        env._resolve_apex(env.boss, events)
        self.assertIn("boss_move:10:5:9:5", events)
        warned = set(env.boss.danger)
        events.clear()
        env._resolve_apex(env.boss, events)
        env._resolve_apex(env.boss, events)
        self.assertEqual(env.boss.position, (9, 5))
        self.assertTrue(warned)
        env = ArenaEnv(campaign_config(100))
        env.boss.seals = 2
        env.player.position = (10, 13)
        events = []
        env._resolve_apex(env.boss, events)
        self.assertIn((10, 11), env.boss.danger)
        env._resolve_apex(env.boss, events)
        self.assertEqual(env.boss.position, (10, 10))
        self.assertIn("boss_move:10:5:10:10", events)

    def test_finale_charge_rushes_farther_than_reflection_phase(self):
        env = ArenaEnv(campaign_config(100))
        env.boss.seals = 4
        env.boss.finale_cycles = 1
        env.player.position = (10, 13)
        events = []
        for _ in range(3):
            env._resolve_apex(env.boss, events)
        self.assertTrue(any(event.startswith("apex_fire:charge_gravity:10:13:") for event in events))
        self.assertEqual(env.boss.position, (10, 11))
        self.assertNotEqual(env.boss.position, env.player.position)

    def test_cage_warning_break_and_fire(self):
        env = ArenaEnv(campaign_config(100))
        env.round = 3
        env.player.position = (10, 15)
        env.step("wait")
        aim = env.step("wait")
        self.assertIn("apex_aim:cage:2", aim.events)
        self.assertFalse(env.apex_cage)
        env.step("wait")
        formed = env.step("wait")
        self.assertTrue(env.apex_cage)
        self.assertIn("apex_cage:10:14", formed.events)
        self.assertIn(Action.ATTACK_N, env.legal_actions())
        self.assertIn(Action.SHOOT_PISTOL_N, env.legal_actions())
        env.step("attack_n")
        finish = env.step("move_n")
        self.assertIn("apex_seal:1", finish.events)
        self.assertEqual(env.player.hp, 100)
        self.assertFalse(env.apex_cage)

    def test_barrage_charge_gravity_and_exposure(self):
        env = ArenaEnv(campaign_config(100))
        boss = env.boss
        boss.seals = 1
        env.player.position = env.apex_seals[1]
        events = []
        env._resolve_apex(boss, events)
        self.assertNotIn(env.player.position, boss.danger)
        env._resolve_apex(boss, events)
        self.assertEqual(boss.seals, 2)
        env.player.position = (10, 13)
        env._resolve_apex(boss, events)
        self.assertIn(env.apex_seals[2], boss.danger)
        env._resolve_apex(boss, events)
        self.assertEqual(boss.seals, 3)
        env.player.position = env.apex_seals[3]
        env._resolve_apex(boss, events)
        env._resolve_apex(boss, events)
        env._resolve_apex(boss, events)
        self.assertEqual(boss.seals, 4)
        self.assertEqual(boss.exposed_rounds, 6)
        self.assertIn("boss_shield_break", events)

    def test_each_attack_can_hurt_a_player_who_ignores_warning(self):
        positions = ((0, (10, 15), 26), (1, (14, 11), 16),
                     (2, (9, 13), 32), (3, (12, 13), 24))
        for seal, position, damage in positions:
            env = ArenaEnv(campaign_config(100))
            boss = env.boss
            boss.seals = seal
            env.player.position = position
            events = []
            for _ in range(3 if seal in (0, 3) else 2):
                env._resolve_apex(boss, events)
            self.assertEqual(env.player.hp, 100 - damage, (seal, events))

    def test_each_law_can_defeat_a_stationary_player(self):
        for seal, position in ((0, (10, 16)), (1, (14, 11)),
                               (2, (9, 13)), (3, (12, 13))):
            env = ArenaEnv(campaign_config(100))
            env.boss.seals = seal
            env.player.position = position
            env.round = 3
            for _ in range(32):
                if env.done:
                    break
                env.step("wait")
            self.assertTrue(env.done, seal)
            self.assertEqual(env.player.hp, 0, seal)

    def test_verdict_requires_an_active_appeal_to_reopen_core(self):
        env = ArenaEnv(campaign_config(100))
        boss = env.boss
        boss.seals = 4
        boss.finale_cycles = 2
        env.player.position = (10, 12)
        events = []
        env._resolve_apex(boss, events)
        self.assertEqual(boss.kind, "verdict")
        self.assertEqual(boss.countdown, 2)
        self.assertEqual(boss.appeal, (10, 11))
        self.assertNotIn(boss.appeal, boss.danger)
        env._resolve_apex(boss, events)
        env.player.position = boss.appeal
        env._resolve_apex(boss, events)
        self.assertIn("apex_appeal", events)
        self.assertEqual(boss.exposed_rounds, 6)
        self.assertEqual(env.player.hp, 100)

        env = ArenaEnv(campaign_config(100))
        boss = env.boss
        boss.seals = 4
        boss.finale_cycles = 2
        env.player.position = (10, 12)
        events = []
        for _ in range(3):
            env._resolve_apex(boss, events)
        self.assertIn("damage:apex_verdict:36", events)
        self.assertEqual(boss.exposed_rounds, 0)

    def test_verdict_has_a_one_step_appeal_from_every_reachable_tile(self):
        base = ArenaEnv(campaign_config(100))
        for position in base._reachable_cells() - {base.boss.position}:
            env = base.clone()
            env.player.position = position
            env.boss.seals = 4
            env.boss.finale_cycles = 2
            env._resolve_apex(env.boss, [])
            self.assertIsNotNone(env.boss.appeal, position)
            self.assertEqual(env._distance(position, env.boss.appeal), 1)
            self.assertNotIn(env.boss.appeal, env.boss.danger)

    def test_finale_combos_cap_overlap_damage_at_36(self):
        for cycle, position, source in ((0, (3, 9), "apex_cage_barrage"),
                                        (1, (10, 12), "apex_charge_gravity")):
            env = ArenaEnv(campaign_config(100))
            env.boss.seals = 4
            env.boss.finale_cycles = cycle
            env.player.position = position
            events = []
            for _ in range(3):
                env._resolve_apex(env.boss, events)
            self.assertIn(f"damage:{source}:36", events)
            self.assertEqual(env.player.hp, 64)
            self.assertEqual(env.boss.exposed_rounds, 0)

    def test_charge_path_stops_at_room_terrain(self):
        env = ArenaEnv(campaign_config(100))
        env.boss.seals = 2
        env.player.position = (6, 18)
        events = []
        env._resolve_apex(env.boss, events)
        self.assertNotIn(env.player.position, env.boss.danger)
        env._resolve_apex(env.boss, events)
        self.assertEqual(env.player.hp, 100)

    def test_finale_combos_have_walking_escape_from_every_reachable_tile(self):
        base = ArenaEnv(campaign_config(100))
        for position in base._reachable_cells() - {base.boss.position}:
            cage = base.clone()
            cage.round = 3
            cage.player.position = position
            cage.boss.seals = 4
            cage._resolve_apex(cage.boss, [])
            cage._resolve_apex(cage.boss, [])
            gate = cage.boss.gate
            direction = next((d for d in "nsew" if cage.add(position, d) == gate), None)
            self.assertIsNotNone(direction, position)
            self.assertIn(f"attack_{direction}", {a.value for a in cage.legal_actions()})
            cage.step(f"attack_{direction}")
            self.assertIn(f"move_{direction}", {a.value for a in cage.legal_actions()})
            cage.step(f"move_{direction}")
            self.assertEqual(cage.player.hp, 100, position)

            rush = base.clone()
            rush.round = 3
            rush.player.position = position
            rush.boss.seals = 4
            rush.boss.finale_cycles = 1
            rush._resolve_apex(rush.boss, [])
            rush._resolve_apex(rush.boss, [])
            safe = False
            for first in rush.legal_actions():
                if not (first.value.startswith("move_") or first == Action.WAIT):
                    continue
                after_first = rush.clone()
                after_first.step(first)
                for second in after_first.legal_actions():
                    if not (second.value.startswith("move_") or second == Action.WAIT):
                        continue
                    after_second = after_first.clone()
                    after_second.step(second)
                    if after_second.player.hp == 100:
                        safe = True
                        break
                if safe:
                    break
            self.assertTrue(safe, position)

    def test_default_hybrid_and_rule_finish(self):
        for rule in (False, True):
            env, agent = ArenaEnv(campaign_config(100)), RuleAgent()
            events = []
            for _ in range(180):
                if env.done:
                    break
                action = (agent.act(env) if rule else
                          select_action({action.value: 1.0 for action in env.legal_actions()}, env, "hybrid")[0])
                events.extend(env.step(action).events)
            self.assertTrue(env.done)
            self.assertEqual(env.player.hp, 100)
            self.assertIn("boss_defeated", events)
            self.assertEqual(events.count("boss_shield_break"), 2)


if __name__ == "__main__":
    unittest.main()
