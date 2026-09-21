import unittest

from agents import RandomAgent, RuleAgent
from arena import ArenaConfig, ArenaEnv, campaign_config
from arena.candidates import build_candidates
from arena.entities import Action, ENEMY_HP, Enemy, EnemyType, Intent, IntentType, PlayerLoadout


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
        self.assertIn("gem 2->1", build_candidates(env)["move_e"])

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
        self.assertLess(first.spikes, fifth.spikes)
        self.assertLess(first.pits, fifth.pits)
        self.assertGreater(first.medkits, fifth.medkits)
        self.assertLess(first.enemy_hp_bonus, fifth.enemy_hp_bonus)
        self.assertLess(first.enemy_damage, fifth.enemy_damage)
        self.assertEqual((first.action_points, fifth.action_points), (2, 2))

        configs = [campaign_config(level) for level in range(1, 31)]
        self.assertEqual([config.enemy_hp_bonus for config in configs], list(range(30)))

    def test_campaign_hp_bonus_is_applied_to_spawned_enemies(self):
        env = ArenaEnv(campaign_config(5))
        self.assertTrue(env.enemies)
        self.assertTrue(all(enemy.max_hp - enemy.hp == 0 for enemy in env.enemies))
        self.assertTrue(all(enemy.max_hp == ENEMY_HP[enemy.enemy_type] + 4 for enemy in env.enemies))

    def test_collecting_last_campaign_gem_completes_level(self):
        env = ArenaEnv(ArenaConfig(width=3, height=3, walls=0, enemies=0, gems=1, fires=0,
                                   medkits=0, finish_on_all_gems=True))
        env.player.position, env.gems = (1, 1), {(2, 1)}
        result = env.step(Action.MOVE_E)
        self.assertTrue(result.done)
        self.assertIn("level_complete", result.events)

    def test_enemy_intent_is_telegraphed_and_deterministic(self):
        env = ArenaEnv(ArenaConfig(width=5, height=5, walls=0, enemies=0, gems=0, fires=0,
                                   medkits=0, enemy_move_interval=2))
        env.player.position = (2, 2)
        env.enemies = [Enemy((2, 4))]
        env._plan_enemy_intents()
        self.assertEqual((env.enemies[0].intent.kind, env.enemies[0].intent.direction,
                          env.enemies[0].intent.countdown), (IntentType.MOVE, "n", 2))
        env.step(Action.WAIT)
        self.assertEqual(env.enemies[0].position, (2, 4))
        env.step(Action.WAIT)
        self.assertEqual(env.enemies[0].position, (2, 3))
        self.assertEqual(env.enemies[0].intent.kind, IntentType.MELEE)
        env.step(Action.WAIT)
        result = env.step(Action.WAIT)
        self.assertEqual(env.player.hp, 95)
        self.assertIn("damage:enemy:5", result.events)

    def test_two_ap_delays_enemy_resolution_until_round_end(self):
        env = ArenaEnv(ArenaConfig(width=5, height=5, walls=0, enemies=0, gems=0, fires=0,
                                   medkits=0, enemy_move_interval=1, action_points=2))
        env.player.position = (2, 2)
        env.enemies = [Enemy((2, 4))]
        env._plan_enemy_intents()

        first = env.step(Action.WAIT)
        self.assertEqual((env.enemies[0].position, env.round, env.ap_remaining), ((2, 4), 1, 1))
        self.assertNotIn("round_end", first.events)

        second = env.step(Action.WAIT)
        self.assertEqual((env.enemies[0].position, env.round, env.ap_remaining), ((2, 3), 2, 2))
        self.assertIn("round_end", second.events)

    def test_heal_wait_and_configured_skill_ap_costs(self):
        env = ArenaEnv(ArenaConfig(width=6, height=5, walls=0, enemies=0, gems=0, fires=0,
                                   medkits=0, action_points=2, dash_ap_cost=2, emp_ap_cost=2))
        env.player.position, env.player.hp, env.player.medkits = (2, 2), 50, 1
        env.enemies = [Enemy((3, 2))]
        env._plan_enemy_intents()
        env.step(Action.HEAL)
        self.assertEqual((env.player.hp, env.ap_remaining), (85, 1))
        self.assertFalse(any(action.value.startswith("dash_") for action in env.legal_actions()))
        self.assertNotIn(Action.EMP, env.legal_actions())
        env.step(Action.WAIT)
        self.assertEqual((env.round, env.ap_remaining), (2, 2))

        from arena.observation import encode_state
        self.assertIn("r=2 ap=2/2", encode_state(env))

    def test_every_enemy_type_routes_around_wall(self):
        config = ArenaConfig(width=6, height=5, walls=0, enemies=0, gems=0, fires=0,
                             medkits=0, enemy_move_interval=1)
        for enemy_type in EnemyType:
            with self.subTest(enemy_type=enemy_type):
                env = ArenaEnv(config)
                env.player.position = (4, 2)
                env.walls = {(2, 2), (3, 2)}
                env.enemies = [Enemy((1, 2), enemy_type=enemy_type)]
                env._plan_enemy_intents()
                self.assertEqual((env.enemies[0].intent.kind, env.enemies[0].intent.direction),
                                 (IntentType.MOVE, "n"))

    def test_enemies_route_around_fire_instead_of_entering_it(self):
        config = ArenaConfig(width=6, height=5, walls=0, enemies=0, gems=0, fires=0,
                             medkits=0, enemy_move_interval=1)
        for enemy_type in EnemyType:
            with self.subTest(enemy_type=enemy_type):
                env = ArenaEnv(config)
                env.player.position = (4, 3)
                env.fires = {(2, 2)}
                env.enemies = [Enemy((1, 2), enemy_type=enemy_type)]
                env._plan_enemy_intents()
                self.assertEqual(env.enemies[0].intent.kind, IntentType.MOVE)
                self.assertNotIn(env.add(env.enemies[0].position, env.enemies[0].intent.direction),
                                 env.fires)

        env = ArenaEnv(config)
        env.player.position, env.fires = (4, 2), {(2, 2)}
        env.enemies = [Enemy((1, 2), enemy_type=EnemyType.CHARGER)]
        env._plan_enemy_intents()
        self.assertEqual(env.enemies[0].intent.kind, IntentType.MOVE)

    def test_player_can_dodge_visible_melee_intent(self):
        env = ArenaEnv(ArenaConfig(width=5, height=5, walls=0, enemies=0, gems=0, fires=0,
                                   medkits=0))
        env.player.position = (2, 2)
        env.enemies = [Enemy((3, 2))]
        env._plan_enemy_intents()
        self.assertEqual(env.enemies[0].intent.kind, IntentType.MELEE)
        result = env.step(Action.MOVE_N)
        self.assertEqual(env.player.hp, 100)
        self.assertNotIn("enemy_melee", result.events)

    def test_charger_hits_another_enemy(self):
        env = ArenaEnv(ArenaConfig(width=8, height=3, walls=0, enemies=0, gems=0, fires=0,
                                   medkits=0, charger_ratio=1, bomber_ratio=0))
        env.player.position = (6, 1)
        charger = Enemy((1, 1), enemy_type=EnemyType.CHARGER)
        victim = Enemy((4, 1))
        env.enemies = [charger, victim]
        env._plan_enemy_intents()
        self.assertEqual((charger.intent.kind, charger.intent.direction), (IntentType.CHARGE, "e"))
        result = env.step(Action.WAIT)
        self.assertEqual(charger.position, (3, 1))
        self.assertEqual(victim.hp, 15)
        self.assertIn("enemy_collision:15", result.events)

    def test_charger_mix_is_seed_deterministic(self):
        config = ArenaConfig(enemies=20, charger_ratio=0.5)
        first, second = ArenaEnv(config), ArenaEnv(config)
        first.reset(42)
        second.reset(42)
        self.assertEqual([enemy.enemy_type for enemy in first.enemies],
                         [enemy.enemy_type for enemy in second.enemies])
        self.assertIn(EnemyType.CHARGER, [enemy.enemy_type for enemy in first.enemies])

    def test_shove_enemy_into_fire(self):
        env = ArenaEnv(ArenaConfig(width=5, height=5, walls=0, enemies=0, gems=0, fires=0,
                                   medkits=0, fire_damage=10))
        env.player.position = (3, 2)
        enemy = Enemy((2, 2), hp=10)
        env.enemies = [enemy]
        env.fires = {(1, 2)}
        env._plan_enemy_intents()
        self.assertIn(Action.SHOVE_W, env.legal_actions())
        self.assertIn("into fire", build_candidates(env)["shove_w"])
        result = env.step(Action.SHOVE_W)
        self.assertNotIn(enemy, env.enemies)
        self.assertEqual((env.kills, env.environment_kills), (1, 1))
        self.assertIn("environment_kill", result.events)
        self.assertEqual(result.reward, 10.05)

    def test_spike_damages_player_and_pushed_enemy(self):
        env = ArenaEnv(ArenaConfig(width=5, height=5, walls=0, enemies=0, gems=0, fires=0,
                                   spikes=0, medkits=0, spike_damage=12))
        env.player.position = (1, 2)
        env.spikes = {(2, 2), (4, 2)}
        result = env.step(Action.MOVE_E)
        self.assertEqual(env.player.hp, 88)
        self.assertIn("damage:spike:12", result.events)

        enemy = Enemy((3, 2), hp=12, stunned=2)
        env.enemies = [enemy]
        env._plan_enemy_intents()
        result = env.step(Action.SHOVE_E)
        self.assertNotIn(enemy, env.enemies)
        self.assertIn("environment_kill", result.events)

    def test_pit_blocks_movement_but_shove_is_instant_kill(self):
        env = ArenaEnv(ArenaConfig(width=6, height=5, walls=0, enemies=0, gems=0, fires=0,
                                   pits=0, medkits=0))
        env.player.position = (1, 2)
        env.pits = {(2, 2), (4, 2)}
        self.assertNotIn(Action.MOVE_E, env.legal_actions())
        self.assertNotIn(Action.DASH_E, env.legal_actions())

        env.player.position = (2, 2)
        enemy = Enemy((3, 2), stunned=2)
        env.enemies = [enemy]
        env._plan_enemy_intents()
        self.assertIn("into pit; instant kill", build_candidates(env)["shove_e"])
        result = env.step(Action.SHOVE_E)
        self.assertNotIn(enemy, env.enemies)
        self.assertEqual((env.kills, env.environment_kills), (1, 1))
        self.assertIn("pit_fall", result.events)

    def test_charger_falls_into_pit_and_enemies_avoid_hazards(self):
        env = ArenaEnv(ArenaConfig(width=8, height=5, walls=0, enemies=0, gems=0, fires=0,
                                   spikes=0, pits=0, medkits=0, enemy_move_interval=1))
        env.player.position = (6, 2)
        charger = Enemy((1, 2), enemy_type=EnemyType.CHARGER)
        env.enemies = [charger]
        env.pits = {(3, 2)}
        env._plan_enemy_intents()
        self.assertEqual(charger.intent.kind, IntentType.CHARGE)
        charger.intent.countdown = 1
        self.assertFalse(env.imminent_threats())
        result = env.step(Action.WAIT)
        self.assertNotIn(charger, env.enemies)
        self.assertIn("pit_fall", result.events)

        env.player.position = (5, 3)
        env.enemies = [Enemy((1, 2))]
        env.pits, env.spikes = {(2, 2)}, {(3, 2)}
        env._plan_enemy_intents()
        target = env.add(env.enemies[0].position, env.enemies[0].intent.direction)
        self.assertNotIn(target, env.pits | env.spikes)

    def test_shove_collision_damages_both_enemies(self):
        env = ArenaEnv(ArenaConfig(width=5, height=5, walls=0, enemies=0, gems=0, fires=0,
                                   medkits=0, collision_damage=5))
        env.player.position = (1, 2)
        pushed, blocker = Enemy((2, 2)), Enemy((3, 2))
        env.enemies = [pushed, blocker]
        env._plan_enemy_intents()
        env.step(Action.SHOVE_E)
        self.assertEqual((pushed.hp, blocker.hp), (25, 25))

    def test_bomber_explosion_hits_player_and_enemy(self):
        env = ArenaEnv(ArenaConfig(width=5, height=5, walls=0, enemies=0, gems=0, fires=0,
                                   medkits=0, bomber_damage=20))
        env.player.position = (2, 3)
        bomber = Enemy((2, 2), enemy_type=EnemyType.BOMBER)
        victim = Enemy((3, 2), hp=20, stunned=2)
        env.enemies = [bomber, victim]
        env._plan_enemy_intents()
        self.assertEqual((bomber.intent.kind, bomber.intent.countdown), (IntentType.EXPLODE, 2))
        env.step(Action.WAIT)
        result = env.step(Action.WAIT)
        self.assertEqual(env.player.hp, 80)
        self.assertNotIn(bomber, env.enemies)
        self.assertNotIn(victim, env.enemies)
        self.assertEqual(env.environment_kills, 1)
        self.assertIn("bomber_explode", result.events)

    def test_archer_shot_is_telegraphed_and_dodgeable(self):
        env = ArenaEnv(ArenaConfig(width=7, height=5, walls=0, enemies=0, gems=0, fires=0,
                                   medkits=0, charger_ratio=0, bomber_ratio=0, archer_ratio=0))
        env.player.position = (5, 2)
        archer = Enemy((1, 2), enemy_type=EnemyType.ARCHER)
        env.enemies = [archer]
        env._plan_enemy_intents()
        self.assertEqual((archer.intent.kind, archer.intent.direction, archer.intent.countdown),
                         (IntentType.SHOOT, "e", 2))
        env.step(Action.WAIT)
        self.assertIn("archer@", env.imminent_threats()[0][0])
        self.assertIn("hp 100->88", build_candidates(env)["wait"])
        result = env.step(Action.MOVE_N)
        self.assertEqual(env.player.hp, 100)
        self.assertIn("shot_blocked", result.events)
        self.assertIn("archer_shot:e:1:2:7:2", result.events)

    def test_campaign_archer_cannot_target_player_during_spawn_protection(self):
        env = ArenaEnv(campaign_config(3))
        env.player.position = (5, 2)
        archer = Enemy((1, 2), enemy_type=EnemyType.ARCHER)
        env.enemies = [archer]
        env.round = 1
        env._plan_enemy_intents()
        self.assertNotEqual(archer.intent.kind, IntentType.SHOOT)
        env.round = 3
        env._plan_enemy_intents()
        self.assertEqual(archer.intent.kind, IntentType.SHOOT)

    def test_archer_shot_hits_first_enemy(self):
        env = ArenaEnv(ArenaConfig(width=8, height=3, walls=0, enemies=0, gems=0, fires=0,
                                   medkits=0, charger_ratio=0, bomber_ratio=0, archer_ratio=0,
                                   archer_damage=12))
        env.player.position = (6, 1)
        archer = Enemy((1, 1), enemy_type=EnemyType.ARCHER)
        victim = Enemy((3, 1))
        env.enemies = [archer, victim]
        env._plan_enemy_intents()
        env.step(Action.WAIT)
        result = env.step(Action.WAIT)
        self.assertEqual(victim.hp, 18)
        self.assertEqual(env.player.hp, 100)
        self.assertIn("archer_friendly_fire", result.events)

    def test_archer_mix_is_seed_deterministic(self):
        config = ArenaConfig(enemies=12, charger_ratio=0, bomber_ratio=0, archer_ratio=1)
        first, second = ArenaEnv(config), ArenaEnv(config)
        first.reset(42)
        second.reset(42)
        self.assertEqual([enemy.enemy_type for enemy in first.enemies],
                         [enemy.enemy_type for enemy in second.enemies])
        self.assertTrue(all(enemy.enemy_type == EnemyType.ARCHER for enemy in first.enemies))

    def test_enemy_move_speeds_match_roles(self):
        env = ArenaEnv(ArenaConfig(width=9, height=7, walls=0, enemies=0, gems=0, fires=0,
                                   medkits=0, charger_ratio=0, bomber_ratio=0,
                                   enemy_move_interval=2))
        env.player.position = (7, 5)
        env.enemies = [Enemy((1, 5), enemy_type=EnemyType.CHASER),
                       Enemy((1, 1), enemy_type=EnemyType.CHARGER),
                       Enemy((2, 1), enemy_type=EnemyType.BOMBER),
                       Enemy((3, 1), enemy_type=EnemyType.ARCHER)]
        env._plan_enemy_intents()
        self.assertEqual([enemy.intent.countdown for enemy in env.enemies], [2, 3, 4, 4])
        env.enemies[1].position = (7, 1)
        env._plan_enemy_intents([env.enemies[1]])
        self.assertEqual((env.enemies[1].intent.kind, env.enemies[1].intent.countdown),
                         (IntentType.CHARGE, 2))

    def test_campaign_enemy_speed_tiers(self):
        expected = {
            1: [2, 3, 4, 4],
            6: [1, 3, 4, 4],
            12: [1, 2, 4, 4],
            18: [1, 2, 3, 4],
            24: [1, 1, 3, 3],
            30: [1, 1, 2, 3],
        }
        types = (EnemyType.CHASER, EnemyType.CHARGER, EnemyType.ARCHER, EnemyType.BOMBER)
        for level, intervals in expected.items():
            env = ArenaEnv(campaign_config(level))
            self.assertEqual([env._enemy_move_interval(Enemy((0, 0), enemy_type=kind)) for kind in types],
                             intervals)

        charger_env = ArenaEnv(campaign_config(12))
        charger_env.player.position = (6, 2)
        charger = Enemy((2, 2), enemy_type=EnemyType.CHARGER)
        charger_env.enemies = [charger]
        charger_env._plan_enemy_intents()
        self.assertEqual((charger.intent.kind, charger.intent.countdown), (IntentType.CHARGE, 1))

        archer_env = ArenaEnv(campaign_config(18))
        archer_env.player.position, archer_env.round = (6, 2), 3
        archer = Enemy((2, 2), enemy_type=EnemyType.ARCHER)
        archer_env.enemies = [archer]
        archer_env._plan_enemy_intents()
        self.assertEqual((archer.intent.kind, archer.intent.countdown), (IntentType.SHOOT, 1))

    def test_enemy_types_have_independent_hp_damage_and_cooldown(self):
        env = ArenaEnv(ArenaConfig(width=7, height=7, walls=0, enemies=0, gems=0, fires=0,
                                   medkits=0, enemy_damage=5, enemy_move_interval=2))
        expected = {
            EnemyType.CHASER: (30, 5, 2),
            EnemyType.CHARGER: (45, 8, 3),
            EnemyType.BOMBER: (24, 20, 2),
            EnemyType.ARCHER: (20, 4, 4),
        }
        for enemy_type, (hp, damage, cooldown) in expected.items():
            enemy = Enemy((3, 2), enemy_type=enemy_type)
            env.player.position, env.enemies = (3, 3), [enemy]
            env._plan_enemy_intents()
            self.assertEqual((enemy.hp, enemy.max_hp, enemy.intent.power, enemy.intent.countdown),
                             (hp, hp, damage, cooldown))

    def test_bow_pushes_into_fire_but_pistol_does_not(self):
        config = ArenaConfig(width=8, height=3, walls=0, enemies=0, gems=0, fires=0,
                             medkits=0, charger_ratio=0, bomber_ratio=0, fire_damage=15)
        bow_env = ArenaEnv(config, PlayerLoadout(bow=True, arrows=2))
        bow_env.player.position = (1, 1)
        bow_env.enemies = [Enemy((3, 1), hp=30, stunned=2)]
        bow_env.fires = {(4, 1)}
        bow_env._plan_enemy_intents()
        self.assertIn(Action.SHOOT_BOW_E, bow_env.legal_actions())
        result = bow_env.step(Action.SHOOT_BOW_E)
        self.assertFalse(bow_env.enemies)
        self.assertEqual((bow_env.player.loadout.arrows, bow_env.environment_kills), (1, 1))
        self.assertIn("environment_kill", result.events)

        pistol_env = ArenaEnv(config, PlayerLoadout(pistol=True, energy=2))
        pistol_env.player.position = (1, 1)
        pistol_env.enemies = [Enemy((3, 1), hp=30, stunned=2)]
        pistol_env.fires = {(4, 1)}
        pistol_env._plan_enemy_intents()
        result = pistol_env.step(Action.SHOOT_PISTOL_E)
        self.assertEqual((pistol_env.enemies[0].position, pistol_env.enemies[0].hp), ((3, 1), 18))
        self.assertEqual(pistol_env.player.loadout.energy, 1)
        self.assertIn("shoot_pistol:e:2", result.events)

    def test_weapon_pickup_and_loadout_survive_next_level(self):
        env = ArenaEnv(ArenaConfig(width=5, height=5, walls=0, enemies=0, gems=0, fires=0,
                                   medkits=0))
        env.player.position = (2, 2)
        env.bow_pickups = {(3, 2)}
        env.step(Action.MOVE_E)
        self.assertEqual((env.player.loadout.bow, env.player.loadout.arrows), (True, 3))
        next_level = ArenaEnv(campaign_config(2), env.player.loadout)
        self.assertEqual((next_level.player.loadout.bow, next_level.player.loadout.arrows), (True, 3))

    def test_campaign_weapon_pickups_are_reachable(self):
        for level in (1, 2, 3, 4):
            env = ArenaEnv(campaign_config(level))
            reachable = {env.player.position}
            frontier = [env.player.position]
            while frontier:
                position = frontier.pop()
                for direction in ("n", "s", "w", "e"):
                    target = env.add(position, direction)
                    if env.in_bounds(target) and target not in env.walls and target not in reachable:
                        reachable.add(target)
                        frontier.append(target)
            pickups = env.bow_pickups | env.pistol_pickups | env.arrow_bundles | env.energy_cells
            self.assertTrue(pickups <= reachable)

    def test_campaign_maps_have_two_exits_and_reachable_gems(self):
        for level in (1, 5, 10, 13, 20):
            env = ArenaEnv(campaign_config(level))
            for seed in range(30):
                env.reset(seed)
                exits = sum(env.in_bounds(env.add(env.player.position, direction)) and
                            env.add(env.player.position, direction) not in env.walls and
                            env.add(env.player.position, direction) not in env.barrels and
                            not env.enemy_at(env.add(env.player.position, direction))
                            for direction in ("n", "s", "w", "e"))
                self.assertGreaterEqual(exits, 2)
                self.assertTrue(env.gems <= env._reachable_cells())
                self.assertTrue(env.medkits <= env._reachable_cells())

    def test_campaign_keeps_missing_weapon_pickups_available(self):
        self.assertEqual(campaign_config(13).bow_pickups, 1)
        self.assertEqual(campaign_config(13).pistol_pickups, 1)

    def test_weapon_pickup_candidate_reports_acquisition(self):
        env = ArenaEnv(ArenaConfig(width=5, height=5, walls=0, enemies=0, gems=0, fires=0,
                                   medkits=0), PlayerLoadout(arrows=12, energy=12))
        env.player.position = (2, 2)
        env.bow_pickups = {(3, 2)}
        self.assertIn("acquire_bow", build_candidates(env)["move_e"])

    def test_ranged_candidates_require_visible_target(self):
        env = ArenaEnv(ArenaConfig(width=10, height=3, walls=0, enemies=0, gems=0, fires=0,
                                   medkits=0), PlayerLoadout(True, True, 3, 6))
        env.player.position = (1, 1)
        env.enemies = [Enemy((8, 1))]
        env._plan_enemy_intents()
        self.assertNotIn(Action.SHOOT_BOW_E, env.legal_actions())
        self.assertIn(Action.SHOOT_PISTOL_E, env.legal_actions())
        env.walls = {(5, 1)}
        self.assertNotIn(Action.SHOOT_PISTOL_E, env.legal_actions())

    def test_weapon_candidates_stay_within_model_budget(self):
        env = ArenaEnv(ArenaConfig(width=11, height=11, walls=0, enemies=0, gems=0, fires=0,
                                   medkits=0), PlayerLoadout(True, True, 12, 24))
        env.player.position = (5, 5)
        env.enemies = [Enemy((5, 2)), Enemy((8, 5)), Enemy((5, 8)), Enemy((2, 5))]
        env._plan_enemy_intents()
        self.assertLessEqual(len(env.legal_actions()), 12)

    def test_barrels_chain_and_damage_nearby_enemy(self):
        env = ArenaEnv(ArenaConfig(width=7, height=3, walls=0, enemies=0, gems=0, fires=0,
                                   medkits=0, barrel_damage=20))
        env.player.position = (1, 1)
        env.barrels = {(2, 1), (3, 1)}
        victim = Enemy((4, 1), hp=20, stunned=2)
        env.enemies = [victim]
        env._plan_enemy_intents()
        result = env.step(Action.ATTACK_E)
        self.assertFalse(env.barrels)
        self.assertNotIn(victim, env.enemies)
        self.assertEqual(result.events.count("barrel_explode"), 2)
        self.assertEqual(env.environment_kills, 1)

    def test_pistol_can_detonate_barrel_before_enemy(self):
        env = ArenaEnv(ArenaConfig(width=8, height=3, walls=0, enemies=0, gems=0, fires=0,
                                   medkits=0), PlayerLoadout(pistol=True, energy=2))
        env.player.position = (1, 1)
        env.barrels = {(4, 1)}
        victim = Enemy((5, 1), hp=20, stunned=2)
        env.enemies = [victim]
        env._plan_enemy_intents()
        self.assertIn(Action.SHOOT_PISTOL_E, env.legal_actions())
        self.assertIn("barrel/3 blast", build_candidates(env)["shoot_pistol_e"])
        env.step(Action.SHOOT_PISTOL_E)
        self.assertNotIn(victim, env.enemies)
        self.assertEqual(env.player.loadout.energy, 1)

    def test_dash_moves_two_cells_and_ticks_cooldown(self):
        env = ArenaEnv(ArenaConfig(width=6, height=5, walls=0, enemies=0, gems=0, fires=0,
                                   medkits=0))
        env.player.position = (1, 2)
        self.assertIn(Action.DASH_E, env.legal_actions())
        self.assertIn("cd=3", build_candidates(env)["dash_e"])
        env.step(Action.DASH_E)
        self.assertEqual((env.player.position, env.player.cooldowns["dash"]), ((3, 2), 3))
        self.assertNotIn(Action.DASH_E, env.legal_actions())
        for expected in (2, 1, 0):
            env.step(Action.WAIT)
            self.assertEqual(env.player.cooldowns["dash"], expected)
        self.assertIn(Action.DASH_E, env.legal_actions())

    def test_dash_cannot_cross_enemy_or_wall(self):
        env = ArenaEnv(ArenaConfig(width=6, height=5, walls=0, enemies=0, gems=0, fires=0,
                                   medkits=0))
        env.player.position = (1, 2)
        env.walls = {(2, 2)}
        self.assertNotIn(Action.DASH_E, env.legal_actions())

    def test_dash_iframes_ignore_hazards_and_same_action_enemy_hit(self):
        env = ArenaEnv(ArenaConfig(width=8, height=5, walls=0, enemies=0, gems=0, fires=0,
                                   spikes=0, medkits=0, action_points=1, dash_ap_cost=1))
        env.player.position = (1, 2)
        env.fires, env.spikes = {(2, 2)}, {(3, 2)}
        archer = Enemy((6, 2), enemy_type=EnemyType.ARCHER)
        archer.intent = Intent(IntentType.SHOOT, "w", 1, 12)
        env.enemies = [archer]
        result = env.step(Action.DASH_E)
        self.assertEqual((env.player.position, env.player.hp, env.player.invulnerable), ((3, 2), 100, False))
        self.assertIn("invulnerable:fire", result.events)
        self.assertIn("invulnerable:spike", result.events)
        self.assertIn("invulnerable:shot", result.events)

    def test_rule_uses_environmental_shove(self):
        env = ArenaEnv(ArenaConfig(width=5, height=5, walls=0, enemies=0, gems=0, fires=0,
                                   medkits=0, charger_ratio=0, bomber_ratio=0))
        env.player.position = (3, 2)
        env.enemies = [Enemy((2, 2), hp=10)]
        env.fires = {(1, 2)}
        env._plan_enemy_intents()
        self.assertEqual(RuleAgent().act(env), Action.SHOVE_W)

    def test_rule_dashes_out_of_overlapping_blasts(self):
        env = ArenaEnv(ArenaConfig(width=5, height=5, walls=0, enemies=0, gems=0, fires=0,
                                   medkits=0, bomber_radius=2, charger_ratio=0, bomber_ratio=0))
        env.player.position = (2, 2)
        env.enemies = [Enemy((2, 1), enemy_type=EnemyType.BOMBER),
                       Enemy((1, 2), enemy_type=EnemyType.BOMBER)]
        env._plan_enemy_intents()
        self.assertEqual(RuleAgent().act(env), Action.DASH_S)

    def test_emp_interrupts_immediate_enemy_action_and_cools_down(self):
        env = ArenaEnv(ArenaConfig(width=5, height=5, walls=0, enemies=0, gems=0, fires=0,
                                   medkits=0, charger_ratio=0, bomber_ratio=0))
        env.player.position = (2, 2)
        env.enemies = [Enemy((2, 1))]
        env._plan_enemy_intents()
        self.assertIn(Action.EMP, env.legal_actions())
        self.assertIn("cd=4", build_candidates(env)["emp"])
        result = env.step(Action.EMP)
        self.assertEqual(env.player.hp, 100)
        self.assertIn("emp:1", result.events)
        self.assertEqual(env.player.cooldowns["emp"], 4)
        self.assertNotIn(Action.EMP, env.legal_actions())
        env.step(Action.WAIT)
        self.assertEqual(env.player.cooldowns["emp"], 3)

    def test_emp_delays_bomber_explosion(self):
        env = ArenaEnv(ArenaConfig(width=5, height=5, walls=0, enemies=0, gems=0, fires=0,
                                   medkits=0, charger_ratio=0, bomber_ratio=0))
        env.player.position = (2, 2)
        bomber = Enemy((2, 1), enemy_type=EnemyType.BOMBER)
        env.enemies = [bomber]
        env._plan_enemy_intents()
        env.step(Action.EMP)
        result = env.step(Action.WAIT)
        self.assertIn(bomber, env.enemies)
        self.assertNotIn("bomber_explode", result.events)

    def test_rule_uses_emp_when_surrounded_by_blasts(self):
        env = ArenaEnv(ArenaConfig(width=3, height=3, walls=0, enemies=0, gems=0, fires=0,
                                   medkits=0, charger_ratio=0, bomber_ratio=0))
        env.player.position = (1, 1)
        env.enemies = [Enemy(position, enemy_type=EnemyType.BOMBER)
                       for position in ((1, 0), (1, 2), (0, 1), (2, 1))]
        env._plan_enemy_intents()
        for enemy in env.enemies:
            enemy.intent.countdown = 1
        self.assertEqual(RuleAgent().act(env), Action.EMP)

    def test_observation_and_candidates_share_immediate_threats(self):
        from arena.observation import encode_state

        env = ArenaEnv(ArenaConfig(width=5, height=5, walls=0, enemies=0, gems=0, fires=0,
                                   medkits=0, charger_ratio=0, bomber_ratio=0))
        env.player.position = (2, 2)
        bomber = Enemy((2, 1), enemy_type=EnemyType.BOMBER)
        env.enemies = [bomber]
        env._plan_enemy_intents()
        bomber.intent.countdown = 1
        self.assertIn("Threat bomber", encode_state(env))
        self.assertIn("hp 100->80", build_candidates(env)["wait"])


if __name__ == "__main__":
    unittest.main()
