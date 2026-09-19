import unittest

from agents import RandomAgent, RuleAgent
from arena import ArenaConfig, ArenaEnv, campaign_config
from arena.candidates import build_candidates
from arena.entities import Action, Enemy, EnemyType, IntentType


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
        self.assertGreater(first.medkits, fifth.medkits)

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

    def test_observation_and_candidates_share_immediate_threats(self):
        from arena.observation import encode_state

        env = ArenaEnv(ArenaConfig(width=5, height=5, walls=0, enemies=0, gems=0, fires=0,
                                   medkits=0, charger_ratio=0, bomber_ratio=0))
        env.player.position = (2, 2)
        bomber = Enemy((2, 1), enemy_type=EnemyType.BOMBER)
        env.enemies = [bomber]
        env._plan_enemy_intents()
        bomber.intent.countdown = 1
        self.assertIn("Immediate threats: bomber", encode_state(env))
        self.assertIn("hp 100->80", build_candidates(env)["wait"])


if __name__ == "__main__":
    unittest.main()
