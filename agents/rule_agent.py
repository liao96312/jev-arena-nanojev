from arena.entities import Action, IntentType
from arena.env import ArenaEnv


class RuleAgent:
    name = "rule"

    def act(self, env: ArenaEnv) -> Action:
        return max(env.legal_actions(), key=lambda action: self._value(env, action))

    def _value(self, env: ArenaEnv, action: Action) -> float:
        simulation = env.clone()
        before_hp = sum(enemy.hp for enemy in simulation.enemies)
        result = simulation.step(action)
        enemy_damage = before_hp - sum(enemy.hp for enemy in simulation.enemies)
        value = (result.reward + 1.5 * (simulation.player.hp - env.player.hp) +
                 .15 * enemy_damage + 4 * (simulation.kills - env.kills) +
                 20 * (simulation.environment_kills - env.environment_kills) +
                 4 * (simulation.gems_collected - env.gems_collected) -
                 1.2 * self._imminent_damage(simulation) +
                 .05 * len(simulation.legal_actions() if not simulation.done else ()))
        if simulation.player.hp <= 0:
            return -10_000
        targets = list(simulation.medkits) if simulation.player.hp <= 50 and simulation.medkits else list(simulation.gems)
        if targets:
            value -= .05 * min(simulation._distance(simulation.player.position, target) for target in targets)
        if simulation.player.position == env.previous_player_position:
            value -= .5
        return value

    @staticmethod
    def _imminent_damage(env: ArenaEnv) -> int:
        damage = 0
        for enemy in env.enemies:
            intent = enemy.intent
            if not intent or intent.countdown > 1:
                continue
            if intent.kind == IntentType.EXPLODE:
                if env._distance(enemy.position, env.player.position) <= env.config.bomber_radius:
                    damage += intent.power
            elif intent.kind == IntentType.MELEE and intent.direction:
                if env.add(enemy.position, intent.direction) == env.player.position:
                    damage += intent.power
            elif intent.kind == IntentType.CHARGE and intent.direction:
                position = enemy.position
                for _ in range(env.config.charger_range):
                    position = env.add(position, intent.direction)
                    if not env.in_bounds(position) or position in env.walls:
                        break
                    if position == env.player.position:
                        damage += intent.power
                        break
                    if env.enemy_at(position):
                        break
        return damage
