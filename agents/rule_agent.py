from arena.entities import Action
from arena.env import ArenaEnv


class RuleAgent:
    name = "rule"

    def act(self, env: ArenaEnv) -> Action:
        return max(env.legal_actions(), key=lambda action: self._value(env, action))

    def _value(self, env: ArenaEnv, action: Action) -> float:
        if env.boss and not env.boss.exposed_rounds and action.value.startswith("shoot_"):
            return -100.0
        simulation = env.clone()
        before_hp = sum(enemy.hp for enemy in simulation.enemies)
        result = simulation.step(action)
        enemy_damage = before_hp - sum(enemy.hp for enemy in simulation.enemies)
        value = (result.reward + 1.5 * (simulation.player.hp - env.player.hp) +
                 .15 * enemy_damage + 4 * (simulation.kills - env.kills) +
                 20 * (simulation.environment_kills - env.environment_kills) +
                 4 * (simulation.gems_collected - env.gems_collected) -
                 1.2 * sum(power for _, power in simulation.imminent_threats()) +
                 .05 * len(simulation.legal_actions() if not simulation.done else ()))
        if simulation.player.hp <= 0:
            return -10_000
        targets = list(simulation.medkits) if simulation.player.hp <= 50 and simulation.medkits else list(simulation.gems)
        if targets:
            value -= .05 * min(simulation._distance(simulation.player.position, target) for target in targets)
        if simulation.boss and not simulation.boss.exposed_rounds:
            # Keep the Boss's locked ray passing through a mirror until it fires.
            lure = (10, 15)
            value -= .8 * simulation._distance(simulation.player.position, lure)
        if env.boss and env.boss.exposed_rounds and action.value.startswith("shoot_"):
            value += 20
        if simulation.player.position == env.previous_player_position:
            value -= .5
        return value
