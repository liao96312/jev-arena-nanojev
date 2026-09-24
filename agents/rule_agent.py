from arena.entities import Action
from arena.env import ArenaEnv
from arena.boss import ChronoMantis, FurnaceHydra, IronGardener, MirrorSeraph, PrismWarden, SiegeLeviathan, VoidAngler
from nanojev_adapter.policy import select_action


class RuleAgent:
    name = "rule"

    def act(self, env: ArenaEnv) -> Action:
        if isinstance(env.boss, (ChronoMantis, VoidAngler, IronGardener, MirrorSeraph, SiegeLeviathan)):
            actions = {action.value: 1.0 for action in env.legal_actions()}
            choice, _ = select_action(actions, env, "hybrid")
            return Action(choice)
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
            if isinstance(simulation.boss, FurnaceHydra):
                remaining = [x for x in (10, 7, 13) if x not in simulation.boss.valves_opened]
                head = (simulation.boss.head_x if simulation.boss.attack_kind == "wave" and
                        simulation.boss.target else remaining[0])
                value -= 1.2 * simulation._distance(simulation.player.position, (head, 12))
            elif isinstance(simulation.boss, PrismWarden):
                baits = simulation.prism_baits()
                if baits:
                    value -= .8 * min(simulation._distance(simulation.player.position, bait) for bait in baits)
                if simulation.boss.target and simulation.boss_ray() and simulation.boss_ray()[-1] in simulation.reflectors:
                    value += 2
        if env.boss and env.boss.exposed_rounds and action.value.startswith("shoot_"):
            value += 20
        if isinstance(simulation.boss, FurnaceHydra) and simulation.boss.exposed_rounds:
            value -= 2 * abs(simulation.player.position[0] - 10)
        if simulation.boss and simulation.boss.exposed_rounds:
            value -= 2 * abs(simulation.player.position[0] - simulation.boss.position[0])
        if (isinstance(env.boss, FurnaceHydra) and env.boss.target and
                env.boss.attack_kind == "fireball"):
            value += 5 * min(2, env._distance(simulation.player.position, env.boss.target))
        if simulation.player.position == env.previous_player_position:
            value -= .5
        return value
