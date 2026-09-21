from arena.env import ArenaEnv


def evaluate_state(env: ArenaEnv) -> float:
    if env.player.hp <= 0:
        return -1_000_000.0
    if env.done and env.config.finish_on_all_gems and not env.gems:
        return 1_000_000.0
    threats = sum(power for _, power in env.imminent_threats())
    mobility = len(env.legal_actions()) if not env.done else 0
    gem_distance = (min(env._distance(env.player.position, gem) for gem in env.gems)
                    if env.gems else 0)
    return (1.5 * env.player.hp + env.score + 30 * env.gems_collected + 15 * env.kills +
            10 * env.environment_kills - 2 * threats + .3 * mobility - .5 * gem_distance)
