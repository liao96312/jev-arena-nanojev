from __future__ import annotations

from .env import ArenaEnv


def _direction(origin: tuple[int, int], target: tuple[int, int]) -> str:
    dx, dy = target[0] - origin[0], target[1] - origin[1]
    vertical = "N" if dy < 0 else "S" if dy > 0 else ""
    horizontal = "W" if dx < 0 else "E" if dx > 0 else ""
    return vertical + horizontal or "HERE"


def _nearest(origin: tuple[int, int], positions: list[tuple[int, int]]) -> str:
    if not positions:
        return "none"
    target = min(positions, key=lambda p: abs(p[0] - origin[0]) + abs(p[1] - origin[1]))
    distance = abs(target[0] - origin[0]) + abs(target[1] - origin[1])
    return f"{_direction(origin, target)}/{distance}"


def encode_state(env: ArenaEnv) -> str:
    adjacent = []
    for name, short in (("north", "n"), ("south", "s"), ("west", "w"), ("east", "e")):
        target = env.add(env.player.position, short)
        enemy = env.enemy_at(target)
        if not env.in_bounds(target) or target in env.walls:
            value = "wall"
        elif enemy:
            value = f"enemy{enemy.hp}"
        elif target in env.fires:
            value = "fire"
        else:
            value = "safe"
        adjacent.append(f"{name[0].upper()}:{value}")
    enemy_positions = [enemy.position for enemy in env.enemies]
    tactical_enemies = sorted(env.enemies, key=lambda enemy: env._distance(env.player.position, enemy.position))[:3]
    intents = "; ".join(
        f"{enemy.enemy_type.value}:{_direction(env.player.position, enemy.position)}/"
        f"{env._distance(env.player.position, enemy.position)}/hp{enemy.hp}/"
        f"{enemy.intent.kind.value}{'_' + enemy.intent.direction if enemy.intent.direction else ''}"
        f"@{enemy.intent.countdown}"
        for enemy in tactical_enemies if enemy.intent
    ) or "none"
    memory = env.last_action or "none"
    threats = env.imminent_threats()
    threat_summary = ("; ".join(f"{label} dmg={power}" for label, power in threats)
                      if threats else "none")
    pickups = (list(env.bow_pickups) + list(env.pistol_pickups) + list(env.arrow_bundles) +
               list(env.energy_cells))
    inventory = ""
    if env.player.loadout.bow or env.player.loadout.pistol or pickups:
        inventory = (f" Inv bow={env.player.loadout.arrows if env.player.loadout.bow else 'no'} "
                     f"pistol={env.player.loadout.energy if env.player.loadout.pistol else 'no'} "
                     f"pickup={_nearest(env.player.position, pickups)}.")
    dash_cd = env.player.cooldowns.get("dash", 0)
    emp_cd = env.player.cooldowns.get("emp", 0)
    cooldowns = f"{dash_cd}/{emp_cd}" if emp_cd else str(dash_cd)
    return (
        f"HP={env.player.hp}/100 score={env.score} pos={env.player.position[0]},{env.player.position[1]} "
        f"cd={cooldowns} last={memory}. "
        f"Adj {','.join(adjacent)}. "
        f"Near enemy={_nearest(env.player.position, enemy_positions)} "
        f"gem={_nearest(env.player.position, list(env.gems))} "
        f"kit={_nearest(env.player.position, list(env.medkits))} "
        f"fire={_nearest(env.player.position, list(env.fires))}. "
        f"Intent {intents}. "
        f"Immediate threats: {threat_summary}. "
        f"Count enemy={len(env.enemies)} gem={len(env.gems)} kit={env.player.medkits}."
        f"{inventory}"
    )
