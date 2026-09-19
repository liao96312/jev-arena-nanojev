from __future__ import annotations

from .entities import DIRECTIONS
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
    return f"{_direction(origin, target)} distance {distance}"


def encode_state(env: ArenaEnv) -> str:
    adjacent = []
    for name, short in (("north", "n"), ("south", "s"), ("west", "w"), ("east", "e")):
        target = env.add(env.player.position, short)
        enemy = env.enemy_at(target)
        if not env.in_bounds(target) or target in env.walls:
            value = "wall"
        elif enemy:
            value = f"enemy(hp={enemy.hp})"
        elif target in env.fires:
            value = "fire"
        else:
            value = "safe"
        adjacent.append(f"{name[0].upper()} {value}")
    enemy_positions = [enemy.position for enemy in env.enemies]
    memory = f"Last action {env.last_action}." if env.last_action else "No previous action."
    return (
        f"HP {env.player.hp}/100. Score {env.score}. "
        f"{memory} "
        f"Adjacent: {', '.join(adjacent)}. "
        f"Nearest enemy {_nearest(env.player.position, enemy_positions)}. "
        f"Nearest gem {_nearest(env.player.position, list(env.gems))}. "
        f"Nearest medkit {_nearest(env.player.position, list(env.medkits))}. "
        f"Enemies {len(env.enemies)}. Gems remaining {len(env.gems)}. "
        f"Attack {'ready' if any(env.enemy_at(env.add(env.player.position, d)) for d in DIRECTIONS) else 'unavailable'}. "
        f"Medkits carried {env.player.medkits}."
    )
