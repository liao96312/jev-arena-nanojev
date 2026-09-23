from __future__ import annotations

from .env import ArenaEnv
from .boss import FurnaceHydra, StormChoir


ENEMY_CODES = {"chaser": "C", "charger": "G", "archer": "A", "bomber": "B"}
INTENT_CODES = {"move": "m", "melee": "a", "charge": "c", "shoot": "s",
                "explode": "x", "wait": "w"}


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
        elif target in env.pits:
            value = "pit"
        elif enemy:
            value = f"enemy{enemy.hp}"
        elif target in env.fires:
            value = "fire"
        elif target in env.spikes:
            value = "spike"
        else:
            value = "safe"
        adjacent.append(f"{name[0].upper()}:{value}")
    enemy_positions = [enemy.position for enemy in env.enemies]
    tactical_enemies = sorted(env.enemies, key=lambda enemy: env._distance(env.player.position, enemy.position))[:2]
    intents = "; ".join(
        f"{ENEMY_CODES[enemy.enemy_type.value]}:{_direction(env.player.position, enemy.position)}/"
        f"{env._distance(env.player.position, enemy.position)}/h{enemy.hp}/"
        f"{INTENT_CODES[enemy.intent.kind.value]}{enemy.intent.direction or ''}"
        f"@{enemy.intent.countdown}"
        for enemy in tactical_enemies if enemy.intent
    ) or "none"
    memory = env.last_action or "none"
    threats = env.imminent_threats()
    threat_summary = (f"Threat {'; '.join(f'{label} dmg={power}' for label, power in threats)}. "
                      if threats else "")
    pickups = (list(env.bow_pickups) + list(env.pistol_pickups) + list(env.arrow_bundles) +
               list(env.energy_cells))
    inventory = ""
    if env.player.loadout.bow or env.player.loadout.pistol or pickups:
        inventory = (f" W bow={env.player.loadout.arrows if env.player.loadout.bow else 'no'} "
                     f"gun={env.player.loadout.energy if env.player.loadout.pistol else 'no'} "
                     f"pick={_nearest(env.player.position, pickups)}.")
    dash_cd = env.player.cooldowns.get("dash", 0)
    emp_cd = env.player.cooldowns.get("emp", 0)
    cooldowns = f"{dash_cd}/{emp_cd}" if emp_cd else str(dash_cd)
    hazards = ""
    if env.spikes:
        hazards += f" s{_nearest(env.player.position, list(env.spikes))}"
    if env.pits:
        hazards += f" p{_nearest(env.player.position, list(env.pits))}"
    barrel = f" barrel={_nearest(env.player.position, list(env.barrels))}" if env.barrels else ""
    boss = ""
    if isinstance(env.boss, FurnaceHydra):
        boss = (f" Boss furnace@{env.boss.position[0]},{env.boss.position[1]} hp={env.boss.hp} "
                f"valves={sorted(env.boss.valves_opened)}/3 exposed={env.boss.exposed_rounds} "
                f"attack={env.boss.attack_kind} aim={env.boss.target or 'none'} "
                f"dmg=wave{env.boss.wave_damage}/fireball{env.boss.fireball_damage} "
                f"coolant={sorted(env.coolant_valves)}.")
    elif isinstance(env.boss, StormChoir):
        boss = (f" Boss storm@{env.boss.position} hp={env.boss.hp} exposed={env.boss.exposed_rounds} "
                f"{env.boss.attack_kind} aim={env.boss.target or '-'} "
                f"links={max(0, len(env.storm_chain()) - 2)}/4 "
                f"relay=8,8|10,8|12,8 arc{env.boss.arc_damage}/surge{env.boss.surge_damage}.")
    elif env.boss:
        target = env.boss.target
        boss = (f" Boss prism@{env.boss.position[0]},{env.boss.position[1]} hp={env.boss.hp} "
                f"reflect={env.boss.reflections}/3 exposed={env.boss.exposed_rounds} "
                f"aim={target if target else 'none'} "
                f"lunge={env.boss.lunge_target or 'none'} "
                f"dmg=beam{env.boss.beam_damage}/lunge{env.boss.lunge_damage} "
                f"mirrors={sorted(env.reflectors - env.boss.used_reflectors)}.")
    return (
        f"HP={env.player.hp}/100 score={env.score} pos={env.player.position[0]},{env.player.position[1]} "
        f"r={env.round} ap={env.ap_remaining}/{env.config.action_points} "
        f"cd={cooldowns} last={memory}. "
        f"Adj {','.join(adjacent)}. "
        f"Near e={_nearest(env.player.position, enemy_positions)} "
        f"gem={_nearest(env.player.position, list(env.gems))} "
        f"kit={_nearest(env.player.position, list(env.medkits))} "
        f"fire={_nearest(env.player.position, list(env.fires))}{hazards}{barrel}. "
        f"I {intents}. "
        f"{threat_summary}"
        f"# e={len(env.enemies)} g={len(env.gems)} kit={env.player.medkits}."
        f"{inventory}{boss}"
    )
