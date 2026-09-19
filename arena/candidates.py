from .entities import Action
from .env import ArenaEnv


def build_candidates(env: ArenaEnv) -> dict[str, str]:
    descriptions = {
        Action.MOVE_N: "Move one cell north",
        Action.MOVE_S: "Move one cell south",
        Action.MOVE_W: "Move one cell west",
        Action.MOVE_E: "Move one cell east",
        Action.ATTACK_N: "Attack the adjacent enemy north",
        Action.ATTACK_S: "Attack the adjacent enemy south",
        Action.ATTACK_W: "Attack the adjacent enemy west",
        Action.ATTACK_E: "Attack the adjacent enemy east",
        Action.SHOVE_N: "Shove the adjacent enemy north",
        Action.SHOVE_S: "Shove the adjacent enemy south",
        Action.SHOVE_W: "Shove the adjacent enemy west",
        Action.SHOVE_E: "Shove the adjacent enemy east",
        Action.HEAL: "Use one carried medkit",
        Action.WAIT: "Remain in the current cell",
    }
    candidates = {}
    for action in env.legal_actions():
        description = descriptions[action]
        if action.value.startswith("move_"):
            target = env.add(env.player.position, action.value[-1])
            if target in env.fires:
                description += " onto fire"
            else:
                description += " onto a safe tile"
            if target == env.previous_player_position:
                description += "; this returns to the previous cell"
            if env.gems:
                before = min(env._distance(env.player.position, gem) for gem in env.gems)
                after = min(env._distance(target, gem) for gem in env.gems)
                description += f"; nearest gem distance {before} to {after}"
            if env.player.hp <= 50 and env.medkits:
                before = min(env._distance(env.player.position, medkit) for medkit in env.medkits)
                after = min(env._distance(target, medkit) for medkit in env.medkits)
                description += f"; nearest medkit distance {before} to {after}"
        elif action.value.startswith("shove_"):
            direction = action.value[-1]
            enemy = env.enemy_at(env.add(env.player.position, direction))
            destination = env.add(enemy.position, direction)
            if destination in env.fires:
                description += f" into fire; expected {env.config.fire_damage} fire damage"
            elif destination in env.walls:
                description += f" into a wall; expected {env.config.collision_damage} collision damage"
            elif env.enemy_at(destination):
                description += f" into another enemy; both take {env.config.collision_damage} collision damage"
            else:
                description += " by one cell"
        candidates[action.value] = description
    return candidates
