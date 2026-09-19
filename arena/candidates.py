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
        candidates[action.value] = description
    return candidates
