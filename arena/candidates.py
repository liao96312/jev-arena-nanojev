from .entities import Action
from .env import ArenaEnv


def _immediate_consequence(env: ArenaEnv, action: Action) -> str:
    simulation = env.clone()
    result = simulation.step(action)
    parts = []
    if simulation.player.hp != env.player.hp:
        parts.append(f"hp {env.player.hp}->{simulation.player.hp}")
    if simulation.gems_collected != env.gems_collected:
        parts.append(f"gem+{simulation.gems_collected - env.gems_collected}")
    if simulation.kills != env.kills:
        parts.append(f"kill+{simulation.kills - env.kills}")
    if simulation.environment_kills != env.environment_kills:
        parts.append(f"env_kill+{simulation.environment_kills - env.environment_kills}")
    if simulation.player.loadout.arrows != env.player.loadout.arrows:
        parts.append(f"arrows {env.player.loadout.arrows}->{simulation.player.loadout.arrows}")
    if simulation.player.loadout.energy != env.player.loadout.energy:
        parts.append(f"energy {env.player.loadout.energy}->{simulation.player.loadout.energy}")
    threats = simulation.imminent_threats()
    if threats:
        parts.append(f"next_damage={sum(power for _, power in threats)}")
    if result.done:
        parts.append("terminal")
    return "; ".join(parts)


def build_candidates(env: ArenaEnv) -> dict[str, str]:
    descriptions = {
        Action.MOVE_N: "Move N",
        Action.MOVE_S: "Move S",
        Action.MOVE_W: "Move W",
        Action.MOVE_E: "Move E",
        Action.ATTACK_N: "Attack N enemy",
        Action.ATTACK_S: "Attack S enemy",
        Action.ATTACK_W: "Attack W enemy",
        Action.ATTACK_E: "Attack E enemy",
        Action.SHOVE_N: "Shove N enemy",
        Action.SHOVE_S: "Shove S enemy",
        Action.SHOVE_W: "Shove W enemy",
        Action.SHOVE_E: "Shove E enemy",
        Action.DASH_N: "Dash N x2",
        Action.DASH_S: "Dash S x2",
        Action.DASH_W: "Dash W x2",
        Action.DASH_E: "Dash E x2",
        Action.SHOOT_BOW_N: "Bow N",
        Action.SHOOT_BOW_S: "Bow S",
        Action.SHOOT_BOW_W: "Bow W",
        Action.SHOOT_BOW_E: "Bow E",
        Action.SHOOT_PISTOL_N: "Pistol N",
        Action.SHOOT_PISTOL_S: "Pistol S",
        Action.SHOOT_PISTOL_W: "Pistol W",
        Action.SHOOT_PISTOL_E: "Pistol E",
        Action.EMP: "EMP nearby; stun=1; cd=4",
        Action.HEAL: "Use medkit",
        Action.WAIT: "Wait",
    }
    candidates = {}
    for action in env.legal_actions():
        description = descriptions[action]
        if action.value.startswith("move_"):
            target = env.add(env.player.position, action.value[-1])
            if target in env.fires:
                description += "; fire"
            else:
                description += "; clear"
            if target == env.previous_player_position:
                description += "; previous cell"
            if env.gems:
                before = min(env._distance(env.player.position, gem) for gem in env.gems)
                after = min(env._distance(target, gem) for gem in env.gems)
                description += f"; gem {before}->{after}"
            if env.player.hp <= 50 and env.medkits:
                before = min(env._distance(env.player.position, medkit) for medkit in env.medkits)
                after = min(env._distance(target, medkit) for medkit in env.medkits)
                description += f"; kit {before}->{after}"
        elif action.value.startswith("shove_"):
            direction = action.value[-1]
            enemy = env.enemy_at(env.add(env.player.position, direction))
            destination = env.add(enemy.position, direction)
            if destination in env.fires:
                description += f" into fire; damage={env.config.fire_damage}"
            elif destination in env.walls:
                description += f" into wall; damage={env.config.collision_damage}"
            elif env.enemy_at(destination):
                description += f" into enemy; both damage={env.config.collision_damage}"
            else:
                description += " x1"
        elif action.value.startswith("dash_"):
            direction = action.value[-1]
            middle = env.add(env.player.position, direction)
            target = env.add(middle, direction)
            fire_tiles = sum(position in env.fires for position in (middle, target))
            description += f"; cd={env.config.dash_cooldown}"
            description += (f"; fire={fire_tiles}" if fire_tiles else "; fire=0")
            if env.gems:
                before = min(env._distance(env.player.position, gem) for gem in env.gems)
                after = min(env._distance(target, gem) for gem in env.gems)
                description += f"; gem {before}->{after}"
        elif action.value.startswith("shoot_"):
            weapon = "bow" if action.value.startswith("shoot_bow_") else "pistol"
            range_ = env.config.bow_range if weapon == "bow" else env.config.pistol_range
            enemy, distance = env._ray_target(env.player.position, action.value[-1], range_)
            damage = env.config.bow_damage if weapon == "bow" else env.config.pistol_damage
            description += f"; {enemy.enemy_type.value}/{distance} hp {enemy.hp}->{max(0, enemy.hp - damage)}"
            if weapon == "bow" and enemy.hp > damage:
                description += "; push=1"
        consequence = _immediate_consequence(env, action)
        candidates[action.value] = description + ("; immediate: " + consequence if consequence else "")
    return candidates
