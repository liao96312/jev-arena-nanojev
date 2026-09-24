from .entities import Action
from .env import ArenaEnv
from .boss import ApexArbiter, ChronoMantis, FurnaceHydra, IronGardener, MirrorSeraph, NullWeaver, PrismWarden, SiegeLeviathan, StormChoir, VoidAngler


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
    if simulation.player.loadout.bow != env.player.loadout.bow:
        parts.append("acquire_bow")
    if simulation.player.loadout.pistol != env.player.loadout.pistol:
        parts.append("acquire_pistol")
    if env.boss and simulation.boss:
        if isinstance(env.boss, PrismWarden) and simulation.boss.reflections != env.boss.reflections:
            parts.append(f"reflect {env.boss.reflections}->{simulation.boss.reflections}")
        if isinstance(env.boss, FurnaceHydra) and simulation.boss.valves_opened != env.boss.valves_opened:
            parts.append(f"valves {len(env.boss.valves_opened)}->{len(simulation.boss.valves_opened)}")
        if isinstance(env.boss, StormChoir) and simulation.boss.exposed_rounds and not env.boss.exposed_rounds:
            parts.append("lightning grounded")
        if isinstance(env.boss, VoidAngler) and simulation.boss.drained_nodes != env.boss.drained_nodes:
            parts.append(f"armor {len(env.boss.drained_nodes)}->{len(simulation.boss.drained_nodes)}")
        if isinstance(env.boss, IronGardener) and simulation.boss.refluxed_roots != env.boss.refluxed_roots:
            parts.append(f"reflux {len(env.boss.refluxed_roots)}->{len(simulation.boss.refluxed_roots)}")
        if isinstance(env.boss, MirrorSeraph) and simulation.boss.broken_locks != env.boss.broken_locks:
            parts.append(f"mirror_locks {len(env.boss.broken_locks)}->{len(simulation.boss.broken_locks)}")
        if isinstance(env.boss, SiegeLeviathan) and simulation.boss.broken_locks != env.boss.broken_locks:
            parts.append(f"rail_locks {len(env.boss.broken_locks)}->{len(simulation.boss.broken_locks)}")
        if isinstance(env.boss, NullWeaver) and simulation.boss.node_index != env.boss.node_index:
            parts.append(f"nodes {env.boss.node_index}->{simulation.boss.node_index}")
        if isinstance(env.boss, ApexArbiter) and simulation.boss.seals != env.boss.seals:
            parts.append(f"seals {env.boss.seals}->{simulation.boss.seals}")
        if simulation.boss.exposed_rounds and not env.boss.exposed_rounds:
            parts.append("core exposed")
        if simulation.boss.hp != env.boss.hp:
            parts.append(f"boss_hp {env.boss.hp}->{simulation.boss.hp}")
    elif env.boss and not simulation.boss:
        parts.append("boss defeated")
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
        Action.ATTACK_N: "Attack N",
        Action.ATTACK_S: "Attack S",
        Action.ATTACK_W: "Attack W",
        Action.ATTACK_E: "Attack E",
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
        if action.value.startswith("attack_"):
            target = env.add(env.player.position, action.value[-1])
            description += ("; break cage gate" if isinstance(env.boss, ApexArbiter) and
                            target == env.boss.gate and target in env.apex_cage else
                            "; barrel blast" if target in env.barrels else "; enemy")
        elif action.value.startswith("move_"):
            target = env.add(env.player.position, action.value[-1])
            if target in env.fires:
                description += "; fire"
            elif target in env.spikes:
                description += f"; spike damage={env.config.spike_damage}"
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
            target = env.add(env.player.position, direction)
            enemy = env.enemy_at(target)
            if enemy is None:
                description += "; push rail cover x1"
            else:
                destination = env.add(enemy.position, direction)
                if destination in env.pits:
                    description += " into pit; instant kill"
                elif destination in env.fires:
                    description += f" into fire; damage={env.config.fire_damage}"
                elif destination in env.spikes:
                    description += f" into spike; damage={env.config.spike_damage}"
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
            spike_tiles = sum(position in env.spikes for position in (middle, target))
            description += f"; cd={env.config.dash_cooldown}"
            if fire_tiles:
                description += f"; fire={fire_tiles}"
            if spike_tiles:
                description += f"; spike={spike_tiles}"
            if env.gems:
                before = min(env._distance(env.player.position, gem) for gem in env.gems)
                after = min(env._distance(target, gem) for gem in env.gems)
                description += f"; gem {before}->{after}"
        elif action.value.startswith("shoot_"):
            weapon = "bow" if action.value.startswith("shoot_bow_") else "pistol"
            range_ = env.config.bow_range if weapon == "bow" else env.config.pistol_range
            enemy_target = env._ray_target(env.player.position, action.value[-1], range_)
            barrel_target = env._barrel_target(env.player.position, action.value[-1], range_)
            gate_distance = env._apex_gate_distance(action.value[-1], range_)
            if gate_distance is not None:
                description += f"; break cage gate/{gate_distance}"
            elif barrel_target and (not enemy_target or barrel_target[1] < enemy_target[1]):
                description += f"; barrel/{barrel_target[1]} blast"
            elif enemy_target:
                enemy, distance = enemy_target
                damage = env.config.bow_damage if weapon == "bow" else env.config.pistol_damage
                boss_types = (PrismWarden, FurnaceHydra, StormChoir, ChronoMantis, VoidAngler,
                              IronGardener, MirrorSeraph, SiegeLeviathan, NullWeaver, ApexArbiter)
                kind = ("prism_warden" if isinstance(enemy, PrismWarden) else
                        "furnace_hydra" if isinstance(enemy, FurnaceHydra) else
                        "storm_choir" if isinstance(enemy, StormChoir) else
                        "chrono_mantis" if isinstance(enemy, ChronoMantis) else
                        "void_angler" if isinstance(enemy, VoidAngler) else
                        "iron_gardener" if isinstance(enemy, IronGardener) else
                        "mirror_seraph" if isinstance(enemy, MirrorSeraph) else
                        "siege_leviathan" if isinstance(enemy, SiegeLeviathan) else
                        "null_weaver" if isinstance(enemy, NullWeaver) else
                        "apex_arbiter" if isinstance(enemy, ApexArbiter) else enemy.enemy_type.value)
                remaining = (max(0, enemy.hp - damage) if not isinstance(enemy, boss_types)
                             or enemy.exposed_rounds else enemy.hp)
                description += f"; {kind}/{distance} hp {enemy.hp}->{remaining}"
                if weapon == "bow" and not isinstance(enemy, boss_types) and enemy.hp > damage:
                    description += "; push=1"
            else:
                description += "; no target"
        consequence = _immediate_consequence(env, action)
        candidates[action.value] = description + ("; immediate: " + consequence if consequence else "")
    return candidates
