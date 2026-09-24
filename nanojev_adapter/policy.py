import math
from collections import deque

from arena.boss import ApexArbiter, ChronoMantis, FurnaceHydra, IronGardener, MirrorSeraph, NullWeaver, PrismWarden, SiegeLeviathan, StormChoir, VoidAngler


def _gem_route_actions(env, probabilities: dict[str, float], targets=None) -> set[str]:
    moves = {}
    for action in probabilities:
        if action.startswith(("move_", "dash_")):
            target = env.add(env.player.position, action[-1])
            moves[action] = env.add(target, action[-1]) if action.startswith("dash_") else target
    blocked = (set(env.walls) | set(env.pits) | set(env.barrels) | set(env.null_void) | set(env.apex_cage) |
               set(env.rail_covers.values()) | {enemy.position for enemy in env.enemies})
    if isinstance(env.boss, NullWeaver) and not env.boss.exposed_rounds and targets is not None:
        blocked |= set(env.null_nodes) - set(targets)
    if targets is None:
        targets = (env.medkits if env.player.hp <= 60 and env.medkits else
                   ((env.bow_pickups if not env.player.loadout.bow else set()) |
                    (env.pistol_pickups if not env.player.loadout.pistol else set())) or env.gems)

    def distance(start: tuple[int, int], avoid_fire: bool) -> int | None:
        queue, seen = deque([(start, 0)]), {start}
        while queue:
            position, steps = queue.popleft()
            if position in targets:
                return steps
            for direction in ("n", "s", "w", "e"):
                target = env.add(position, direction)
                if (target not in seen and env.in_bounds(target) and target not in blocked and
                        (not avoid_fire or target not in env.fires | env.spikes)):
                    seen.add(target)
                    queue.append((target, steps + 1))
        return None

    for avoid_fire in (True, False):
        distances = {action: distance(target, avoid_fire) for action, target in moves.items()
                     if not avoid_fire or target not in env.fires | env.spikes}
        reachable = {action: value for action, value in distances.items() if value is not None}
        if reachable:
            best = min(reachable.values())
            return {action for action, value in reachable.items() if value == best}
    return set()


def greedy(probabilities: dict[str, float]) -> str:
    if not probabilities:
        raise ValueError("probability distribution is empty")
    return max(sorted(probabilities), key=probabilities.__getitem__)


def select_action(probabilities: dict[str, float], env, mode: str = "hybrid") -> tuple[str, str]:
    if mode not in {"model", "memory", "hybrid"}:
        raise ValueError("policy mode must be model, memory, or hybrid")
    argmax = greedy(probabilities)
    if mode == "model":
        return argmax, "model_argmax"
    risks = {action: 0 for action in probabilities}
    if env.enemies or env.fires or env.spikes or env.barrels or env.boss:
        legal = {action.value for action in env.legal_actions()}
        for action in probabilities:
            if action not in legal:
                continue
            simulation = env.clone()
            simulation.step(action)
            pending = (sum(power for _, power in simulation.imminent_threats())
                       if not simulation.done and simulation.ap_remaining < simulation.config.action_points else 0)
            risks[action] = (float("inf") if simulation.player.hp <= 0 else
                             simulation.damage_taken - env.damage_taken + pending)
    lowest_risk = min(risks.values())
    safest = {action for action, risk in risks.items() if risk == lowest_risk}
    if mode == "hybrid" and isinstance(env.boss, NullWeaver) and not env.boss.exposed_rounds:
        wrong_nodes = set(env.null_nodes) - {env.null_nodes[env.boss.node_index]}
        def landing(action: str):
            if not action.startswith(("move_", "dash_")):
                return None
            target = env.add(env.player.position, action[-1])
            return env.add(target, action[-1]) if action.startswith("dash_") else target
        safe_nodes = {action for action in safest if landing(action) not in wrong_nodes}
        if safe_nodes:
            safest = safe_nodes
    if mode == "hybrid" and env.player.hp <= 50 and "heal" in safest:
        return "heal", "survival_heal"
    safe_non_backtracking = [action for action in probabilities if action.startswith("move_") and
                             env.add(env.player.position, action[-1]) != env.previous_player_position and
                             env.add(env.player.position, action[-1]) not in env.fires | env.spikes]

    def score(action: str) -> float:
        value = math.log(max(probabilities[action], 1e-12))
        if action.startswith(("move_", "dash_")):
            target = env.add(env.player.position, action[-1])
            if action.startswith("dash_"):
                target = env.add(target, action[-1])
            if target == env.previous_player_position and safe_non_backtracking:
                value -= 1.4
            if action.startswith("move_") and target in env.fires:
                value -= 2.0
            if action.startswith("move_") and target in env.spikes:
                value -= 2.4
            objective_gain = 0
            if mode == "hybrid" and env.gems:
                before = min(env._distance(env.player.position, gem) for gem in env.gems)
                after = min(env._distance(target, gem) for gem in env.gems)
                objective_gain = before - after
                value += .8 * objective_gain
            if mode == "hybrid" and env.player.hp <= 50 and env.medkits:
                before = min(env._distance(env.player.position, medkit) for medkit in env.medkits)
                after = min(env._distance(target, medkit) for medkit in env.medkits)
                medkit_gain = before - after
                objective_gain = max(objective_gain, medkit_gain)
                value += .6 * medkit_gain
            if action.startswith("dash_") and objective_gain <= 0 and not env.imminent_threats():
                value -= 1.25
        elif mode == "hybrid" and action.startswith("attack_") and env.player.hp > 25:
            value += .4
        elif mode == "hybrid" and action.startswith("shoot_"):
            value += 1.5
        elif action == "heal" and env.player.hp <= 40:
            value += 1.0
        elif action == "wait" and env.last_action == "wait" and safe_non_backtracking:
            return -math.inf
        return value

    chosen = max(sorted(safest), key=score)
    if mode == "hybrid" and isinstance(env.boss, (PrismWarden, FurnaceHydra, StormChoir, ChronoMantis, VoidAngler, IronGardener, MirrorSeraph, SiegeLeviathan, NullWeaver, ApexArbiter)):
        boss = env.boss
        if boss.exposed_rounds:
            shots = {action for action in safest if action.startswith("shoot_")}
            if shots:
                return max(sorted(shots), key=probabilities.__getitem__), "boss_tactics"
            strikes = {action for action in safest if action.startswith("attack_")}
            if strikes:
                return max(sorted(strikes), key=probabilities.__getitem__), "boss_tactics"
            targets = ({(boss.position[0], y) for y in range(7, 14)} if isinstance(boss, (SiegeLeviathan, NullWeaver, ApexArbiter)) else
                       {(boss.position[0], y) for y in range(7, 15)} if isinstance(boss, MirrorSeraph)
                       else {(boss.position[0], y) for y in range(6, 17)}) - env.walls
        elif isinstance(boss, ApexArbiter):
            if (boss.seals == 0 or boss.kind == "cage_barrage") and boss.countdown:
                if boss.gate in env.apex_cage:
                    direction = next((direction for direction in ("n", "s", "w", "e")
                                      if env.add(env.player.position, direction) == boss.gate), None)
                    attack = f"attack_{direction}"
                    if direction and attack in safest:
                        return attack, "boss_tactics"
                if boss.gate_broken and boss.gate:
                    targets = {boss.gate}
                elif boss.countdown == 2 and env.player.position == boss.target and "wait" in safest:
                    return "wait", "boss_tactics"
                else:
                    targets = {boss.target or env.player.position}
            elif boss.seals == 0:
                targets = {(10, 15)}
            elif boss.seals == 1:
                targets = {env.apex_seals[1]}
                if env.player.position in targets and "wait" in safest:
                    return "wait", "boss_tactics"
            elif boss.seals == 2:
                targets = {(10, 13)}
                if env.player.position in targets and not boss.countdown and "wait" in safest:
                    return "wait", "boss_tactics"
            elif boss.seals == 3:
                targets = {env.apex_seals[3]}
                if env.player.position in targets and "wait" in safest:
                    return "wait", "boss_tactics"
            elif boss.kind == "verdict" and boss.countdown and boss.appeal:
                targets = {boss.appeal}
                if env.player.position == boss.appeal and "wait" in safest:
                    return "wait", "boss_tactics"
            else:
                targets = {(10, 12)}
        elif isinstance(boss, NullWeaver):
            targets = {env.null_nodes[boss.node_index]}
        elif isinstance(boss, SiegeLeviathan):
            targets = {(cover[0], cover[1] + 1) for cover in env.rail_covers.values()
                       if (cover[0], 7) in env.rail_locks - boss.broken_locks}
            if not targets:
                targets = {(10, 14)}
            if (boss.rail_target is None or boss.rail_axis == "v" and boss.rail_target == env.player.position[0]) and env.player.position in targets and "wait" in safest:
                return "wait", "boss_tactics"
        elif isinstance(boss, MirrorSeraph):
            if boss.copied_action is None:
                remaining = env.mirror_locks - boss.broken_locks
                direction = ({(6, 6): "e", (14, 6): "w", (10, 11): "s"}[min(remaining)]
                             if remaining else "n")
                action = f"move_{direction}"
                if env.ap_remaining == 1 and action in safest:
                    return action, "boss_tactics"
                if env.ap_remaining == 2 and "wait" in safest and action in probabilities:
                    return "wait", "boss_tactics"
            targets = {(10, y) for y in range(12, 15)} - env.walls
        elif isinstance(boss, PrismWarden):
            targets = env.prism_baits()
            if boss.target and env.boss_ray() and env.boss_ray()[-1] in env.reflectors:
                if "wait" in safest:
                    return "wait", "boss_tactics"
            if boss.target is None and env.player.position in targets and "wait" in safest:
                return "wait", "boss_tactics"
        elif isinstance(boss, FurnaceHydra):
            remaining = [x for x in (10, 7, 13) if x not in boss.valves_opened]
            head = (boss.head_x if boss.target and boss.attack_kind == "wave" else remaining[0])
            targets = {(head, 12)}
            if (env.player.position in targets and "wait" in safest and
                    not (boss.target and boss.attack_kind == "fireball")):
                return "wait", "boss_tactics"
        elif isinstance(boss, StormChoir):
            targets = env.relay_pads
            chain_ready = (boss.target == env.player.position and boss.attack_kind == "chain" and
                           len(env.storm_chain()) == 6)
            if env.player.position in targets and "wait" in safest and (chain_ready or
                    (boss.target is None and boss.attacks % 2 == 0)):
                return "wait", "boss_tactics"
        elif isinstance(boss, VoidAngler):
            targets = env.gravity_nodes - boss.drained_nodes
            if boss.attack_kind == "mine" and boss.target in targets:
                targets = {boss.target}
            if (boss.attack_kind == "mine" and boss.target == env.player.position and
                    env.player.position in targets and "wait" in safest):
                return "wait", "boss_tactics"
            if boss.target is None and boss.attacks % 2 == 0 and env.player.position in targets and "wait" in safest:
                return "wait", "boss_tactics"
        elif isinstance(boss, IronGardener):
            targets = env.root_plates - {(x, 12) for x in boss.refluxed_roots}
            if boss.attack_kind == "flame" and boss.target in targets:
                targets = {boss.target}
            if (boss.attack_kind == "flame" and boss.target == env.player.position and
                    env.player.position in targets and "wait" in safest):
                return "wait", "boss_tactics"
            if boss.target is None and boss.attacks % 2 == 0 and env.player.position in targets and "wait" in safest:
                return "wait", "boss_tactics"
        else:
            landing_x = env.chrono_landing_x()
            targets = {anchor for anchor in env.time_anchors if anchor[0] == landing_x}
            if env.player.position in targets and "wait" in safest:
                return "wait", "boss_tactics"
        routes = _gem_route_actions(env, probabilities, targets) & safest
        if routes:
            return max(sorted(routes), key=probabilities.__getitem__), "boss_tactics"
    needs_medkit = env.player.hp <= 50 and bool(env.medkits)
    if mode == "hybrid" and env.gems and (chosen.startswith(("move_", "dash_")) or
                                           chosen == "wait" or needs_medkit):
        routes = _gem_route_actions(env, probabilities) & safest
        if routes:
            routed = max(sorted(routes), key=probabilities.__getitem__)
            if routed != chosen:
                return routed, "planner_route"
    if chosen == argmax:
        return chosen, "model_argmax"
    if argmax not in safest:
        return chosen, "survival_dodge"
    if argmax.startswith("move_") and env.add(env.player.position, argmax[-1]) == env.previous_player_position:
        return chosen, "backtrack_avoided"
    return chosen, "planner_rerank"


def greedy_avoid_backtrack(probabilities: dict[str, float], env) -> tuple[str, str]:
    return select_action(probabilities, env, "memory")
