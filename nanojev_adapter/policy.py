import math
from collections import deque


def _gem_route_actions(env, probabilities: dict[str, float]) -> set[str]:
    moves = {}
    for action in probabilities:
        if action.startswith(("move_", "dash_")):
            target = env.add(env.player.position, action[-1])
            moves[action] = env.add(target, action[-1]) if action.startswith("dash_") else target
    blocked = set(env.walls) | set(env.pits) | set(env.barrels) | {enemy.position for enemy in env.enemies}
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
    if env.enemies or env.fires or env.spikes or env.barrels:
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
