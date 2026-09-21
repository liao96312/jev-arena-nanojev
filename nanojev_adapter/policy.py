import math
from collections import deque


def _gem_route_actions(env, probabilities: dict[str, float]) -> set[str]:
    moves = {action: env.add(env.player.position, action[-1]) for action in probabilities
             if action.startswith("move_")}
    blocked = set(env.walls) | set(env.pits) | {enemy.position for enemy in env.enemies}

    def distance(start: tuple[int, int], avoid_fire: bool) -> int | None:
        queue, seen = deque([(start, 0)]), {start}
        while queue:
            position, steps = queue.popleft()
            if position in env.gems:
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
    safe_non_backtracking = [action for action in probabilities if action.startswith("move_") and
                             env.add(env.player.position, action[-1]) != env.previous_player_position and
                             env.add(env.player.position, action[-1]) not in env.fires | env.spikes]

    def score(action: str) -> float:
        value = math.log(max(probabilities[action], 1e-12))
        if action.startswith("move_"):
            target = env.add(env.player.position, action[-1])
            if target == env.previous_player_position and safe_non_backtracking:
                return -math.inf
            if target in env.fires:
                value -= 2.0
            if target in env.spikes:
                value -= 2.4
            if mode == "hybrid" and env.gems:
                before = min(env._distance(env.player.position, gem) for gem in env.gems)
                after = min(env._distance(target, gem) for gem in env.gems)
                value += .8 * (before - after)
            if mode == "hybrid" and env.player.hp <= 50 and env.medkits:
                before = min(env._distance(env.player.position, medkit) for medkit in env.medkits)
                after = min(env._distance(target, medkit) for medkit in env.medkits)
                value += .6 * (before - after)
        elif mode == "hybrid" and action.startswith("attack_") and env.player.hp > 25:
            value += .4
        elif action == "heal" and env.player.hp <= 40:
            value += 1.0
        elif action == "wait" and env.last_action == "wait" and safe_non_backtracking:
            return -math.inf
        return value

    chosen = max(sorted(probabilities), key=score)
    if mode == "hybrid" and env.gems and (chosen.startswith("move_") or chosen == "wait"):
        routes = _gem_route_actions(env, probabilities)
        if routes:
            routed = max(sorted(routes), key=probabilities.__getitem__)
            if routed != chosen:
                return routed, "planner_route"
    if chosen == argmax:
        return chosen, "model_argmax"
    if argmax.startswith("move_") and env.add(env.player.position, argmax[-1]) == env.previous_player_position:
        return chosen, "backtrack_avoided"
    return chosen, "planner_rerank"


def greedy_avoid_backtrack(probabilities: dict[str, float], env) -> tuple[str, str]:
    return select_action(probabilities, env, "memory")
