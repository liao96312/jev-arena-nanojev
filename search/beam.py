import heapq
import math

from arena.env import ArenaEnv
from .evaluator import evaluate_state


def _state_key(env: ArenaEnv) -> tuple:
    return (tuple(env.observation().items()), env.gems_collected, env.kills,
            env.environment_kills, env.damage_taken, env.done)


def beam_search(env: ArenaEnv, depth: int = 6, width: int = 16,
                temperature: float = 5.0) -> dict:
    if depth < 1 or width < 1 or temperature <= 0:
        raise ValueError("depth, width, and temperature must be positive")
    if env.done:
        raise ValueError("cannot search a finished episode")

    values, visits, frontier = {}, {}, []
    cache: dict[tuple, float] = {}
    expanded = cache_hits = 0
    for action in env.legal_actions():
        child = env.clone()
        result = child.step(action)
        cumulative = result.reward
        rank = cumulative + .97 * evaluate_state(child)
        values[action.value], visits[action.value] = rank, 1
        frontier.append((rank, action.value, child, cumulative))
        cache[_state_key(child)] = cumulative
        expanded += 1
    frontier = heapq.nlargest(width, frontier, key=lambda node: node[0])

    for ply in range(1, depth):
        next_frontier = []
        for _, root, state, cumulative in frontier:
            if state.done:
                next_frontier.append((values[root], root, state, cumulative))
                continue
            for action in state.legal_actions():
                child = state.clone()
                result = child.step(action)
                next_cumulative = cumulative + .97 ** ply * result.reward
                key = _state_key(child)
                if cache.get(key, -math.inf) >= next_cumulative:
                    cache_hits += 1
                    continue
                cache[key] = next_cumulative
                rank = next_cumulative + .97 ** (ply + 1) * evaluate_state(child)
                values[root] = max(values[root], rank)
                visits[root] += 1
                next_frontier.append((rank, root, child, next_cumulative))
                expanded += 1
        if not next_frontier:
            break
        frontier = heapq.nlargest(width, next_frontier, key=lambda node: node[0])

    peak = max(values.values())
    weights = {action: math.exp((value - peak) / temperature) for action, value in values.items()}
    total = math.fsum(weights.values())
    return {"action_probs": {action: weight / total for action, weight in weights.items()},
            "action_values": values, "visits": visits, "expanded": expanded,
            "transposition_hits": cache_hits, "unique_states": len(cache)}
