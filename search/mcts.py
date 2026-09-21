import math
import random
from dataclasses import dataclass, field

from arena.entities import Action
from arena.env import ArenaEnv
from .evaluator import evaluate_state


@dataclass
class Node:
    env: ArenaEnv
    parent: "Node | None" = None
    action: Action | None = None
    reward: float = 0.0
    children: dict[Action, "Node"] = field(default_factory=dict)
    visits: int = 0
    value_sum: float = 0.0
    _untried: list[Action] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._untried = list(self.env.legal_actions()) if not self.env.done else []

    @property
    def untried(self) -> list[Action]:
        return list(self._untried)


def _select(node: Node, exploration: float) -> Node:
    return max(node.children.values(), key=lambda child:
               child.value_sum / child.visits + exploration *
               math.sqrt(math.log(node.visits) / child.visits))


def _rollout(env: ArenaEnv, depth: int, rng: random.Random) -> float:
    simulation = env.clone()
    total = 0.0
    for ply in range(depth):
        if simulation.done:
            break
        total += .97 ** ply * simulation.step(rng.choice(simulation.legal_actions())).reward
    return total + .01 * evaluate_state(simulation)


def mcts_search(env: ArenaEnv, iterations: int = 128, rollout_depth: int = 4,
                exploration: float = math.sqrt(2), seed: int = 0) -> dict:
    if env.done:
        raise ValueError("cannot search a finished episode")
    if iterations < 1 or rollout_depth < 0 or exploration < 0:
        raise ValueError("iterations must be positive; rollout depth and exploration non-negative")
    root, rng = Node(env.clone()), random.Random(seed)
    for _ in range(iterations):
        node = root
        path_reward = 0.0
        ply = 0
        while not node.env.done and not node.untried and node.children:
            node = _select(node, exploration)
            path_reward += .97 ** ply * node.reward
            ply += 1
        if not node.env.done and node.untried:
            action = rng.choice(node.untried)
            node._untried.remove(action)
            child_env = node.env.clone()
            reward = child_env.step(action).reward
            path_reward += .97 ** ply * reward
            child = Node(child_env, node, action, reward)
            node.children[action] = child
            node = child
        value = path_reward + .97 ** (ply + 1) * _rollout(node.env, rollout_depth, rng)
        while node:
            node.visits += 1
            node.value_sum += value
            node = node.parent

    actions = env.legal_actions()
    visits = {action.value: root.children.get(action).visits if action in root.children else 0
              for action in actions}
    values = {action.value: (root.children[action].value_sum / root.children[action].visits
                             if action in root.children else -math.inf) for action in actions}
    total = sum(visits.values())
    probabilities = {action: count / total for action, count in visits.items()}
    return {"action_probs": probabilities, "action_values": values, "visits": visits,
            "iterations": iterations, "tree_nodes": 1 + sum(1 for _ in _walk(root))}


def _walk(node: Node):
    for child in node.children.values():
        yield child
        yield from _walk(child)
