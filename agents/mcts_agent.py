from arena.entities import Action
from arena.env import ArenaEnv
from search.mcts import mcts_search


class MCTSAgent:
    name = "mcts"

    def __init__(self, iterations: int = 128, rollout_depth: int = 4, seed: int = 0):
        self.iterations, self.rollout_depth, self.seed = iterations, rollout_depth, seed
        self.last_result: dict = {}

    def act(self, env: ArenaEnv) -> Action:
        self.last_result = mcts_search(env, self.iterations, self.rollout_depth,
                                       seed=self.seed + env.tick)
        visits = self.last_result["visits"]
        return Action(max(sorted(visits), key=visits.__getitem__))
