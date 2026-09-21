from arena.entities import Action
from arena.env import ArenaEnv
from search import beam_search


class BeamSearchAgent:
    name = "beam"

    def __init__(self, depth: int = 6, width: int = 16):
        self.depth, self.width = depth, width
        self.last_result: dict = {}

    def act(self, env: ArenaEnv) -> Action:
        self.last_result = beam_search(env, self.depth, self.width)
        values = self.last_result["action_values"]
        return Action(max(sorted(values), key=values.__getitem__))
