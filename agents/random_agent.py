import random

from arena.entities import Action
from arena.env import ArenaEnv


class RandomAgent:
    name = "random"

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def act(self, env: ArenaEnv) -> Action:
        return self.rng.choice(env.legal_actions())
