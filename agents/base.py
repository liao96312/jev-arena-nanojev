from typing import Protocol

from arena.entities import Action
from arena.env import ArenaEnv


class Agent(Protocol):
    name: str

    def act(self, env: ArenaEnv) -> Action: ...
