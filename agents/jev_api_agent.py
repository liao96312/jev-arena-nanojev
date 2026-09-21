from nanojev_adapter.typesafe_client import TypeSafeJevClient

from .nanojev_agent import NanoJevAgent


class JevApiAgent(NanoJevAgent):
    name = "jev"

    def __init__(self, api_key: str | None = None, **kwargs):
        super().__init__(client=TypeSafeJevClient(api_key=api_key), **kwargs)
