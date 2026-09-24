import json
import time
from collections import OrderedDict

from arena.candidates import build_candidates
from arena.boss import SiegeLeviathan
from arena.entities import Action
from arena.env import ArenaEnv
from nanojev_adapter.client import NanoJevClient
from nanojev_adapter.policy import select_action
from nanojev_adapter.schema import decision_request, parse_probabilities


class NanoJevAgent:
    name = "nanojev"

    def __init__(self, client: NanoJevClient | None = None, max_batch_states: int = 1,
                 policy_mode: str = "hybrid", cache_size: int = 128):
        if max_batch_states <= 0 or cache_size < 0:
            raise ValueError("max_batch_states must be positive and cache_size non-negative")
        self.client = client or NanoJevClient()
        self.max_batch_states = max_batch_states
        self.policy_mode = policy_mode
        self.cache_size = cache_size
        self._cache: OrderedDict[str, dict[str, float]] = OrderedDict()
        self.last_probabilities: dict[str, float] = {}
        self.last_latency_ms = 0.0
        self.last_selection_reason = "forced"
        self.last_batch_probabilities: list[dict[str, float]] = []
        self.last_batch_reasons: list[str] = []
        self.last_batch_latency_ms = 0.0
        self.last_batch_state_latency_ms: list[float] = []
        self.last_cache_hits = 0

    def act(self, env: ArenaEnv) -> Action:
        action = self.act_many([env])[0]
        self.last_probabilities = self.last_batch_probabilities[0]
        self.last_latency_ms = self.last_batch_state_latency_ms[0]
        self.last_selection_reason = self.last_batch_reasons[0]
        return action

    def act_many(self, envs: list[ArenaEnv]) -> list[Action]:
        if not envs:
            return []
        candidates = [build_candidates(env) for env in envs]
        actions: list[Action | None] = [None] * len(envs)
        probabilities: list[dict[str, float]] = [{} for _ in envs]
        reasons = ["forced"] * len(envs)
        state_latencies = [0.0] * len(envs)
        pending, request_states = [], {}
        for i, offered in enumerate(candidates):
            if len(offered) == 1:
                choice = next(iter(offered))
                actions[i], probabilities[i] = Action(choice), {choice: 1.0}
                continue
            if self.policy_mode == "hybrid" and isinstance(envs[i].boss, SiegeLeviathan):
                distribution = {action: 1 / len(offered) for action in offered}
                choice, _ = select_action(distribution, envs[i], self.policy_mode)
                actions[i], probabilities[i], reasons[i] = Action(choice), distribution, "boss_rule_assist"
                continue
            request_state = decision_request(envs[i])["states"][0]
            key = json.dumps((request_state["state"], request_state["questions"]),
                             ensure_ascii=False, sort_keys=True)
            if self.cache_size and key in self._cache:
                distribution = self._cache.pop(key)
                self._cache[key] = distribution
                choice, reason = select_action(distribution, envs[i], self.policy_mode)
                actions[i], probabilities[i], reasons[i] = Action(choice), dict(distribution), reason
                continue
            pending.append(i)
            request_states[i] = request_state, key

        latency = 0.0
        cache_hits = len(envs) - len(pending) - sum(len(offered) == 1 for offered in candidates)
        for start in range(0, len(pending), self.max_batch_states):
            batch = pending[start:start + self.max_batch_states]
            payload = {"states": [request_states[i][0] for i in batch]}
            started = time.perf_counter()
            try:
                response = self.client.evaluate(payload)
            except RuntimeError as exc:
                if "max_length" not in str(exc):
                    raise
                # Late Boss descriptions can exceed the fixed local context; keep the game playable.
                for i in batch:
                    distribution = {action: 1 / len(candidates[i]) for action in candidates[i]}
                    choice, _ = select_action(distribution, envs[i], self.policy_mode)
                    actions[i], probabilities[i], reasons[i] = Action(choice), distribution, "model_input_fallback"
                continue
            request_latency = (time.perf_counter() - started) * 1000
            latency += request_latency
            for i in batch:
                state_latencies[i] = request_latency
            returned = {state.get("id"): state for state in response.get("states", [])}
            for i, request_state in zip(batch, payload["states"]):
                state_id = request_state["id"]
                if state_id not in returned:
                    raise ValueError(f"NanoJev response is missing state {state_id}")
                distribution = parse_probabilities({"states": [returned[state_id]]}, set(candidates[i]))
                if self.cache_size:
                    key = request_states[i][1]
                    self._cache[key] = dict(distribution)
                    if len(self._cache) > self.cache_size:
                        self._cache.popitem(last=False)
                choice, reason = select_action(distribution, envs[i], self.policy_mode)
                actions[i], probabilities[i], reasons[i] = Action(choice), distribution, reason

        self.last_batch_probabilities = probabilities
        self.last_batch_reasons = reasons
        self.last_batch_latency_ms = latency
        self.last_batch_state_latency_ms = state_latencies
        self.last_cache_hits = cache_hits
        return [action for action in actions if action is not None]
