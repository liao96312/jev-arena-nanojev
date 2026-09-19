import math

from arena.candidates import build_candidates
from arena.env import ArenaEnv
from arena.observation import encode_state


def decision_request(env: ArenaEnv) -> dict:
    return {
        "states": [{
            "id": f"arena_seed_{env.seed}_tick_{env.tick}",
            "state": encode_state(env),
            "questions": {
                "action": {
                    "type": "choice",
                    "instructions": "Choose the best action for survival and score.",
                    "criteria": build_candidates(env),
                }
            },
        }]
    }


def parse_probabilities(response: dict, candidate_ids: set[str]) -> dict[str, float]:
    try:
        states = response["states"]
        probabilities = states[0]["answers"]["action"]["probabilities"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("invalid NanoJev response shape") from exc
    if not isinstance(states, list) or len(states) != 1 or not isinstance(probabilities, dict):
        raise ValueError("invalid NanoJev response shape")
    if set(probabilities) != candidate_ids:
        raise ValueError("NanoJev probabilities do not match candidate actions")
    if any(type(value) not in (int, float) or not math.isfinite(value) or value < 0 or value > 1
           for value in probabilities.values()):
        raise ValueError("NanoJev returned invalid probabilities")
    if not math.isclose(math.fsum(probabilities.values()), 1, abs_tol=1e-5):
        raise ValueError("NanoJev probabilities do not sum to one")
    return {key: float(value) for key, value in probabilities.items()}
