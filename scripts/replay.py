import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from arena import ArenaConfig, ArenaEnv
from arena.entities import PlayerLoadout


def canonical(value):
    return json.loads(json.dumps(value))


def recorded_state_matches(actual: dict, recorded: dict) -> bool:
    return {key: canonical(actual[key]) for key in recorded} == recorded


def verify(path: Path) -> dict:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError("replay is empty")
    env = ArenaEnv(ArenaConfig(**rows[0].get("config", {})),
                   PlayerLoadout(*rows[0].get("initial_loadout", (False, False, 0, 0))))
    env.reset(rows[0]["seed"])
    for index, row in enumerate(rows):
        if row["seed"] != env.seed or row["tick"] != env.tick:
            raise ValueError(f"step {index}: replay sequence mismatch")
        if not recorded_state_matches(env.observation(), row["state"]):
            raise ValueError(f"step {index}: state mismatch")
        if [action.value for action in env.legal_actions()] != row["candidate_actions"]:
            raise ValueError(f"step {index}: candidates mismatch")
        result = env.step(row["chosen_action"])
        if result.reward != row["reward"] or not recorded_state_matches(env.observation(), row["next_state"]):
            raise ValueError(f"step {index}: transition mismatch")
    return {"verified": True, "seed": env.seed, "steps": len(rows), "done": env.done}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.input)))
