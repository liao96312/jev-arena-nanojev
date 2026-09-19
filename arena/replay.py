import json
from dataclasses import asdict
from pathlib import Path

from .env import ArenaEnv, StepResult


class ReplayLogger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")

    def log(self, state: dict, candidate_actions: list[str], env: ArenaEnv, action: str, result: StepResult,
            probabilities: dict[str, float] | None = None, decision_ms: float = 0,
            selection_reason: str = "agent") -> None:
        record = {
            "seed": env.seed,
            "config": asdict(env.config),
            "initial_loadout": state["loadout"],
            "tick": state["tick"],
            "state": state,
            "next_state": env.observation(),
            "candidate_actions": candidate_actions,
            "probabilities": probabilities or {},
            "chosen_action": action,
            "selection_reason": selection_reason,
            "reward": result.reward,
            "events": result.events,
            "decision_ms": decision_ms,
            "hp_before": state["hp"],
            "hp_after": env.player.hp,
            "score": env.score,
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
