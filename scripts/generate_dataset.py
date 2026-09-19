import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents import RuleAgent
from arena import ArenaConfig, ArenaEnv
from arena.candidates import build_candidates
from arena.dataset import split_for_seed, validate_dataset
from arena.entities import Action
from arena.observation import encode_state


def rollout_probabilities(env: ArenaEnv, candidate_ids, horizon: int = 8,
                          temperature: float = 5.0, discount: float = .97) -> tuple[dict[str, float], dict[str, float]]:
    if horizon <= 0 or temperature <= 0:
        raise ValueError("horizon and temperature must be positive")
    returns = {}
    for candidate in candidate_ids:
        simulation, teacher = env.clone(), RuleAgent()
        total, weight = 0.0, 1.0
        if candidate.startswith("move_"):
            target = env.add(env.player.position, candidate[-1])
            if env.gems:
                before = min(env._distance(env.player.position, gem) for gem in env.gems)
                after = min(env._distance(target, gem) for gem in env.gems)
                total += 2.0 * (before - after)
            if env.player.hp <= 50 and env.medkits:
                before = min(env._distance(env.player.position, medkit) for medkit in env.medkits)
                after = min(env._distance(target, medkit) for medkit in env.medkits)
                total += before - after
            if target == env.previous_player_position:
                total -= 2.0
        for step in range(horizon):
            action = Action(candidate) if step == 0 else teacher.act(simulation)
            result = simulation.step(action)
            total += weight * result.reward
            if result.done:
                break
            weight *= discount
        returns[candidate] = total
    peak = max(returns.values())
    weights = {key: math.exp((value - peak) / temperature) for key, value in returns.items()}
    total = math.fsum(weights.values())
    return {key: value / total for key, value in weights.items()}, returns


def generate(output: Path, records: int, per_seed: int = 50, targets: str = "one_hot",
             rollout_horizon: int = 8, temperature: float = 5.0) -> dict:
    if targets not in {"one_hot", "rollout"}:
        raise ValueError("targets must be one_hot or rollout")
    output.parent.mkdir(parents=True, exist_ok=True)
    written = seed = 0
    quotas = {
        "train": records * 70 // 100,
        "dev": records * 10 // 100,
        "calibration": records * 5 // 100,
        "test": records * 10 // 100,
    }
    quotas["ood"] = records - sum(quotas.values())
    counts = {split: 0 for split in quotas}
    with output.open("w", encoding="utf-8") as handle:
        while written < records:
            split = split_for_seed(seed)
            if counts[split] >= quotas[split]:
                seed += 1
                continue
            config = ArenaConfig(enemies=5, fires=15) if split == "ood" else ArenaConfig()
            env, agent = ArenaEnv(config), RuleAgent()
            env.reset(seed)
            if split == "ood":
                env.player.hp = 50
            sampled = 0
            while not env.done and sampled < per_seed and counts[split] < quotas[split]:
                candidates = build_candidates(env)
                action = agent.act(env)
                if len(candidates) >= 2:
                    record_id = f"arena_seed_{seed}_tick_{env.tick}"
                    if targets == "rollout":
                        probabilities, returns = rollout_probabilities(
                            env, candidates, rollout_horizon, temperature)
                    else:
                        probabilities = {key: float(key == action.value) for key in candidates}
                        returns = None
                    row = {
                        "id": record_id,
                        "state_id": record_id,
                        "family_id": "arena_v1",
                        "split": split,
                        "state": encode_state(env),
                        "questions": {"action": {
                            "type": "choice",
                            "instructions": "Choose the best action for survival and score.",
                            "criteria": candidates,
                        }},
                        "gold_probs": {"action": probabilities},
                        "gold_probs_kind": {"action": "optimal_action_policy"},
                        "metadata": {"source_group_id": f"map_seed_{seed}", "seed": seed, "tick": env.tick,
                                     "target_source": targets, "rollout_horizon": rollout_horizon if returns else None,
                                     "rollout_temperature": temperature if returns else None,
                                     "potential_shaping": "gem_progress=2, low_hp_medkit_progress=1, backtrack=-2" if returns else None,
                                     "action_returns": returns},
                    }
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    written += 1
                    counts[split] += 1
                    sampled += 1
                env.step(action)
            seed += 1
    return validate_dataset(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("datasets/generated/arena_rule_1k.jsonl"))
    parser.add_argument("--records", type=int, default=1000)
    parser.add_argument("--targets", choices=("one_hot", "rollout"), default="one_hot")
    parser.add_argument("--rollout-horizon", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=5.0)
    args = parser.parse_args()
    if args.records <= 0:
        parser.error("--records must be positive")
    print(json.dumps(generate(args.output, args.records, targets=args.targets,
                              rollout_horizon=args.rollout_horizon,
                              temperature=args.temperature), ensure_ascii=False))


if __name__ == "__main__":
    main()
