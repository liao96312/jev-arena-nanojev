import argparse
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents import RandomAgent, RuleAgent
from arena import ArenaConfig, ArenaEnv


def percentile(values: list[int], fraction: float) -> int:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)] if ordered else 0


def tactical_metrics(env: ArenaEnv) -> tuple[bool, bool, float, float]:
    outcomes = []
    for action in env.legal_actions():
        simulation = env.clone()
        result = simulation.step(action)
        value = result.reward + simulation.player.hp - env.player.hp
        survives = simulation.player.hp > 0
        def next_survives(next_action) -> bool:
            future = simulation.clone()
            future.step(next_action)
            return future.player.hp > 0
        future_survives = survives and (simulation.done or any(
            next_survives(next_action) for next_action in simulation.legal_actions()))
        outcomes.append((value, survives, future_survives))
    values = sorted((item[0] for item in outcomes), reverse=True)
    peak = values[0]
    weights = [math.exp(value - peak) for value in values]
    total = math.fsum(weights)
    entropy = -sum((weight / total) * math.log(weight / total) for weight in weights if weight)
    return (not any(item[1] for item in outcomes), not any(item[2] for item in outcomes),
            values[0] - values[1] if len(values) > 1 else 0.0, entropy)


def analyze(agent_name: str, episodes: int, max_ticks: int, analysis_states: int) -> dict:
    branches: list[int] = []
    tactical: list[tuple[bool, bool, float, float]] = []
    terminal_states = deaths = 0
    for seed in range(episodes):
        env = ArenaEnv(ArenaConfig(max_ticks=max_ticks))
        env.reset(seed)
        agent = RandomAgent(seed) if agent_name == "random" else RuleAgent()
        while not env.done:
            branches.append(len(env.legal_actions()))
            if len(tactical) < analysis_states:
                tactical.append(tactical_metrics(env))
            env.step(agent.act(env))
        terminal_states += 1
        deaths += env.player.hp <= 0
    counts = Counter(branches)
    total = len(branches)
    return {
        "agent": agent_name,
        "episodes": terminal_states,
        "states": total,
        "deaths": deaths,
        "average_branch_factor": round(statistics.mean(branches), 3),
        "p50_branch_factor": percentile(branches, 0.5),
        "p95_branch_factor": percentile(branches, 0.95),
        "one_action_rate": round(counts[1] / total, 6),
        "two_action_rate": round(counts[2] / total, 6),
        "six_or_more_action_rate": round(sum(n for branch, n in counts.items() if branch >= 6) / total, 6),
        "tactical_states_analyzed": len(tactical),
        "immediate_forced_death_rate": round(sum(item[0] for item in tactical) / len(tactical), 6) if tactical else 0,
        "two_step_forced_death_rate": round(sum(item[1] for item in tactical) / len(tactical), 6) if tactical else 0,
        "mean_action_value_gap": round(statistics.mean(item[2] for item in tactical), 6) if tactical else 0,
        "mean_policy_entropy": round(statistics.mean(item[3] for item in tactical), 6) if tactical else 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="统计 Arena 的真实动作分支因子")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max-ticks", type=int, default=500)
    parser.add_argument("--agents", nargs="+", choices=("random", "rule"), default=("random", "rule"))
    parser.add_argument("--analysis-states", type=int, default=500,
                        help="每个 Agent 做两步生存搜索的状态上限；分支统计仍覆盖全部状态")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.episodes <= 0 or args.max_ticks <= 0 or args.analysis_states < 0:
        parser.error("episodes/max-ticks must be positive and analysis-states non-negative")
    result = [analyze(name, args.episodes, args.max_ticks, args.analysis_states) for name in args.agents]
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
