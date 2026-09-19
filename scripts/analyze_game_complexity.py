import argparse
import json
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


def analyze(agent_name: str, episodes: int, max_ticks: int) -> dict:
    branches: list[int] = []
    terminal_states = deaths = 0
    for seed in range(episodes):
        env = ArenaEnv(ArenaConfig(max_ticks=max_ticks))
        env.reset(seed)
        agent = RandomAgent(seed) if agent_name == "random" else RuleAgent()
        while not env.done:
            branches.append(len(env.legal_actions()))
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
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="统计 Arena 的真实动作分支因子")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max-ticks", type=int, default=500)
    parser.add_argument("--agents", nargs="+", choices=("random", "rule"), default=("random", "rule"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.episodes <= 0 or args.max_ticks <= 0:
        parser.error("episodes and max-ticks must be positive")
    result = [analyze(name, args.episodes, args.max_ticks) for name in args.agents]
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
