import argparse
import csv
import math
import statistics
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents import BeamSearchAgent, MCTSAgent, NanoJevAgent, RandomAgent, RuleAgent
from arena import ArenaConfig, ArenaEnv, campaign_config


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)] if ordered else 0.0


def run(agent_name: str, episodes: int, max_ticks: int = 500, campaign_level: int = 0,
        mcts_iterations: int = 128, mcts_rollout_depth: int = 4) -> list[dict]:
    rows = []
    for seed in range(episodes):
        config = replace(campaign_config(campaign_level), max_ticks=max_ticks) if campaign_level else ArenaConfig(max_ticks=max_ticks)
        env = ArenaEnv(config)
        env.reset(seed)
        agent = (RandomAgent(seed) if agent_name == "random" else
                 BeamSearchAgent() if agent_name == "beam" else RuleAgent())
        if agent_name == "mcts":
            agent = MCTSAgent(mcts_iterations, mcts_rollout_depth, seed)
        reward, latencies = 0.0, []
        while not env.done:
            started = time.perf_counter()
            action = agent.act(env)
            latencies.append((time.perf_counter() - started) * 1000)
            reward += env.step(action).reward
        rows.append({"agent": agent_name, "seed": seed, "reward": round(reward, 3),
                     "ticks": env.tick, "gems": env.gems_collected, "kills": env.kills,
                     "damage": env.damage_taken, "death": int(env.player.hp <= 0),
                     "latency_p50_ms": round(percentile(latencies, .5), 3),
                     "latency_p95_ms": round(percentile(latencies, .95), 3), "mean_entropy": 0.0,
                     "selector_interventions": 0, "two_step_backtracks": 0, "unique_cells": 0})
    return rows


def run_nanojev(episodes: int, max_ticks: int = 500, policy: str = "hybrid",
                max_batch_states: int = 1, campaign_level: int = 0) -> list[dict]:
    config = replace(campaign_config(campaign_level), max_ticks=max_ticks) if campaign_level else ArenaConfig(max_ticks=max_ticks)
    envs = [ArenaEnv(config) for _ in range(episodes)]
    for seed, env in enumerate(envs):
        env.reset(seed)
    agent, rewards = NanoJevAgent(policy_mode=policy, max_batch_states=max_batch_states), [0.0] * episodes
    latencies, entropies = [[] for _ in envs], [[] for _ in envs]
    positions = [[] for _ in envs]
    interventions = [0] * episodes
    while active := [i for i, env in enumerate(envs) if not env.done]:
        actions = agent.act_many([envs[i] for i in active])
        for offset, (i, action) in enumerate(zip(active, actions)):
            rewards[i] += envs[i].step(action).reward
            if agent.last_batch_state_latency_ms[offset]:
                latencies[i].append(agent.last_batch_state_latency_ms[offset])
            distribution = agent.last_batch_probabilities[offset]
            entropies[i].append(-sum(p * math.log(p) for p in distribution.values() if p))
            positions[i].append(envs[i].player.position)
            interventions[i] += agent.last_batch_reasons[offset] != "model_argmax"
    return [{"agent": f"nanojev_{policy}", "seed": seed, "reward": round(rewards[seed], 3),
             "ticks": env.tick, "gems": env.gems_collected, "kills": env.kills,
             "damage": env.damage_taken, "death": int(env.player.hp <= 0),
             "latency_p50_ms": round(percentile(latencies[seed], .5), 3),
             "latency_p95_ms": round(percentile(latencies[seed], .95), 3),
             "mean_entropy": round(statistics.mean(entropies[seed]), 6),
             "selector_interventions": interventions[seed],
             "two_step_backtracks": sum(i >= 2 and positions[seed][i] == positions[seed][i-2]
                                        for i in range(len(positions[seed]))),
             "unique_cells": len(set(positions[seed]))}
            for seed, env in enumerate(envs)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max-ticks", type=int, default=500)
    parser.add_argument("--agents", nargs="+", choices=("random", "rule", "beam", "mcts", "nanojev"), default=("random", "rule"))
    parser.add_argument("--policy", choices=("model", "memory", "hybrid"), default="hybrid")
    parser.add_argument("--max-batch-states", type=int, default=1)
    parser.add_argument("--campaign-level", type=int, default=0)
    parser.add_argument("--mcts-iterations", type=int, default=128)
    parser.add_argument("--mcts-rollout-depth", type=int, default=4)
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args()
    rows = []
    for name in args.agents:
        rows += (run_nanojev(args.episodes, args.max_ticks, args.policy, args.max_batch_states,
                             args.campaign_level) if name == "nanojev" else
                 run(name, args.episodes, args.max_ticks, args.campaign_level,
                     args.mcts_iterations, args.mcts_rollout_depth))
    for name in args.agents:
        label = f"nanojev_{args.policy}" if name == "nanojev" else name
        own = [row for row in rows if row["agent"] == label]
        print(f"{name:6} mean_reward={statistics.mean(r['reward'] for r in own):8.2f} "
              f"mean_ticks={statistics.mean(r['ticks'] for r in own):7.1f} "
              f"gems={sum(r['gems'] for r in own):4} kills={sum(r['kills'] for r in own):4} "
              f"p95_ms={statistics.mean(r['latency_p95_ms'] for r in own):7.1f}")
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0])
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
