import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents import NanoJevAgent, RandomAgent, RuleAgent
from arena import ArenaEnv
from arena.observation import encode_state
from arena.replay import ReplayLogger


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", choices=("random", "rule", "nanojev"), default="rule")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--replay", type=Path)
    args = parser.parse_args()
    env = ArenaEnv()
    env.reset(args.seed)
    agent = {"random": lambda: RandomAgent(args.seed), "rule": RuleAgent,
             "nanojev": NanoJevAgent}[args.agent]()
    logger = ReplayLogger(args.replay) if args.replay else None
    total = 0.0
    while not env.done:
        state = env.observation()
        candidates = [action.value for action in env.legal_actions()]
        action = agent.act(env)
        result = env.step(action)
        total += result.reward
        if logger:
            logger.log(state, candidates, env, action.value, result,
                       getattr(agent, "last_probabilities", None),
                       getattr(agent, "last_latency_ms", 0),
                       getattr(agent, "last_selection_reason", "agent"))
    print(encode_state(env))
    print(f"agent={agent.name} reward={total:.2f} ticks={env.tick} gems={env.gems_collected} kills={env.kills}")


if __name__ == "__main__":
    main()
