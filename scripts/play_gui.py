import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents import NanoJevAgent, RandomAgent, RuleAgent
from arena import ArenaEnv, campaign_config
from arena.renderer import ArenaRenderer


def make_agent(name: str, seed: int):
    return {"random": lambda: RandomAgent(seed), "rule": RuleAgent, "nanojev": NanoJevAgent}[name]()


def decide(agent, env):
    action = agent.act(env)
    return (action, getattr(agent, "last_probabilities", {action.value: 1.0}),
            getattr(agent, "last_latency_ms", 0.0), getattr(agent, "last_selection_reason", ""))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", choices=("random", "rule", "nanojev"), default="rule")
    parser.add_argument("--seed", type=int, default=61005)
    parser.add_argument("--decision-ms", type=int, default=280)
    args = parser.parse_args()

    level, campaign_score = 1, 0
    env, renderer = ArenaEnv(campaign_config(level)), ArenaRenderer()
    env.reset(args.seed)
    agent_name, agent = args.agent, make_agent(args.agent, args.seed)
    decision_ms = max(80, args.decision_ms)
    paused, running, last_step = False, True, 0
    generation, pending, pending_generation = 0, None, 0
    animation, animation_ms, level_advance_at = None, 220, 0
    probabilities: dict[str, float] = {}
    latency = 0.0
    action_name, reason, error_message = "-", "", ""
    clock = renderer.pg.time.Clock()
    executor = ThreadPoolExecutor(max_workers=1)
    try:
        while running:
            now = renderer.pg.time.get_ticks()
            for event in renderer.pg.event.get():
                if event.type == renderer.pg.QUIT:
                    running = False
                elif event.type == renderer.pg.KEYDOWN:
                    if event.key == renderer.pg.K_ESCAPE:
                        running = False
                    elif event.key == renderer.pg.K_SPACE:
                        paused = not paused
                    elif event.key in (renderer.pg.K_LEFTBRACKET, renderer.pg.K_MINUS,
                                       renderer.pg.K_KP_MINUS):
                        decision_ms = min(1000, decision_ms + 50)
                    elif event.key in (renderer.pg.K_RIGHTBRACKET, renderer.pg.K_EQUALS,
                                       renderer.pg.K_KP_PLUS):
                        decision_ms = max(80, decision_ms - 50)
                    elif event.key == renderer.pg.K_r:
                        env.reset(args.seed + level - 1)
                        agent = make_agent(agent_name, args.seed + level - 1)
                        generation += 1
                        animation, level_advance_at = None, 0
                        probabilities, action_name, reason, error_message = {}, "-", "", ""
                        paused = False
                    elif event.key in (renderer.pg.K_1, renderer.pg.K_2, renderer.pg.K_3):
                        agent_name = {renderer.pg.K_1: "random", renderer.pg.K_2: "rule",
                                      renderer.pg.K_3: "nanojev"}[event.key]
                        agent = make_agent(agent_name, args.seed)
                        generation += 1
                        animation = None
                        probabilities, action_name, reason, error_message = {}, "-", "", ""
                        paused = False
            if level_advance_at and now >= level_advance_at:
                campaign_score += env.score
                level += 1
                env = ArenaEnv(campaign_config(level))
                env.reset(args.seed + level - 1)
                agent = make_agent(agent_name, args.seed + level - 1)
                generation += 1
                pending, animation, level_advance_at = None, None, 0
                probabilities, action_name, reason = {}, "-", ""
                last_step = now
            animation_duration = 1000 if animation and "level_complete" in animation[3] else animation_ms
            if animation and now - animation[4] >= animation_duration:
                animation = None
            if pending and pending.done():
                if pending_generation != generation or env.done:
                    pending = None
                elif not paused and not animation and now - last_step >= decision_ms:
                    try:
                        action, probabilities, latency, reason = pending.result()
                    except Exception:
                        error_message = "NanoJev 推理失败，游戏已暂停；按 R 重试"
                        paused, pending = True, None
                        continue
                    old_position = env.player.position
                    old_enemies = {id(enemy): enemy.position for enemy in env.enemies}
                    action_name = action.value
                    result = env.step(action)
                    animation = (old_position, old_enemies, action_name, result.events, now)
                    if "level_complete" in result.events:
                        level_advance_at = now + 1200
                    last_step, pending = now, None
            if pending is None and not paused and not env.done:
                pending_generation = generation
                pending = executor.submit(decide, agent, env.clone())
            animation_frame = None
            if animation:
                animation_frame = (*animation[:4], min(1, (now - animation[4]) / animation_ms))
            renderer.draw(env, agent_name, probabilities, latency, paused or env.done, action_name,
                          reason, animation_frame, decision_ms, level, campaign_score, error_message)
            clock.tick(60)
    finally:
        executor.shutdown(wait=False, cancel_futures=True)
        renderer.close()


if __name__ == "__main__":
    main()
