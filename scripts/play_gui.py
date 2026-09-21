import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents import JevApiAgent, NanoJevAgent, RandomAgent, RuleAgent
from arena import ArenaEnv, campaign_config
from arena.campaign_save import CampaignSave, load_campaign, save_campaign
from arena.renderer import ArenaRenderer


def make_agent(name: str, seed: int):
    if name == "jev":
        key = os.environ.get("TYPESAFE_API_KEY", "")
        key_path = Path(os.environ.get("TYPESAFE_API_KEY_FILE", Path.home() / "Desktop" /
                                       "typesafe-api-key.txt"))
        if not key and key_path.is_file():
            key = key_path.read_text(encoding="utf-8").strip()
        return JevApiAgent(api_key=key)
    return {"random": lambda: RandomAgent(seed), "rule": RuleAgent,
            "nanojev": NanoJevAgent}[name]()


def switch_agent(name: str, seed: int, pending):
    if pending:
        pending.cancel()
    return make_agent(name, seed), None


def decide(agent, env):
    action = agent.act(env)
    return (action, getattr(agent, "last_probabilities", {action.value: 1.0}),
            getattr(agent, "last_latency_ms", 0.0), getattr(agent, "last_selection_reason", ""))


def keyboard_command(pg, event) -> str:
    key, typed = event.key, getattr(event, "unicode", "").lower()
    if key == pg.K_ESCAPE:
        return "quit"
    if key == pg.K_SPACE:
        return "pause"
    if key in (pg.K_r, pg.K_F5) or typed == "r" or getattr(event, "scancode", -1) == pg.KSCAN_R:
        return "restart"
    if key in (pg.K_LEFTBRACKET, pg.K_MINUS, pg.K_KP_MINUS, pg.K_LEFT):
        return "slower"
    if key in (pg.K_RIGHTBRACKET, pg.K_EQUALS, pg.K_KP_PLUS, pg.K_RIGHT):
        return "faster"
    for number, keys in ((1, (pg.K_1, pg.K_KP1)), (2, (pg.K_2, pg.K_KP2)),
                         (3, (pg.K_3, pg.K_KP3)), (4, (pg.K_4, pg.K_KP4))):
        if key in keys or typed == str(number):
            return f"agent_{number}"
    return ""


def restart_level(env, seed: int, pending):
    if pending:
        pending.cancel()
    env.reset(seed)
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", choices=("random", "rule", "nanojev", "jev"), default="nanojev")
    parser.add_argument("--seed", type=int, default=61005)
    parser.add_argument("--decision-ms", type=int, default=280)
    args = parser.parse_args()

    save_path = Path(__file__).resolve().parents[1] / "saves" / "campaign.json"
    saved = load_campaign(save_path)
    level, campaign_score = saved.level, saved.score
    env, renderer = ArenaEnv(campaign_config(level), saved.loadout), ArenaRenderer()
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
                    continue
                command = (keyboard_command(renderer.pg, event) if event.type == renderer.pg.KEYDOWN else
                           "restart" if event.type == renderer.pg.MOUSEBUTTONDOWN and event.button == 1
                           and renderer.restart_button.collidepoint(event.pos) else "")
                if command:
                    if command == "quit":
                        running = False
                    elif command == "pause":
                        paused = not paused
                    elif command == "slower":
                        decision_ms = min(1000, decision_ms + 50)
                    elif command == "faster":
                        decision_ms = max(80, decision_ms - 50)
                    elif command == "restart":
                        pending = restart_level(env, args.seed + level - 1, pending)
                        agent = make_agent(agent_name, args.seed + level - 1)
                        generation += 1
                        animation, level_advance_at = None, 0
                        probabilities, action_name, reason, error_message = {}, "restart", "", ""
                        paused, last_step = False, now
                    elif command.startswith("agent_"):
                        agent_name = {"agent_1": "random", "agent_2": "rule",
                                      "agent_3": "nanojev", "agent_4": "jev"}[command]
                        agent, pending = switch_agent(agent_name, args.seed, pending)
                        generation += 1
                        animation = None
                        probabilities, action_name, reason, error_message = {}, "-", "", ""
                        paused = False
            if level_advance_at and now >= level_advance_at:
                campaign_score += env.score
                level += 1
                env = ArenaEnv(campaign_config(level), env.player.loadout)
                env.reset(args.seed + level - 1)
                save_campaign(save_path, CampaignSave(level, campaign_score, env.player.loadout))
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
                    except Exception as exc:
                        error_message = ("Jev API 未启用；按 3 切回本地 NanoJev" if
                                         agent.name == "jev" and "TYPESAFE_API_KEY" in str(exc) else
                                         f"{agent.name} 推理失败；可切换其他智能体或按 R 重试")
                        paused, pending = True, None
                        continue
                    old_position = env.player.position
                    old_enemies = {id(enemy): enemy.position for enemy in env.enemies}
                    action_name = action.value
                    result = env.step(action)
                    save_campaign(save_path, CampaignSave(level, campaign_score, env.player.loadout))
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
