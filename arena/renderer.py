from __future__ import annotations

import math
from pathlib import Path

from arena.env import ArenaEnv
from arena.entities import EnemyType, IntentType

AGENT_NAMES = {"random": "随机", "rule": "规则", "nanojev": "NanoJev"}
ACTION_NAMES = {
    "move_n": "向上移动", "move_s": "向下移动", "move_w": "向左移动", "move_e": "向右移动",
    "attack_n": "向上攻击", "attack_s": "向下攻击", "attack_w": "向左攻击", "attack_e": "向右攻击",
    "shoot_bow_n": "向上射箭", "shoot_bow_s": "向下射箭", "shoot_bow_w": "向左射箭", "shoot_bow_e": "向右射箭",
    "shoot_pistol_n": "向上开枪", "shoot_pistol_s": "向下开枪", "shoot_pistol_w": "向左开枪", "shoot_pistol_e": "向右开枪",
    "emp": "释放 EMP", "heal": "使用药包", "wait": "原地等待", "-": "等待决策",
}
REASON_NAMES = {
    "model_argmax": "模型首选", "backtrack_avoided": "避免折返",
    "planner_rerank": "规划重排", "planner_route": "最短路导航", "forced": "唯一可选",
}


class ArenaRenderer:
    CELL = 34
    PANEL = 350
    FOOTER = 72
    COLORS = {
        "background": (12, 17, 28), "grid": (35, 43, 58), "wall": (79, 91, 110),
        "player": (74, 222, 128), "enemy": (248, 96, 108), "gem": (69, 190, 255),
        "fire": (255, 139, 51), "medkit": (255, 235, 105), "text": (232, 238, 247),
        "muted": (145, 158, 178), "bar": (77, 140, 245), "selected": (74, 222, 128),
    }

    def __init__(self, width: int = 20, height: int = 20):
        import pygame

        self.pg = pygame
        pygame.init()
        self.map_width, self.map_height = width * self.CELL, height * self.CELL
        self.screen = pygame.display.set_mode((self.map_width + self.PANEL, self.map_height + self.FOOTER))
        pygame.display.set_caption("Jev 竞技场 · NanoJev 决策演示")
        font_path = "C:/Windows/Fonts/msyh.ttc"
        self.font = pygame.font.Font(font_path, 22)
        self.small = pygame.font.Font(font_path, 17)
        self.sprites = self._load_sprites()

    def draw(self, env: ArenaEnv, agent: str, probabilities: dict[str, float],
             latency_ms: float, paused: bool, action: str = "-", selection_reason: str = "",
             animation: tuple[tuple[int, int], dict[int, tuple[int, int]], str,
                              tuple[str, ...], float] | None = None, decision_ms: int = 280,
             level: int = 1, score_offset: int = 0, error_message: str = "") -> None:
        pg, colors = self.pg, self.COLORS
        old_player, old_enemies, animated_action, events, progress = (
            animation if animation else (env.player.position, {}, "", (), 1.0))
        eased = progress * progress * (3 - 2 * progress)
        self.screen.fill(colors["background"])
        for y in range(env.config.height):
            for x in range(env.config.width):
                rect = pg.Rect(x * self.CELL, y * self.CELL, self.CELL, self.CELL)
                pg.draw.rect(self.screen, colors["grid"], rect, 1)
        for position in env.walls:
            self._sprite(position, "wall")
        for position in env.pits:
            self._sprite(position, "pit")
        for position in env.spikes:
            self._sprite(position, "spike")
        for position in env.fires:
            self._sprite(position, "fire")
        for position in env.barrels:
            self._sprite(position, "barrel")
        for position in env.gems:
            self._sprite(position, "gem")
        for position in env.medkits:
            self._sprite(position, "medkit")
        for positions, sprite in ((env.bow_pickups, "item_bow"),
                                  (env.pistol_pickups, "item_pulse_pistol"),
                                  (env.arrow_bundles, "ammo_arrows"),
                                  (env.energy_cells, "ammo_energy_cell")):
            for position in positions:
                self._sprite(position, sprite)
        for enemy in env.enemies:
            self._intent_line(env, enemy)
        for enemy in env.enemies:
            previous = old_enemies.get(id(enemy), enemy.position)
            position = (previous[0] + (enemy.position[0] - previous[0]) * eased,
                        previous[1] + (enemy.position[1] - previous[1]) * eased)
            self._sprite(position, f"enemy_{enemy.enemy_type.value}")
            self._health_bar(position, enemy.hp, enemy.max_hp)
            self._intent(enemy.position, enemy.intent)
        player_position = env.player.position
        attack_effect = None
        if animation:
            if animated_action.startswith(("move_", "dash_")):
                player_position = (old_player[0] + (player_position[0] - old_player[0]) * eased,
                                   old_player[1] + (player_position[1] - old_player[1]) * eased)
            elif animated_action.startswith(("attack_", "shove_")):
                dx, dy = {"n": (0, -1), "s": (0, 1), "w": (-1, 0), "e": (1, 0)}[animated_action[-1]]
                lunge = 0.22 * math.sin(progress * math.pi)
                player_position = (player_position[0] + dx * lunge, player_position[1] + dy * lunge)
                attack_effect = (env.player.position, (dx, dy), progress)
            elif animated_action.startswith("shoot_"):
                dx, dy = {"n": (0, -1), "s": (0, 1), "w": (-1, 0), "e": (1, 0)}[animated_action[-1]]
                self._ranged_effect(env.player.position, (dx, dy), progress,
                                    animated_action.startswith("shoot_bow_"))
        self._sprite(player_position, "player")
        if attack_effect:
            self._attack_effect(*attack_effect)
        self._event_feedback(events)
        if error_message:
            surface = self.font.render(error_message, True, (255, 130, 130))
            background = surface.get_rect(center=(self.map_width // 2, self.map_height // 2)).inflate(28, 18)
            self.pg.draw.rect(self.screen, (55, 20, 25), background, border_radius=8)
            self.screen.blit(surface, surface.get_rect(center=background.center))

        left = self.map_width + 20
        self._text(f"第 {level} 关 · NanoJev", left, 20, colors["text"])
        self._text(f"智能体：{AGENT_NAMES.get(agent, agent)}", left, 55, colors["muted"])
        self._text(f"动作：{ACTION_NAMES.get(action, action)}", left, 80, colors["text"])
        self._text(f"推理耗时：{latency_ms:.1f} 毫秒", left, 105, colors["muted"])
        self._text(f"决策间隔：{decision_ms} 毫秒", left, 130, colors["muted"], small=True)
        if selection_reason:
            self._text(f"决策依据：{REASON_NAMES.get(selection_reason, selection_reason)}", left, 152,
                       colors["muted"], small=True)
        self._text(f"难度：敌 {env.config.enemies}  火 {env.config.fires}  刺 {env.config.spikes}  坑 {env.config.pits}",
                   left, 174, colors["muted"], small=True)
        dash_cd = env.player.cooldowns.get("dash", 0)
        emp_cd = env.player.cooldowns.get("emp", 0)
        self._text(f"技能：冲刺 {'就绪' if not dash_cd else dash_cd}  EMP {'就绪' if not emp_cd else emp_cd}",
                   left, 196, colors["selected"] if not dash_cd and not emp_cd else colors["muted"], small=True)
        loadout = env.player.loadout
        self._text(f"武器：弓 {'未获得' if not loadout.bow else f'{loadout.arrows} 箭'}  "
                   f"手枪 {'未获得' if not loadout.pistol else f'{loadout.energy} 发'}",
                   left, 218, colors["muted"], small=True)
        y = 245
        for name, probability in sorted(probabilities.items(), key=lambda item: item[1], reverse=True):
            self._text(f"{ACTION_NAMES.get(name, name)}  {probability:>6.1%}", left, y,
                       colors["text"], small=True)
            pg.draw.rect(self.screen, colors["grid"], (left, y + 20, 290, 8))
            pg.draw.rect(self.screen, colors["bar"], (left, y + 20, int(290 * probability), 8))
            y += 38

        footer_y = self.map_height + 15
        status = (f"生命 {env.player.hp:3}/100   总分 {score_offset + env.score:3}   宝石 {env.gems_collected}   "
                  f"击败 {env.kills}   轮次 {env.round}   AP {env.ap_remaining}/{env.config.action_points}   "
                  f"行动 {env.tick}/{env.config.max_ticks}")
        self._text(status, 12, footer_y, colors["text"])
        controls = "[1] 随机  [2] 规则  [3] NanoJev  [[ / ]] 调速  [空格] 暂停  [R] 重开  [Esc] 退出"
        self._text(controls + ("  已暂停/结束" if paused else ""), 12, footer_y + 28,
                   colors["muted"], small=True)
        pg.display.flip()

    def _load_sprites(self) -> dict[str, object]:
        root = Path(__file__).resolve().parents[1] / "assets" / "sprites"
        sprites = {}
        for name in ("player", "enemy_chaser", "enemy_charger", "enemy_bomber", "enemy_archer",
                     "gem", "fire", "medkit", "wall", "item_bow", "item_pulse_pistol",
                     "ammo_arrows", "ammo_energy_cell", "barrel", "spike", "pit"):
            source = self.pg.image.load(str(root / f"{name}.png")).convert_alpha()
            bounds = source.get_bounding_rect(min_alpha=16)
            cropped = source.subsurface(bounds)
            limit = self.CELL if name == "wall" else self.CELL - 3
            scale = min(limit / cropped.get_width(), limit / cropped.get_height())
            size = max(1, round(cropped.get_width() * scale)), max(1, round(cropped.get_height() * scale))
            sprites[name] = self.pg.transform.smoothscale(cropped, size)
        return sprites

    def _sprite(self, position: tuple[float, float], name: str) -> None:
        sprite = self.sprites[name]
        center = (position[0] * self.CELL + self.CELL // 2, position[1] * self.CELL + self.CELL // 2)
        self.screen.blit(sprite, sprite.get_rect(center=center))

    def _attack_effect(self, position: tuple[int, int], direction: tuple[int, int], progress: float) -> None:
        cx = position[0] * self.CELL + self.CELL // 2
        cy = position[1] * self.CELL + self.CELL // 2
        dx, dy = direction
        reach = self.CELL * (0.35 + 0.55 * progress)
        end = (cx + dx * reach, cy + dy * reach)
        side = (-dy * 7, dx * 7)
        self.pg.draw.line(self.screen, (255, 232, 120),
                          (cx + side[0], cy + side[1]), (end[0] - side[0], end[1] - side[1]),
                          max(1, round(6 * (1 - progress))))
        self.pg.draw.circle(self.screen, (255, 170, 70), (round(end[0]), round(end[1])),
                            max(2, round(7 * (1 - progress))), 2)

    def _health_bar(self, position: tuple[float, float], hp: int, maximum: int) -> None:
        x, y = position[0] * self.CELL + 4, (position[1] + 1) * self.CELL - 5
        width = self.CELL - 8
        self.pg.draw.rect(self.screen, (38, 18, 24), (x, y, width, 3))
        self.pg.draw.rect(self.screen, self.COLORS["enemy"], (x, y, round(width * hp / maximum), 3))

    def _ranged_effect(self, position, direction, progress: float, bow: bool) -> None:
        start = (position[0] * self.CELL + self.CELL // 2, position[1] * self.CELL + self.CELL // 2)
        distance = self.CELL * 5 * min(1, progress * 2)
        end = (start[0] + direction[0] * distance, start[1] + direction[1] * distance)
        color = (110, 255, 125) if bow else (80, 220, 255)
        self.pg.draw.line(self.screen, color, start, end, 3 if bow else 5)
        self.pg.draw.circle(self.screen, color, (round(end[0]), round(end[1])), 4)

    def _intent(self, position, intent) -> None:
        if not intent:
            return
        arrows = {"n": "↑", "s": "↓", "w": "←", "e": "→"}
        icon = ("⚔" if intent.kind == IntentType.MELEE else
                "B" if intent.kind == IntentType.EXPLODE else
                "C" + arrows.get(intent.direction, "·") if intent.kind == IntentType.CHARGE else
                "A" + arrows.get(intent.direction, "·") if intent.kind == IntentType.SHOOT else
                arrows.get(intent.direction, "·"))
        color = ((255, 115, 115) if intent.kind == IntentType.MELEE else
                 (235, 100, 255) if intent.kind == IntentType.EXPLODE else
                 (255, 175, 70) if intent.kind == IntentType.CHARGE else (105, 210, 255))
        if intent.kind == IntentType.SHOOT:
            color = (255, 90, 135)
        label = self.small.render(f"{icon}{intent.countdown}", True, color)
        center = (position[0] * self.CELL + self.CELL // 2, position[1] * self.CELL + 4)
        self.screen.blit(label, label.get_rect(center=center))

    def _intent_line(self, env: ArenaEnv, enemy) -> None:
        intent = enemy.intent
        if not intent or intent.kind != IntentType.SHOOT or not intent.direction:
            return
        target = enemy.position
        end = target
        while True:
            target = env.add(target, intent.direction)
            if not env.in_bounds(target) or target in env.walls:
                break
            end = target
            if target == env.player.position or env.enemy_at(target):
                break
        start_pixel = (enemy.position[0] * self.CELL + self.CELL // 2,
                       enemy.position[1] * self.CELL + self.CELL // 2)
        end_pixel = (end[0] * self.CELL + self.CELL // 2, end[1] * self.CELL + self.CELL // 2)
        self.pg.draw.line(self.screen, (255, 90, 135), start_pixel, end_pixel, 3)

    def _event_feedback(self, events: tuple[str, ...]) -> None:
        labels = []
        for event in events:
            if event == "gem": labels.append("获得宝石 +10")
            elif event == "medkit": labels.append("拾取药包 +3")
            elif event == "pickup_bow": labels.append("拾取复合弓：箭矢 +3")
            elif event == "pickup_pistol": labels.append("拾取脉冲手枪：能量 +6")
            elif event == "pickup_arrows": labels.append("拾取箭束 +3")
            elif event == "pickup_energy": labels.append("拾取能量弹匣 +6")
            elif event == "kill": labels.append("击败敌人 +20")
            elif event == "environment_kill": labels.append("环境击杀！")
            elif event.startswith("shove:"): labels.append("推动敌人")
            elif event.startswith("enemy_collision:"): labels.append("敌人碰撞")
            elif event == "bomber_explode": labels.append("炸弹怪爆炸！")
            elif event == "barrel_explode": labels.append("爆炸桶连锁爆炸！")
            elif event == "pit_fall": labels.append("敌人坠入深坑！")
            elif event.startswith("archer_shot:"): labels.append("射手放箭！")
            elif event.startswith("shoot_bow:"): labels.append("复合弓射击！")
            elif event.startswith("shoot_pistol:"): labels.append("脉冲手枪射击！")
            elif event.startswith("emp:"): labels.append(f"EMP 控制 {event.split(':')[1]} 个敌人！")
            elif event.startswith("dash:"): labels.append("冲刺！")
            elif event == "heal": labels.append("恢复生命")
            elif event == "level_complete": labels.append("关卡完成！准备进入下一关")
            elif event.startswith("damage:"): labels.append(f"受到 {event.rsplit(':', 1)[1]} 点伤害")
        if labels:
            message = "  ·  ".join(labels)
            surface = self.font.render(message, True, (255, 235, 130))
            background = surface.get_rect(center=(self.map_width // 2, 28)).inflate(24, 12)
            self.pg.draw.rect(self.screen, (35, 28, 20), background, border_radius=7)
            self.screen.blit(surface, surface.get_rect(center=background.center))

    def _text(self, value: str, x: int, y: int, color: tuple[int, int, int], small: bool = False) -> None:
        self.screen.blit((self.small if small else self.font).render(value, True, color), (x, y))

    def close(self) -> None:
        self.pg.quit()
