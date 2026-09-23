from __future__ import annotations

import math
from pathlib import Path

from arena.env import ArenaEnv
from arena.entities import EnemyType, IntentType
from arena.boss import FurnaceHydra

AGENT_NAMES = {"random": "随机", "rule": "规则", "nanojev": "NanoJev", "jev": "Jev API"}
ACTION_NAMES = {
    "move_n": "向上移动", "move_s": "向下移动", "move_w": "向左移动", "move_e": "向右移动",
    "attack_n": "向上攻击", "attack_s": "向下攻击", "attack_w": "向左攻击", "attack_e": "向右攻击",
    "shove_n": "向上推动", "shove_s": "向下推动", "shove_w": "向左推动", "shove_e": "向右推动",
    "dash_n": "向上冲刺", "dash_s": "向下冲刺", "dash_w": "向左冲刺", "dash_e": "向右冲刺",
    "shoot_bow_n": "向上射箭", "shoot_bow_s": "向下射箭", "shoot_bow_w": "向左射箭", "shoot_bow_e": "向右射箭",
    "shoot_pistol_n": "向上开枪", "shoot_pistol_s": "向下开枪", "shoot_pistol_w": "向左开枪", "shoot_pistol_e": "向右开枪",
    "emp": "释放 EMP", "heal": "使用药包", "wait": "原地等待", "restart": "已重新开始", "-": "等待决策",
}
REASON_NAMES = {
    "model_argmax": "模型首选", "backtrack_avoided": "避免折返",
    "planner_rerank": "规划重排", "planner_route": "最短路导航",
    "survival_heal": "低血量优先治疗", "survival_dodge": "避开敌方攻击", "forced": "唯一可选",
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
        pygame.display.set_caption("Jev 竞技场 · AI 决策演示")
        font_path = "C:/Windows/Fonts/msyh.ttc"
        self.font = pygame.font.Font(font_path, 22)
        self.small = pygame.font.Font(font_path, 17)
        self.player_facing = "s"
        self.restart_button = pygame.Rect(self.map_width + self.PANEL - 112,
                                          self.map_height + 34, 100, 30)
        self.level_button = pygame.Rect(self.map_width + self.PANEL - 222,
                                        self.map_height + 34, 100, 30)
        self.sprites = self._load_sprites()

    def draw(self, env: ArenaEnv, agent: str, probabilities: dict[str, float],
             latency_ms: float, paused: bool, action: str = "-", selection_reason: str = "",
             animation: tuple[tuple[int, int], dict[int, tuple[int, int]], str,
                              tuple[str, ...], float] | None = None, decision_ms: int = 280,
             level: int = 1, score_offset: int = 0, error_message: str = "",
             level_selection: str | None = None) -> None:
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
        for position in env.reflectors:
            center = (position[0] * self.CELL + self.CELL // 2,
                      position[1] * self.CELL + self.CELL // 2)
            pg.draw.polygon(self.screen, (77, 216, 241),
                            [(center[0], center[1] - 15), (center[0] + 12, center[1]),
                             (center[0], center[1] + 15), (center[0] - 12, center[1])], 3)
            pg.draw.line(self.screen, (235, 250, 255),
                         (center[0] - 7, center[1] + 6), (center[0] + 7, center[1] - 6), 2)
        for position in env.coolant_valves:
            opened = isinstance(env.boss, FurnaceHydra) and position[0] in env.boss.valves_opened
            center = (position[0] * self.CELL + self.CELL // 2,
                      position[1] * self.CELL + self.CELL // 2)
            pg.draw.circle(self.screen, (68, 183, 215) if opened else (87, 232, 255), center, 14, 3)
            pg.draw.circle(self.screen, (180, 245, 255), center, 7, 2)
            pg.draw.line(self.screen, (165, 241, 255), (center[0] - 8, center[1]),
                         (center[0] + 8, center[1]), 2)
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
        if env.boss:
            self._boss_telegraph(env)
        for enemy in env.enemies:
            previous = old_enemies.get(id(enemy), enemy.position)
            position = (previous[0] + (enemy.position[0] - previous[0]) * eased,
                        previous[1] + (enemy.position[1] - previous[1]) * eased)
            self._sprite(position, f"enemy_{enemy.enemy_type.value}")
            self._health_bar(position, enemy.hp, enemy.max_hp)
            self._intent(enemy.position, enemy.intent)
        if env.boss:
            boss_position = env.boss.position
            move = next((event.split(":") for event in events if event.startswith("boss_move:")), None)
            if move:
                boss_position = (int(move[1]) + (int(move[3]) - int(move[1])) * eased,
                                 int(move[2]) + (int(move[4]) - int(move[2])) * eased)
            furnace = isinstance(env.boss, FurnaceHydra)
            self._sprite(boss_position, "boss_furnace_hydra" if furnace else "boss_prism_warden")
            if "boss_shield_break" in events:
                self._sprite(boss_position, "effect_boss_magma_wave" if furnace else "effect_boss_prism_burst")
            if any(event.startswith("boss_hit:") for event in events):
                center = (round((boss_position[0] + .5) * self.CELL),
                          round((boss_position[1] + .5) * self.CELL))
                pg.draw.circle(self.screen, (255, 245, 255), center,
                               round(26 * (1 - progress) + 4), 3)
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
                shot_event = next((event for event in events
                                   if event.startswith(animated_action.rsplit("_", 1)[0] + ":")), "")
                distance = int(shot_event.rsplit(":", 1)[1]) if shot_event else 5
                self._ranged_effect(env.player.position, (dx, dy), progress,
                                    animated_action.startswith("shoot_bow_"), distance)
        facing_action = animated_action or env.last_action or ""
        if facing_action.startswith(("move_", "dash_", "attack_", "shove_", "shoot_")):
            self.player_facing = facing_action[-1]
        self._sprite(player_position, f"player_{self.player_facing}")
        if animation and animated_action.startswith("dash_"):
            self._dash_effect(player_position, animated_action[-1], progress)
        if attack_effect:
            self._attack_effect(*attack_effect)
        for event in events:
            if event.startswith("enemy_attack_at:"):
                parts = event.split(":")
                start, end = (int(parts[1]), int(parts[2])), (int(parts[3]), int(parts[4]))
                direction = (end[0] - start[0], end[1] - start[1])
                self._attack_effect(start, direction, progress, "effect_enemy_claw")
                continue
            if event.startswith("explosion_at:"):
                parts = event.split(":")
                self._explosion_effect((int(parts[1]), int(parts[2])), progress,
                                       int(parts[3]) if len(parts) > 3 else 1)
                continue
            if event.startswith("boss_prism_shot:"):
                parts = event.split(":")
                start, end = (int(parts[1]), int(parts[2])), (int(parts[3]), int(parts[4]))
                reflected = parts[5] == "1"
                shot_start, shot_end = ((end, start) if reflected and progress >= .5 else (start, end))
                shot_progress = ((progress - .5) * 2 if reflected and progress >= .5 else
                                 progress * 2 if reflected else progress)
                self._projectile_effect(shot_start, shot_end, shot_progress,
                                        "projectile_boss_prism", (215, 95, 255))
                continue
            if event.startswith("furnace_fireball:"):
                parts = event.split(":")
                self._projectile_effect((int(parts[1]), int(parts[2])),
                                        (int(parts[3]), int(parts[4])), progress,
                                        "projectile_boss_fireball", (255, 110, 40))
                continue
            if event.startswith("furnace_wave:"):
                x = int(event.split(":")[1])
                for y in range(8, 17):
                    if progress >= (y - 8) / 13:
                        self._sprite((x, y), "effect_boss_magma_wave")
                continue
            if not event.startswith("archer_shot:"):
                continue
            parts = event.split(":")
            if len(parts) == 6:
                self._projectile_effect((int(parts[2]), int(parts[3])),
                                        (int(parts[4]), int(parts[5])), progress,
                                        "projectile_enemy_laser", (255, 65, 135))
        self._event_feedback(events)
        if error_message:
            surface = self.font.render(error_message, True, (255, 130, 130))
            background = surface.get_rect(center=(self.map_width // 2, self.map_height // 2)).inflate(28, 18)
            self.pg.draw.rect(self.screen, (55, 20, 25), background, border_radius=8)
            self.screen.blit(surface, surface.get_rect(center=background.center))

        left = self.map_width + 20
        self._text(f"第 {level} 关 · Jev Arena", left, 20, colors["text"])
        if env.boss:
            if isinstance(env.boss, FurnaceHydra):
                state = (f"核心开放 {env.boss.exposed_rounds} 回合" if env.boss.exposed_rounds else
                         f"冷却阀 {len(env.boss.valves_opened)}/3")
                label, tint = "熔炉三头机", (255, 171, 104)
            else:
                state = (f"核心开放 {env.boss.exposed_rounds} 回合" if env.boss.exposed_rounds else
                         f"镜面反射 {env.boss.reflections}/3")
                label, tint = "棱镜守卫", (244, 164, 255)
            self._text(f"{label}  HP {env.boss.hp}/{env.boss.max_hp}  {state}",
                       left, 265, tint, small=True)
        self._text(f"智能体：{AGENT_NAMES.get(agent, agent)}", left, 55, colors["muted"])
        self._text(f"动作：{ACTION_NAMES.get(action, action)}", left, 80, colors["text"])
        self._text(f"推理耗时：{latency_ms:.1f} 毫秒", left, 105, colors["muted"])
        self._text(f"决策间隔：{decision_ms} 毫秒", left, 130, colors["muted"], small=True)
        if selection_reason:
            self._text(f"决策依据：{REASON_NAMES.get(selection_reason, selection_reason)}", left, 152,
                       colors["muted"], small=True)
        self._text(f"难度：敌 {env.config.enemies}  火 {env.config.fires}  刺 {env.config.spikes}  "
                   f"坑 {env.config.pits}  敌生命 +{env.config.enemy_hp_bonus}",
                   left, 174, colors["muted"], small=True)
        dash_cd = env.player.cooldowns.get("dash", 0)
        emp_cd = env.player.cooldowns.get("emp", 0)
        self._text(f"技能：冲刺 {'就绪' if not dash_cd else dash_cd}  EMP {'就绪' if not emp_cd else emp_cd}",
                   left, 196, colors["selected"] if not dash_cd and not emp_cd else colors["muted"], small=True)
        loadout = env.player.loadout
        self._text(f"武器：弓 {'未获得' if not loadout.bow else f'{loadout.arrows} 箭'}  "
                   f"手枪 {'未获得' if not loadout.pistol else f'{loadout.energy} 发'}",
                   left, 218, colors["muted"], small=True)
        y = 294 if env.boss else 245
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
        controls = "[1] 随机  [2] 规则  [3] NanoJev  [4] Jev API  [L] 选关  [←/→] 调速  [空格] 暂停"
        self._text(controls + ("  已暂停/结束" if paused else ""), 12, footer_y + 28,
                   colors["muted"], small=True)
        pg.draw.rect(self.screen, (45, 105, 165), self.restart_button, border_radius=6)
        label = self.small.render("R / F5 重开", True, colors["text"])
        self.screen.blit(label, label.get_rect(center=self.restart_button.center))
        pg.draw.rect(self.screen, (87, 70, 160), self.level_button, border_radius=6)
        label = self.small.render("L 选择关卡", True, colors["text"])
        self.screen.blit(label, label.get_rect(center=self.level_button.center))
        if level_selection is not None:
            overlay = pg.Surface(self.screen.get_size(), pg.SRCALPHA)
            overlay.fill((4, 8, 16, 210))
            self.screen.blit(overlay, (0, 0))
            box = pg.Rect(0, 0, 430, 230)
            box.center = self.screen.get_rect().center
            pg.draw.rect(self.screen, (24, 33, 52), box, border_radius=14)
            pg.draw.rect(self.screen, (105, 88, 220), box, 2, border_radius=14)
            title = self.font.render("选择关卡", True, colors["text"])
            self.screen.blit(title, title.get_rect(center=(box.centerx, box.top + 42)))
            self._text("输入 1 到 100（整十关为 Boss）", box.left + 60, box.top + 78,
                       colors["muted"], small=True)
            value = self.font.render(level_selection or "_", True, (128, 222, 255))
            self.screen.blit(value, value.get_rect(center=(box.centerx, box.top + 135)))
            hint = self.small.render("回车确认 · Backspace 删除 · Esc 取消", True, colors["muted"])
            self.screen.blit(hint, hint.get_rect(center=(box.centerx, box.bottom - 38)))
        pg.display.flip()

    def _load_sprites(self) -> dict[str, object]:
        root = Path(__file__).resolve().parents[1] / "assets" / "sprites"
        sprites = {}
        for name in ("player", "player_n", "player_e", "enemy_chaser", "enemy_charger", "enemy_bomber", "enemy_archer",
                     "gem", "fire", "medkit", "wall", "item_bow", "item_pulse_pistol",
                     "ammo_arrows", "ammo_energy_cell", "barrel", "spike", "pit",
                     "projectile_enemy_laser", "projectile_player_pulse", "projectile_player_arrow",
                     "effect_player_slash", "effect_enemy_claw", "boss_prism_warden",
                     "projectile_boss_prism", "effect_boss_prism_burst", "boss_furnace_hydra",
                     "projectile_boss_fireball", "effect_boss_magma_wave"):
            source = self.pg.image.load(str(root / f"{name}.png")).convert_alpha()
            bounds = source.get_bounding_rect(min_alpha=16)
            cropped = source.subsurface(bounds)
            if name in ("effect_player_slash", "effect_enemy_claw"):
                limit = round(self.CELL * 1.75)
            elif name == "projectile_enemy_laser":
                limit = round(self.CELL * 1.8)
            elif name in ("projectile_player_pulse", "projectile_player_arrow"):
                limit = round(self.CELL * 1.35)
            elif name in ("projectile_boss_prism", "effect_boss_prism_burst", "boss_prism_warden"):
                limit = round(self.CELL * (1.5 if name == "projectile_boss_prism" else
                                           2.9 if name == "boss_prism_warden" else 2.5))
            elif name in ("boss_furnace_hydra", "projectile_boss_fireball", "effect_boss_magma_wave"):
                limit = round(self.CELL * (3.2 if name == "boss_furnace_hydra" else
                                           1.7 if name == "projectile_boss_fireball" else 2.1))
            else:
                limit = self.CELL if name == "wall" else self.CELL - 3
            scale = min(limit / cropped.get_width(), limit / cropped.get_height())
            size = max(1, round(cropped.get_width() * scale)), max(1, round(cropped.get_height() * scale))
            sprites[name] = self.pg.transform.smoothscale(cropped, size)
        sprites["player_s"] = sprites["player"]
        sprites["player_w"] = self.pg.transform.flip(sprites["player_e"], True, False)
        return sprites

    def _sprite(self, position: tuple[float, float], name: str) -> None:
        sprite = self.sprites[name]
        center = (position[0] * self.CELL + self.CELL // 2, position[1] * self.CELL + self.CELL // 2)
        self.screen.blit(sprite, sprite.get_rect(center=center))

    def _attack_effect(self, position: tuple[int, int], direction: tuple[int, int], progress: float,
                       sprite_name: str = "effect_player_slash") -> None:
        angle = 0 if direction[0] > 0 else 180 if direction[0] < 0 else 90 if direction[1] < 0 else -90
        source = self.sprites[sprite_name]
        pulse = 0.72 + 0.34 * math.sin(progress * math.pi)
        size = max(1, round(source.get_width() * pulse)), max(1, round(source.get_height() * pulse))
        sprite = self.pg.transform.rotate(self.pg.transform.smoothscale(source, size), angle)
        sprite.set_alpha(max(0, round(255 * min(1, (1 - progress) * 2.8))))
        center = (round((position[0] + direction[0] * .62) * self.CELL + self.CELL / 2),
                  round((position[1] + direction[1] * .62) * self.CELL + self.CELL / 2))
        self.screen.blit(sprite, sprite.get_rect(center=center))

    def _explosion_effect(self, position: tuple[int, int], progress: float, blast_radius: int = 1) -> None:
        center = (round(position[0] * self.CELL + self.CELL / 2),
                  round(position[1] * self.CELL + self.CELL / 2))
        maximum = (blast_radius + .45) * self.CELL
        radius = max(4, round(maximum * min(1, progress * 1.7)))
        overlay = self.pg.Surface(self.screen.get_size(), self.pg.SRCALPHA)
        fade = max(0, 1 - progress)
        self.pg.draw.circle(overlay, (255, 55, 20, round(70 * fade)), center, radius)
        self.pg.draw.circle(overlay, (255, 190, 45, round(230 * fade)), center, radius,
                            max(2, round(8 * fade)))
        core = max(3, round(radius * (.55 - .25 * progress)))
        for index in range(7):
            angle = index * math.tau / 7
            lobe = max(3, round(core * (.42 + .08 * (index % 2))))
            offset = core * .62
            self.pg.draw.circle(overlay, (255, 105 + index * 9, 25, round(210 * fade)),
                                (center[0] + math.cos(angle) * offset,
                                 center[1] + math.sin(angle) * offset), lobe)
        self.pg.draw.circle(overlay, (255, 235, 145, round(245 * fade)), center, core)
        for index in range(12):
            angle = index * math.tau / 12
            start = radius * .45
            end = radius * (1 + .18 * (index % 3))
            self.pg.draw.line(overlay, (255, 125, 35, round(220 * fade)),
                              (center[0] + math.cos(angle) * start, center[1] + math.sin(angle) * start),
                              (center[0] + math.cos(angle) * end, center[1] + math.sin(angle) * end), 3)
        reach = (blast_radius + .5) * self.CELL * min(1, progress * 1.7)
        diamond = [(center[0], center[1] - reach), (center[0] + reach, center[1]),
                   (center[0], center[1] + reach), (center[0] - reach, center[1])]
        self.pg.draw.polygon(overlay, (255, 205, 70, round(210 * fade)), diamond, 3)
        self.screen.blit(overlay, (0, 0))

    def _dash_effect(self, position: tuple[float, float], direction: str, progress: float) -> None:
        center = (round(position[0] * self.CELL + self.CELL / 2),
                  round(position[1] * self.CELL + self.CELL / 2))
        dx, dy = {"n": (0, -1), "s": (0, 1), "w": (-1, 0), "e": (1, 0)}[direction]
        for length, alpha in ((30, 75), (20, 130), (11, 210)):
            color = (65, min(255, 150 + alpha // 3), 255)
            start = (center[0] - dx * length, center[1] - dy * length)
            self.pg.draw.line(self.screen, color, start, center, max(1, alpha // 70))
        radius = round(self.CELL * (.48 + .1 * math.sin(progress * math.pi)))
        self.pg.draw.circle(self.screen, (105, 245, 255), center, radius, 3)
        self.pg.draw.circle(self.screen, (235, 255, 255), center, max(3, radius - 5), 1)

    def _health_bar(self, position: tuple[float, float], hp: int, maximum: int) -> None:
        x, y = position[0] * self.CELL + 4, (position[1] + 1) * self.CELL - 5
        width = self.CELL - 8
        self.pg.draw.rect(self.screen, (38, 18, 24), (x, y, width, 3))
        self.pg.draw.rect(self.screen, self.COLORS["enemy"], (x, y, round(width * hp / maximum), 3))

    def _ranged_effect(self, position, direction, progress: float, bow: bool, cells: int) -> None:
        start = (position[0] * self.CELL + self.CELL // 2, position[1] * self.CELL + self.CELL // 2)
        destination = (position[0] + direction[0] * cells, position[1] + direction[1] * cells)
        self._projectile_effect(position, destination, progress,
                                "projectile_player_arrow" if bow else "projectile_player_pulse",
                                (110, 255, 125) if bow else (70, 225, 255))

    def _projectile_effect(self, start, end, progress: float, sprite_name: str,
                           glow: tuple[int, int, int]) -> None:
        travel = min(1.0, progress * 1.45)
        x = start[0] + (end[0] - start[0]) * travel
        y = start[1] + (end[1] - start[1]) * travel
        center = (round(x * self.CELL + self.CELL / 2), round(y * self.CELL + self.CELL / 2))
        direction = (end[0] - start[0], end[1] - start[1])
        angle = 0 if direction[0] > 0 else 180 if direction[0] < 0 else 90 if direction[1] < 0 else -90
        pulse = 0.92 + 0.12 * math.sin(progress * math.pi)
        source = self.sprites[sprite_name]
        size = (max(1, round(source.get_width() * pulse)), max(1, round(source.get_height() * pulse)))
        sprite = self.pg.transform.rotate(self.pg.transform.smoothscale(source, size), angle)
        if progress > 0.72:
            sprite.set_alpha(max(0, round(255 * (1 - progress) / 0.28)))
        tail = (round(center[0] - direction[0] * self.CELL * 0.32),
                round(center[1] - direction[1] * self.CELL * 0.32))
        self.pg.draw.circle(self.screen, glow, tail, max(2, round(5 * (1 - progress * 0.35))), 2)
        self.screen.blit(sprite, sprite.get_rect(center=center))
        if progress < 0.22:
            muzzle = (round(start[0] * self.CELL + self.CELL / 2),
                      round(start[1] * self.CELL + self.CELL / 2))
            self.pg.draw.circle(self.screen, glow, muzzle, round(12 * (1 - progress / 0.22)), 2)
        if progress > 0.68:
            impact = (round(end[0] * self.CELL + self.CELL / 2),
                      round(end[1] * self.CELL + self.CELL / 2))
            radius = max(3, round(4 + 16 * (progress - 0.68) / 0.32))
            self.pg.draw.circle(self.screen, glow, impact, radius, 2)

    def _intent(self, position, intent) -> None:
        if not intent or intent.kind in (IntentType.MOVE, IntentType.WAIT):
            return
        icon = ("!" if intent.kind == IntentType.MELEE else "爆" if intent.kind == IntentType.EXPLODE
                else "蓄" if intent.kind == IntentType.CHARGE else "瞄")
        color = ((255, 115, 115) if intent.kind == IntentType.MELEE else
                 (235, 100, 255) if intent.kind == IntentType.EXPLODE else
                 (255, 175, 70) if intent.kind == IntentType.CHARGE else (105, 210, 255))
        if intent.kind == IntentType.SHOOT:
            color = (255, 90, 135)
        label = self.small.render(icon, True, color)
        center = (position[0] * self.CELL + self.CELL // 2, position[1] * self.CELL + 4)
        self.screen.blit(label, label.get_rect(center=center))

    def _intent_line(self, env: ArenaEnv, enemy) -> None:
        intent = enemy.intent
        if not intent or intent.kind in (IntentType.MOVE, IntentType.WAIT):
            return
        if intent.kind == IntentType.EXPLODE:
            overlay = self.pg.Surface(self.screen.get_size(), self.pg.SRCALPHA)
            pulse = .5 + .5 * math.sin(self.pg.time.get_ticks() / 130)
            alpha = round((65 if intent.countdown > 1 else 105) + pulse * 35)
            for y in range(enemy.position[1] - env.config.bomber_radius,
                           enemy.position[1] + env.config.bomber_radius + 1):
                for x in range(enemy.position[0] - env.config.bomber_radius,
                               enemy.position[0] + env.config.bomber_radius + 1):
                    if env.in_bounds((x, y)) and env._distance(enemy.position, (x, y)) <= env.config.bomber_radius:
                        rect = self.pg.Rect(x * self.CELL + 2, y * self.CELL + 2,
                                            self.CELL - 4, self.CELL - 4)
                        self.pg.draw.rect(overlay, (255, 65, 35, alpha), rect, border_radius=7)
                        self.pg.draw.rect(overlay, (255, 190, 55, 210), rect, 2, border_radius=7)
            center = (enemy.position[0] * self.CELL + self.CELL // 2,
                      enemy.position[1] * self.CELL + self.CELL // 2)
            ring = round(self.CELL * (.35 + .12 * pulse))
            self.pg.draw.circle(overlay, (255, 235, 145, 235), center, ring, 3)
            self.screen.blit(overlay, (0, 0))
            return
        if not intent.direction:
            return
        if intent.kind == IntentType.MELEE:
            target = env.add(enemy.position, intent.direction)
            self.pg.draw.rect(self.screen, (255, 80, 95),
                              self.pg.Rect(target[0] * self.CELL + 3, target[1] * self.CELL + 3,
                                           self.CELL - 6, self.CELL - 6), 2, border_radius=5)
            return
        target = enemy.position
        end = target
        limit = env.config.charger_range if intent.kind == IntentType.CHARGE else max(env.config.width,
                                                                                     env.config.height)
        for _ in range(limit):
            target = env.add(target, intent.direction)
            if not env.in_bounds(target) or target in env.walls:
                break
            end = target
            if target == env.player.position or env.enemy_at(target):
                break
        start_pixel = (enemy.position[0] * self.CELL + self.CELL // 2,
                       enemy.position[1] * self.CELL + self.CELL // 2)
        end_pixel = (end[0] * self.CELL + self.CELL // 2, end[1] * self.CELL + self.CELL // 2)
        self._telegraph_line(start_pixel, end_pixel, intent.kind, intent.countdown)

    def _boss_telegraph(self, env: ArenaEnv) -> None:
        if isinstance(env.boss, FurnaceHydra):
            boss = env.boss
            if boss.target is None:
                return
            overlay = self.pg.Surface(self.screen.get_size(), self.pg.SRCALPHA)
            if boss.attack_kind == "wave":
                cells = [(boss.head_x, y) for y in range(8, 17)]
            else:
                x, y = boss.target
                cells = [(x, y), (x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
            for x, y in cells:
                if not env.in_bounds((x, y)):
                    continue
                safe = boss.attack_kind == "wave" and (x, y) == (boss.head_x, 12)
                self.pg.draw.rect(overlay, (70, 215, 250, 115) if safe else (255, 98, 34, 110),
                                  (x * self.CELL + 2, y * self.CELL + 2,
                                   self.CELL - 4, self.CELL - 4), border_radius=5)
            self.screen.blit(overlay, (0, 0))
            return
        if env.boss and env.boss.lunge_target:
            target = env.boss.lunge_target
            overlay = self.pg.Surface(self.screen.get_size(), self.pg.SRCALPHA)
            for x, y in ((target[0], target[1]), (target[0] - 1, target[1]),
                         (target[0] + 1, target[1]), (target[0], target[1] - 1),
                         (target[0], target[1] + 1)):
                if env.in_bounds((x, y)):
                    self.pg.draw.rect(overlay, (255, 78, 45, 115),
                                      (x * self.CELL + 2, y * self.CELL + 2,
                                       self.CELL - 4, self.CELL - 4), border_radius=5)
            center = (target[0] * self.CELL + self.CELL // 2,
                      target[1] * self.CELL + self.CELL // 2)
            self.pg.draw.circle(overlay, (255, 236, 170, 240), center, 19, 3)
            self.screen.blit(overlay, (0, 0))
            return
        path = env.boss_ray()
        if not path or not env.boss:
            return
        overlay = self.pg.Surface(self.screen.get_size(), self.pg.SRCALPHA)
        for cell in path:
            center = (cell[0] * self.CELL + self.CELL // 2,
                      cell[1] * self.CELL + self.CELL // 2)
            self.pg.draw.rect(overlay, (214, 65, 246, 75),
                              (cell[0] * self.CELL + 2, cell[1] * self.CELL + 2,
                               self.CELL - 4, self.CELL - 4), border_radius=5)
            self.pg.draw.circle(overlay, (255, 222, 255, 215), center, 4)
        start = (env.boss.position[0] * self.CELL + self.CELL // 2,
                 env.boss.position[1] * self.CELL + self.CELL // 2)
        end = (path[-1][0] * self.CELL + self.CELL // 2,
               path[-1][1] * self.CELL + self.CELL // 2)
        self.pg.draw.line(overlay, (235, 92, 255, 195), start, end, 3)
        self.screen.blit(overlay, (0, 0))

    def _telegraph_line(self, start, end, kind: IntentType, countdown: int) -> None:
        """Draw readable danger telegraphs without adding another sprite dependency."""
        charge = kind == IntentType.CHARGE
        color = (255, 174, 55) if charge else (255, 70, 135)
        overlay = self.pg.Surface(self.screen.get_size(), self.pg.SRCALPHA)
        urgency = 1.0 if countdown <= 1 else 0.72
        self.pg.draw.line(overlay, (*color, round(45 * urgency)), start, end, 13 if charge else 10)
        self.pg.draw.line(overlay, (*color, round(135 * urgency)), start, end, 6 if charge else 4)

        dx, dy = end[0] - start[0], end[1] - start[1]
        distance = max(1.0, math.hypot(dx, dy))
        ux, uy = dx / distance, dy / distance
        phase = (self.pg.time.get_ticks() / (420 if charge else 620)) % 1
        spacing = 22 if charge else 18
        for offset in range(0, round(distance), spacing):
            travel = (offset + phase * spacing) % distance
            point = (round(start[0] + ux * travel), round(start[1] + uy * travel))
            radius = 4 if charge else 3
            self.pg.draw.circle(overlay, (255, 245, 205, 235) if charge else (255, 225, 240, 235),
                                point, radius)
        self.screen.blit(overlay, (0, 0))

        if charge:
            # A broad impact gate makes the bull's yellow charge lane distinct from gunfire.
            px, py = -uy, ux
            half = 10
            gate_a = (round(end[0] + px * half), round(end[1] + py * half))
            gate_b = (round(end[0] - px * half), round(end[1] - py * half))
            self.pg.draw.line(self.screen, color, gate_a, gate_b, 4)
            self.pg.draw.circle(self.screen, (255, 235, 160), end, 7, 2)
        else:
            # Archer laser ends in a compact crosshair rather than an ambiguous plain line.
            self.pg.draw.circle(self.screen, color, end, 10, 2)
            for ax, ay in ((-15, 0), (15, 0), (0, -15), (0, 15)):
                inner = (end[0] + round(ax * .6), end[1] + round(ay * .6))
                outer = (end[0] + ax, end[1] + ay)
                self.pg.draw.line(self.screen, (255, 220, 235), inner, outer, 2)

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
            elif event.startswith("archer_shot:"): labels.append("敌方能量激光！")
            elif event == "boss_aim": labels.append("棱镜守卫锁定目标！")
            elif event.startswith("boss_prism_shot:"): labels.append("棱镜弹发射！")
            elif event.startswith("boss_reflect:"): labels.append("镜柱反射：护盾松动！")
            elif event.startswith("boss_lunge_aim:"): labels.append("Boss 突进预警：躲开红色区域！")
            elif event == "boss_lunge": labels.append("Boss 突进！")
            elif event == "boss_shield_break": labels.append("护盾破裂！攻击核心！")
            elif event == "boss_shield": labels.append("护盾阻挡攻击")
            elif event.startswith("boss_hit:"): labels.append(f"核心受到 {event.split(':')[1]} 点伤害")
            elif event == "boss_defeated": labels.append("Boss 已击败！")
            elif event.startswith("furnace_aim:wave:"): labels.append("熔岩波预警：蓝色冷却阀可挡火！")
            elif event.startswith("furnace_aim:fireball:"): labels.append("火球锁定：离开橙色落点！")
            elif event.startswith("furnace_wave:"): labels.append("熔岩波来袭！")
            elif event.startswith("furnace_fireball:"): labels.append("熔炉火球发射！")
            elif event.startswith("furnace_valve:"): labels.append("冷却阀反制成功！")
            elif event.startswith("shoot_bow:"): labels.append("复合弓射击！")
            elif event.startswith("shoot_pistol:"): labels.append("脉冲手枪射击！")
            elif event.startswith("emp:"): labels.append(f"EMP 控制 {event.split(':')[1]} 个敌人！")
            elif event.startswith("dash:"): labels.append("冲刺！")
            elif event.startswith("invulnerable:") and "冲刺无敌：免疫伤害" not in labels:
                labels.append("冲刺无敌：免疫伤害")
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
