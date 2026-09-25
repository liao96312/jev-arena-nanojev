from __future__ import annotations

import math
from pathlib import Path

from arena.env import ArenaEnv
from arena.entities import EnemyType, IntentType
from arena.boss import ApexArbiter, ChronoMantis, FurnaceHydra, IronGardener, MirrorSeraph, NullWeaver, PrismWarden, SiegeLeviathan, StormChoir, VoidAngler

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
    "boss_tactics": "Boss 机制反制",
    "model_input_fallback": "输入超出本地模型长度，规则接管",
    "boss_rule_assist": "攻城 Boss 规则辅助",
    "survival_heal": "低血量优先治疗", "survival_dodge": "避开敌方攻击", "forced": "唯一可选",
}


def boss_travel_progress(progress: float, rush: bool, distance: int) -> float:
    """Finish warned rushes quickly while keeping ordinary repositioning readable."""
    duration = .45 if rush else .62 if distance >= 2 else .8
    phase = min(1.0, progress / duration)
    return phase * phase if rush else phase * phase * (3 - 2 * phase)


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
        sheet = pygame.image.load(str(Path(__file__).resolve().parents[1] / "assets" / "sprites" /
                                      "storm_choir_cast_sheet.png")).convert_alpha()
        frame_size = sheet.get_width() // 2
        self.storm_cast_frames = tuple(pygame.transform.smoothscale(
            sheet.subsurface((col * frame_size, row * frame_size, frame_size, frame_size)),
            (self.CELL * 3, self.CELL * 3)) for row in range(2) for col in range(2))

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
                if env.apex_seals:
                    pg.draw.rect(self.screen, (27, 28, 43) if (x + y) % 2 else (32, 33, 51), rect)
                    pg.draw.rect(self.screen, (92, 78, 111), rect, 1)
                elif env.null_nodes:
                    pg.draw.rect(self.screen, (24, 29, 39) if (x + y) % 2 else (29, 35, 46), rect)
                    pg.draw.rect(self.screen, (55, 77, 89), rect, 1)
                elif env.rail_locks:
                    pg.draw.rect(self.screen, (31, 31, 39) if (x + y) % 2 else (39, 37, 43), rect)
                    pg.draw.rect(self.screen, (78, 71, 66), rect, 1)
                elif env.mirror_locks:
                    pg.draw.rect(self.screen, (35, 27, 52) if (x + y) % 2 else (42, 31, 61), rect)
                    pg.draw.rect(self.screen, (84, 63, 119), rect, 1)
                elif env.root_plates:
                    pg.draw.rect(self.screen, (22, 37, 28) if (x + y) % 2 else (28, 43, 32), rect)
                    pg.draw.rect(self.screen, (50, 78, 56), rect, 1)
                elif env.gravity_nodes:
                    pg.draw.rect(self.screen, (22, 19, 43) if (x + y) % 2 else (28, 22, 50), rect)
                    pg.draw.rect(self.screen, (55, 48, 84), rect, 1)
                elif env.time_anchors:
                    pg.draw.rect(self.screen, (37, 28, 53) if (x + y) % 2 else (43, 31, 59), rect)
                    pg.draw.rect(self.screen, (82, 58, 100), rect, 1)
                elif env.grounding_pylons:
                    pg.draw.rect(self.screen, (19, 31, 54) if (x + y) % 2 else (23, 37, 62), rect)
                    pg.draw.rect(self.screen, (42, 69, 98), rect, 1)
                elif (x, y) in env.forge_floor:
                    pg.draw.rect(self.screen, (35, 24, 25) if (x + y) % 2 else (40, 27, 27), rect)
                    pg.draw.rect(self.screen, (69, 44, 42), rect, 1)
                else:
                    pg.draw.rect(self.screen, colors["grid"], rect, 1)
        for position in env.walls:
            self._sprite(position, "wall")
        for x, y in env.apex_cage:
            rect = pg.Rect(x * self.CELL + 2, y * self.CELL + 2, self.CELL - 4, self.CELL - 4)
            pg.draw.rect(self.screen, (255, 115, 57), rect, 3, border_radius=4)
            pg.draw.line(self.screen, (255, 220, 126), rect.midtop, rect.midbottom, 2)
        for index, position in enumerate(env.apex_seals):
            color = ((255, 151, 75), (111, 214, 255), (243, 152, 255), (138, 243, 157))[index]
            active = isinstance(env.boss, ApexArbiter) and index < env.boss.seals
            center = (position[0] * self.CELL + self.CELL // 2,
                      position[1] * self.CELL + self.CELL // 2)
            pg.draw.circle(self.screen, color, center, 16, 4 if active else 2)
            pg.draw.circle(self.screen, color, center, 6 if active else 3)
        for x, y in env.null_void:
            rect = pg.Rect(x * self.CELL + 2, y * self.CELL + 2,
                           self.CELL - 4, self.CELL - 4)
            pg.draw.rect(self.screen, (7, 9, 19), rect, border_radius=3)
            pg.draw.line(self.screen, (194, 95, 226), rect.topleft, rect.bottomright, 2)
            pg.draw.line(self.screen, (194, 95, 226), rect.topright, rect.bottomleft, 2)
        for x, y in env.vine_walls:
            center = (x * self.CELL + self.CELL // 2, y * self.CELL + self.CELL // 2)
            pg.draw.line(self.screen, (131, 246, 94),
                         (center[0] - 11, center[1] + 12), (center[0] + 10, center[1] - 12), 4)
            pg.draw.line(self.screen, (237, 187, 91),
                         (center[0] - 10, center[1] - 7), (center[0] + 11, center[1] + 7), 3)
        for x, y in env.vine_seeds:
            center = (x * self.CELL + self.CELL // 2, y * self.CELL + self.CELL // 2)
            pg.draw.circle(self.screen, (155, 245, 112), center, 8, 2)
            pg.draw.circle(self.screen, (238, 205, 95), center, 3)
        for x, y in env.iron_spores:
            center = (x * self.CELL + self.CELL // 2, y * self.CELL + self.CELL // 2)
            pg.draw.circle(self.screen, (232, 108, 253), center, 13, 3)
            pg.draw.circle(self.screen, (254, 196, 250), center, 5)
        for position in env.root_plates:
            used = isinstance(env.boss, IronGardener) and position[0] in env.boss.refluxed_roots
            center = (position[0] * self.CELL + self.CELL // 2,
                      position[1] * self.CELL + self.CELL // 2)
            pg.draw.circle(self.screen, (98, 119, 88) if used else (133, 236, 122), center, 16, 3)
            pg.draw.line(self.screen, (255, 201, 102),
                         (center[0] - 8, center[1] + 8), (center[0] + 8, center[1] - 8), 2)
        for position in sorted(env.mirror_locks):
            broken = isinstance(env.boss, MirrorSeraph) and position in env.boss.broken_locks
            center = (position[0] * self.CELL + self.CELL // 2,
                      position[1] * self.CELL + self.CELL // 2)
            pg.draw.polygon(self.screen, (104, 91, 123) if broken else (255, 176, 244),
                            [(center[0], center[1] - 15), (center[0] + 14, center[1]),
                             (center[0], center[1] + 15), (center[0] - 14, center[1])], 3)
            mark = {(6, 6): "左", (14, 6): "右", (10, 11): "下"}[position]
            label = self.small.render(mark, True, (130, 112, 143) if broken else (255, 239, 255))
            self.screen.blit(label, label.get_rect(center=center))
        for position in env.rail_locks:
            broken = isinstance(env.boss, SiegeLeviathan) and position in env.boss.broken_locks
            center = (position[0] * self.CELL + self.CELL // 2,
                      position[1] * self.CELL + self.CELL // 2)
            pg.draw.rect(self.screen, (110, 104, 95) if broken else (255, 177, 93),
                         (center[0] - 13, center[1] - 13, 26, 26), 3, border_radius=4)
            pg.draw.circle(self.screen, (110, 104, 95) if broken else (255, 234, 171), center, 5)
        for index, position in enumerate(env.null_nodes):
            palette = ((99, 211, 255), (255, 205, 108), (251, 137, 213), (139, 238, 163))
            color = palette[index]
            active = isinstance(env.boss, NullWeaver) and index == env.boss.node_index
            center = (position[0] * self.CELL + self.CELL // 2,
                      position[1] * self.CELL + self.CELL // 2)
            pg.draw.rect(self.screen, color, (center[0] - 15, center[1] - 15, 30, 30),
                         4 if active else 2, border_radius=5)
            number = self.small.render(str(index + 1), True, color)
            self.screen.blit(number, number.get_rect(center=center))
        for x, y in env.breakable_walls:
            center = (x * self.CELL + self.CELL // 2, y * self.CELL + self.CELL // 2)
            pg.draw.line(self.screen, (255, 155, 218),
                         (center[0] - 10, center[1] - 12), (center[0] + 2, center[1]), 3)
            pg.draw.line(self.screen, (255, 155, 218),
                         (center[0] + 2, center[1]), (center[0] - 3, center[1] + 12), 3)
        for position in env.reflectors:
            used = isinstance(env.boss, PrismWarden) and (position in env.boss.used_reflectors or
                                                          bool(env.boss.reflector_lockout))
            center = (position[0] * self.CELL + self.CELL // 2,
                      position[1] * self.CELL + self.CELL // 2)
            pg.draw.polygon(self.screen, (79, 90, 105) if used else (77, 216, 241),
                            [(center[0], center[1] - 15), (center[0] + 12, center[1]),
                             (center[0], center[1] + 15), (center[0] - 12, center[1])], 3)
            pg.draw.line(self.screen, (235, 250, 255),
                         (center[0] - 7, center[1] + 6), (center[0] + 7, center[1] - 6), 2)
        for position in env.coolant_valves:
            opened = isinstance(env.boss, FurnaceHydra) and position[0] in env.boss.valves_opened
            hot = opened and env.boss.valve_heat.get(position[0], 0) <= 1
            center = (position[0] * self.CELL + self.CELL // 2,
                      position[1] * self.CELL + self.CELL // 2)
            pg.draw.circle(self.screen, (255, 95, 60) if hot else
                           (68, 183, 215) if opened else (87, 232, 255), center, 14, 3)
            pg.draw.circle(self.screen, (180, 245, 255), center, 7, 2)
            pg.draw.line(self.screen, (165, 241, 255), (center[0] - 8, center[1]),
                         (center[0] + 8, center[1]), 2)
        for position in env.grounding_pylons:
            center = (position[0] * self.CELL + self.CELL // 2,
                      position[1] * self.CELL + self.CELL // 2)
            pg.draw.circle(self.screen, (88, 192, 255), center, 17, 3)
            pg.draw.polygon(self.screen, (173, 237, 255),
                            [(center[0], center[1] - 12), (center[0] + 10, center[1] + 8),
                             (center[0] - 10, center[1] + 8)], 2)
        for position in env.relay_pads:
            rect = pg.Rect(position[0] * self.CELL + 6, position[1] * self.CELL + 6,
                           self.CELL - 12, self.CELL - 12)
            pg.draw.rect(self.screen, (71, 211, 255), rect, 3, border_radius=6)
            pg.draw.circle(self.screen, (190, 248, 255), rect.center, 4)
        for position in env.time_anchors:
            cooling = isinstance(env.boss, ChronoMantis) and position in env.boss.anchor_cooldowns
            center = (position[0] * self.CELL + self.CELL // 2,
                      position[1] * self.CELL + self.CELL // 2)
            pg.draw.circle(self.screen, (112, 94, 125) if cooling else (230, 187, 255), center, 16, 3)
            pg.draw.line(self.screen, (255, 226, 177), center,
                         (center[0] + 6, center[1] - 7), 3)
        for position in env.gravity_nodes:
            drained = isinstance(env.boss, VoidAngler) and position in env.boss.drained_nodes
            charged = isinstance(env.boss, VoidAngler) and position in env.boss.node_aftershock
            center = (position[0] * self.CELL + self.CELL // 2,
                      position[1] * self.CELL + self.CELL // 2)
            pg.draw.circle(self.screen, (255, 91, 163) if charged else
                           (105, 96, 130) if drained else (181, 126, 255), center, 16, 3)
            pg.draw.circle(self.screen, (120, 113, 142) if drained else (237, 207, 255), center, 6, 2)
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
        for position in env.rail_covers.values():
            rect = pg.Rect(position[0] * self.CELL + 3, position[1] * self.CELL + 3,
                           self.CELL - 6, self.CELL - 6)
            pg.draw.rect(self.screen, (75, 112, 127), rect, border_radius=5)
            pg.draw.rect(self.screen, (152, 232, 241), rect, 3, border_radius=5)
            pg.draw.line(self.screen, (221, 248, 242), rect.topleft, rect.bottomright, 2)
            pg.draw.line(self.screen, (221, 248, 242), rect.topright, rect.bottomleft, 2)
        for enemy in env.enemies:
            previous = old_enemies.get(id(enemy), enemy.position)
            route = next((event.split(":") for event in events
                          if event.startswith(f"hound_sprint:{previous[0]}:{previous[1]}:")), None)
            if route:
                points = [previous] + [(int(route[i]), int(route[i + 1])) for i in range(3, len(route), 2)]
                phase = min(len(points) - 2, int(eased * (len(points) - 1)))
                fraction = eased * (len(points) - 1) - phase
                start, end = points[phase:phase + 2]
                position = (start[0] + (end[0] - start[0]) * fraction,
                            start[1] + (end[1] - start[1]) * fraction)
                for ghost in points[:phase + 1]:
                    center = (ghost[0] * self.CELL + self.CELL // 2,
                              ghost[1] * self.CELL + self.CELL // 2)
                    pg.draw.circle(self.screen, (65, 180, 205), center, 9, 2)
            else:
                position = (previous[0] + (enemy.position[0] - previous[0]) * eased,
                            previous[1] + (enemy.position[1] - previous[1]) * eased)
            mirror_clone = bool(enemy.summoned_by and enemy.summoned_by.startswith("mirror_"))
            sprite = ("player_n" if mirror_clone else
                      {"furnace": "enemy_furnace_hatchling", "iron": "enemy_vine_hunter"}.get(
                          enemy.summoned_by, f"enemy_{enemy.enemy_type.value}"))
            self._sprite(position, sprite)
            if mirror_clone:
                self.pg.draw.circle(self.screen, (197, 100, 252),
                                    (int((position[0] + .5) * self.CELL),
                                     int((position[1] + .5) * self.CELL)),
                                    self.CELL // 2 - 2, 3)
            self._health_bar(position, enemy.hp, enemy.max_hp)
            self._intent(enemy.position, enemy.intent)
        if env.boss:
            boss_position = env.boss.position
            move = next((event.split(":") for event in events if event.startswith("boss_move:")), None)
            if move:
                start, end = (int(move[1]), int(move[2])), (int(move[3]), int(move[4]))
                rush = ("boss_lunge" in events or "chrono_retreat" in events or any(event.startswith(
                    ("chrono_leap:", "siege_charge:", "null_warp", "apex_fire:charge:",
                     "apex_fire:charge_gravity:", "storm_overdrive_move:")) for event in events))
                travel = boss_travel_progress(progress, rush,
                                              abs(end[0] - start[0]) + abs(end[1] - start[1]))
                boss_position = (start[0] + (end[0] - start[0]) * travel,
                                 start[1] + (end[1] - start[1]) * travel)
                if "chrono_retreat" in events or any(event.startswith("chrono_leap:") for event in events):
                    boss_position = (boss_position[0], boss_position[1] - .9 * math.sin(math.pi * travel))
                if rush and travel < 1:
                    origin = (round((start[0] + .5) * self.CELL), round((start[1] + .5) * self.CELL))
                    current = (round((boss_position[0] + .5) * self.CELL),
                               round((boss_position[1] + .5) * self.CELL))
                    trail = (255, 105, 68) if isinstance(env.boss, SiegeLeviathan) else (196, 128, 255)
                    pg.draw.line(self.screen, trail, origin, current, 5)
                    pg.draw.circle(self.screen, trail, origin, 13, 2)
            furnace = isinstance(env.boss, FurnaceHydra)
            storm = isinstance(env.boss, StormChoir)
            chrono = isinstance(env.boss, ChronoMantis)
            void = isinstance(env.boss, VoidAngler)
            iron = isinstance(env.boss, IronGardener)
            mirror = isinstance(env.boss, MirrorSeraph)
            siege = isinstance(env.boss, SiegeLeviathan)
            null = isinstance(env.boss, NullWeaver)
            apex = isinstance(env.boss, ApexArbiter)
            if move and (null and "null_warp" in events or void and "void_warp" in events):
                boss_position = (int(move[1]), int(move[2])) if progress < .5 else env.boss.position
                for cell in ((int(move[1]), int(move[2])), env.boss.position):
                    center = (round((cell[0] + .5) * self.CELL), round((cell[1] + .5) * self.CELL))
                    self.pg.draw.circle(self.screen, (222, 159, 255) if chrono else
                                        (170, 113, 251) if void else (106, 237, 255), center, 18, 3)
            pulse = pg.time.get_ticks() / 180
            guarded = env.round < env.config.spawn_protection_rounds
            casting = bool(getattr(env.boss, "target", None) or
                           getattr(env.boss, "copied_action", None) or
                           getattr(env.boss, "rail_target", None) is not None or
                           getattr(env.boss, "erase_countdown", 0) or
                           getattr(env.boss, "countdown", 0))
            sway = .38 if guarded else .22 if casting and not move else 0
            boss_draw_position = (boss_position[0] + sway * math.sin(pulse * .55),
                                  boss_position[1] + .09 * math.sin(pulse))
            if storm and (env.boss.target or env.boss.overdrive_rounds or
                          any(event.startswith(("storm_chain:", "storm_surge:"))
                                                  for event in events)):
                sprite = self.storm_cast_frames[(pg.time.get_ticks() // 80) % 4]
                center = ((boss_draw_position[0] + .5) * self.CELL,
                          (boss_draw_position[1] + .5) * self.CELL)
                self.screen.blit(sprite, sprite.get_rect(center=center))
            else:
                self._sprite(boss_draw_position, "boss_apex_arbiter" if apex else
                             "boss_null_weaver" if null else
                             "boss_siege_leviathan" if siege else
                             "boss_mirror_seraph" if mirror else
                             "boss_iron_gardener" if iron else
                             "boss_void_angler" if void else
                             "boss_chrono_mantis" if chrono else
                             "boss_storm_choir" if storm else
                             "boss_furnace_hydra" if furnace else "boss_prism_warden")
            if storm and env.boss.overdrive_rounds:
                center = (round((boss_draw_position[0] + .5) * self.CELL),
                          round((boss_draw_position[1] + .5) * self.CELL))
                pg.draw.circle(self.screen, (107, 231, 255), center,
                               32 + round(4 * math.sin(pulse * 2)), 3)
            if "boss_shield_break" in events:
                self._sprite(boss_position, "effect_boss_law_convergence" if apex else
                             "effect_boss_grid_fracture" if null else
                             "effect_boss_railgun" if siege else
                             "effect_boss_mirror_shards" if mirror else
                             "effect_boss_plasma_thorns" if iron else
                             "effect_boss_gravity_vortex" if void else
                             "effect_boss_temporal_slash" if chrono else
                             "effect_boss_chain_lightning" if storm else
                             "effect_boss_magma_wave" if furnace else "effect_boss_prism_burst")
            if siege and any(event.startswith("rail_fire:") for event in events):
                target = next(event.split(":") for event in events if event.startswith("rail_fire:"))
                self._sprite((int(target[2]) if target[1] == "v" else 10,
                              10 if target[1] == "v" else int(target[2])), "effect_boss_railgun")
            if apex:
                fired = next((event.split(":") for event in events
                              if event.startswith("apex_fire:")), None)
                if fired:
                    effect = {"cage": "effect_boss_magma_wave", "barrage": "effect_boss_chain_lightning",
                              "charge": "effect_boss_mirror_shards", "gravity": "effect_boss_gravity_vortex",
                              "verdict": "effect_boss_law_convergence",
                              "cage_barrage": "effect_boss_law_convergence",
                              "charge_gravity": "effect_boss_law_convergence"}[fired[1]]
                    if fired[1] in ("barrage", "cage_barrage"):
                        for x, y in env.boss.fired_barrage_cells:
                            if env.boss.fired_barrage_volley == 2:
                                origin = (x + 1, y)
                            else:
                                origin = env.boss.position
                            self._sprite((origin[0] + (x - origin[0]) * progress,
                                          origin[1] + (y - origin[1]) * progress),
                                         "projectile_boss_barrage")
                        if fired[1] == "cage_barrage":
                            self._sprite((int(fired[2]), int(fired[3])), effect)
                    else:
                        self._sprite((int(fired[2]), int(fired[3])), effect)
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
            elif any(event.startswith("void_pull:") for event in events):
                pull = next(event.split(":") for event in events if event.startswith("void_pull:"))
                player_position = (int(pull[1]) + (int(pull[3]) - int(pull[1])) * eased,
                                   int(pull[2]) + (int(pull[4]) - int(pull[2])) * eased)
            elif any(event.startswith("void_hook:") for event in events):
                hook = next(event.split(":") for event in events if event.startswith("void_hook:"))
                player_position = (int(hook[1]) + (int(hook[3]) - int(hook[1])) * eased,
                                   int(hook[2]) + (int(hook[4]) - int(hook[2])) * eased)
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
            if event.startswith("boss_prism_followup:"):
                _, x1, y1, x2, y2 = event.split(":")
                self._projectile_effect((int(x1), int(y1)), (int(x2), int(y2)), progress,
                                        "projectile_boss_prism", (255, 92, 190))
                continue
            if event.startswith("boss_cover_break:"):
                _, x, y = event.split(":")
                self._sprite((int(x), int(y)), "effect_boss_prism_burst")
                continue
            if event.startswith("prism_reflector_shove:"):
                _, _, _, x, y = event.split(":")
                self._sprite((int(x), int(y)), "effect_boss_prism_burst")
                continue
            if event.startswith("furnace_fireball:"):
                parts = event.split(":")
                self._projectile_effect((int(parts[1]), int(parts[2])),
                                        (int(parts[3]), int(parts[4])), progress,
                                        "projectile_boss_fireball", (255, 110, 40))
                continue
            if event.startswith("void_mine:"):
                _, x, y = event.split(":")
                self._sprite((int(x), int(y)), "effect_boss_gravity_vortex")
                continue
            if event.startswith("void_node_cut:"):
                x, y = (int(value) for value in event.split(":")[1:])
                self._sprite((x, y), "effect_boss_gravity_vortex")
                continue
            if event.startswith("void_beam:"):
                _, x1, y1, x2, y2 = event.split(":")
                start = ((int(x1) + .5) * self.CELL, (int(y1) + .5) * self.CELL)
                end = ((int(x2) + .5) * self.CELL, (int(y2) + .5) * self.CELL)
                pg.draw.line(self.screen, (187, 117, 255), start, end, max(2, round(9 * progress)))
                self._sprite((int(x2), int(y2)), "effect_boss_gravity_vortex")
                continue
            if event.startswith("void_hook_fire:"):
                _, x1, y1, x2, y2 = event.split(":")
                start = ((int(x1) + .5) * self.CELL, (int(y1) + .5) * self.CELL)
                end = ((int(x2) + .5) * self.CELL, (int(y2) + .5) * self.CELL)
                pg.draw.line(self.screen, (226, 136, 255), start, end, 6)
                pg.draw.circle(self.screen, (255, 220, 255), end, 10, 3)
                self._sprite((int(x2), int(y2)), "effect_boss_gravity_vortex")
                continue
            if event.startswith("iron_flame:"):
                x = int(event.split(":")[1])
                for y in range(7, 17):
                    if progress >= (y - 7) / 12:
                        center = (x * self.CELL + self.CELL // 2,
                                  y * self.CELL + self.CELL // 2)
                        pg.draw.circle(self.screen, (255, 145, 66), center, 12, 3)
                continue
            if event.startswith("iron_thorn:"):
                _, x, y = event.split(":")
                self._sprite((int(x), int(y)), "effect_boss_plasma_thorns")
                continue
            if event.startswith("iron_bloom:"):
                _, x, y = event.split(":")
                self._sprite((int(x), int(y)), "effect_boss_plasma_thorns")
                continue
            if event.startswith("iron_vine_cut:"):
                x, y = (int(value) for value in event.split(":")[1:])
                self._sprite((x, y), "effect_boss_plasma_thorns")
                continue
            if event.startswith("furnace_wave:"):
                parts = event.split(":")
                columns = tuple(map(int, parts[2].split(","))) if len(parts) > 2 else (int(parts[1]),)
                for y in range(8, 17):
                    if progress >= (y - 8) / 13:
                        for x in columns:
                            self._sprite((x, y), "effect_boss_magma_wave")
                continue
            if event.startswith("furnace_valve_strike:"):
                x = int(event.split(":")[1])
                self._sprite((x, 12), "effect_boss_magma_wave")
                continue
            if event.startswith("storm_chain:"):
                values = [int(value) for value in event.split(":")[1:]]
                nodes = list(zip(values[::2], values[1::2]))
                for index, (start, end) in enumerate(zip(nodes, nodes[1:])):
                    local = progress * (len(nodes) - 1) - index
                    if local <= 0:
                        break
                    finish = (start[0] + (end[0] - start[0]) * min(1, local),
                              start[1] + (end[1] - start[1]) * min(1, local))
                    a = (round((start[0] + .5) * self.CELL), round((start[1] + .5) * self.CELL))
                    b = (round((finish[0] + .5) * self.CELL), round((finish[1] + .5) * self.CELL))
                    pg.draw.line(self.screen, (55, 158, 255), a, b, 11)
                    pg.draw.line(self.screen, (218, 250, 255), a, b, 4)
                    if 0 < local < 1:
                        self._sprite(finish, "effect_boss_chain_lightning")
                continue
            if event.startswith("storm_surge:"):
                x, y = (int(value) for value in event.split(":")[1:])
                self._sprite((x, y), "effect_boss_chain_lightning")
                continue
            if event.startswith("storm_net_fire:"):
                x, y = (int(value) for value in event.split(":")[1:])
                for dx, dy in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)):
                    if env.in_bounds((x + dx, y + dy)):
                        self._sprite((x + dx, y + dy), "effect_boss_chain_lightning")
                continue
            if event.startswith("chrono_slash:") or event.startswith("chrono_echo:"):
                x, y = (int(value) for value in event.split(":")[1:])
                self._sprite((x, y), "effect_boss_temporal_slash")
                continue
            if event.startswith("chrono_retreat_burst:"):
                x, y = (int(value) for value in event.split(":")[1:])
                for dx in (-1, 0, 1):
                    if env.in_bounds((x + dx, y)):
                        self._sprite((x + dx, y), "effect_boss_temporal_slash")
                continue
            if event.startswith("chrono_leap:"):
                x, y = (int(value) for value in event.split(":")[1:])
                self._sprite((x, y), "effect_boss_temporal_slash")
                continue
            if event.startswith("chrono_anchor_prime:"):
                x, y = (int(value) for value in event.split(":")[1:])
                self._sprite((x, y), "effect_boss_temporal_slash")
                continue
            if event.startswith("mirror_echo:"):
                x, y = (int(value) for value in event.split(":")[1:])
                for cell in env.mirror_echo_cells((x, y)):
                    self._sprite(cell, "effect_boss_mirror_shards")
                continue
            if event.startswith("mirror_shard:"):
                _, _, x, y = event.split(":")
                self._sprite((int(x), int(y)), "effect_boss_mirror_shards")
                continue
            if event.startswith("mirror_clone_strike:"):
                self._sprite(env.player.position, "effect_boss_mirror_shards")
                continue
            if event.startswith("siege_blast:"):
                _, kind, x, y = event.split(":")
                x, y = int(x), int(y)
                for cell in env.boss.blast_cells or env.siege_blast_cells((x, y), kind):
                    self._sprite(cell, "effect_boss_railgun")
                continue
            if event.startswith("void_pulse:"):
                _, x, y = event.split(":")
                x, y = int(x), int(y)
                for dx in range(-1, 2):
                    for dy in range(-1, 2):
                        if abs(dx) + abs(dy) <= 1 and env.in_bounds((x + dx, y + dy)):
                            self._sprite((x + dx, y + dy), "effect_boss_gravity_vortex")
                continue
            if event.startswith("siege_charge_recoil:"):
                x = int(event.split(":")[1])
                self._sprite((x, 7), "effect_boss_railgun")
                continue
            if event == "apex_appeal_mark":
                self._sprite(env.boss.position, "effect_boss_law_convergence")
                continue
            if event.startswith("null_fracture:"):
                for cell in env.null_void:
                    self._sprite(cell, "effect_boss_grid_fracture")
                continue
            if event.startswith("null_node_strike:"):
                x, y = (int(value) for value in event.split(":")[1:])
                self._sprite((x, y), "effect_boss_grid_fracture")
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
            if isinstance(env.boss, ChronoMantis):
                phase = {"flank": "横切追击", "slash": "近斩锁定", "leap": "跃迁蓄力"}[env.boss.phase]
                state = (f"核心开放 {env.boss.exposed_rounds} 回合" if env.boss.exposed_rounds else
                         f"{phase} · 倒计时 {env.boss.leap_countdown}" if env.boss.phase == "leap" else phase)
                label, tint = "时序螳螂", (230, 182, 255)
            elif isinstance(env.boss, VoidAngler):
                state = (f"核心开放 {env.boss.exposed_rounds} 回合" if env.boss.exposed_rounds else
                         f"装甲吸离 {len(env.boss.drained_nodes)}/3")
                label, tint = "虚空钓手", (199, 155, 255)
            elif isinstance(env.boss, IronGardener):
                state = (f"核心开放 {env.boss.exposed_rounds} 回合" if env.boss.exposed_rounds else
                         f"热回流 {len(env.boss.refluxed_roots)}/4")
                label, tint = "钢铁园丁", (169, 234, 139)
            elif isinstance(env.boss, MirrorSeraph):
                state = (f"核心开放 {env.boss.exposed_rounds} 回合" if env.boss.exposed_rounds else
                         f"镜锁 {len(env.boss.broken_locks)}/3")
                label, tint = "镜像炽天使", (255, 174, 237)
            elif isinstance(env.boss, SiegeLeviathan):
                state = (f"核心开放 {env.boss.exposed_rounds} 回合" if env.boss.exposed_rounds else
                         f"装甲锁 {len(env.boss.broken_locks)}/4")
                label, tint = "攻城利维坦", (255, 193, 122)
            elif isinstance(env.boss, NullWeaver):
                state = (f"核心开放 {env.boss.exposed_rounds} 回合" if env.boss.exposed_rounds else
                         f"逻辑节点 {env.boss.node_index}/4")
                label, tint = "归零织机", (160, 227, 255)
            elif isinstance(env.boss, ApexArbiter):
                state = (f"核心开放 {env.boss.exposed_rounds} 回合" if env.boss.exposed_rounds else
                         f"四印 {env.boss.seals}/4")
                label, tint = "顶点裁决者", (255, 222, 151)
            elif isinstance(env.boss, StormChoir):
                state = (f"高速充能 {env.boss.overdrive_rounds} 回合" if env.boss.overdrive_rounds else
                         f"疲惫 {env.boss.fatigue_rounds} 回合" if env.boss.fatigue_rounds else
                         f"核心开放 {env.boss.exposed_rounds} 回合" if env.boss.exposed_rounds else
                         "接地柱 0/4" if env.boss.target is None else
                         f"雷链 {max(0, len(env.storm_chain()) - 2)}/4 柱")
                label, tint = "风暴合唱环", (132, 211, 255)
            elif isinstance(env.boss, FurnaceHydra):
                state = (f"核心开放 {env.boss.exposed_rounds} 回合" if env.boss.exposed_rounds else
                         f"冷却阀 {len(env.boss.valves_opened)}/3")
                label, tint = "熔炉三头机", (255, 171, 104)
            else:
                state = (f"核心开放 {env.boss.exposed_rounds} 回合" if env.boss.exposed_rounds else
                         f"镜面反射 {env.boss.reflections}/3")
                label, tint = "棱镜守卫", (244, 164, 255)
            self._text(f"{label}  HP {env.boss.hp}/{env.boss.max_hp}  {state}",
                       left, 265, tint, small=True)
            if isinstance(env.boss, PrismWarden):
                self._text(f"镜组停摆 {env.boss.reflector_lockout}  ·  灰色镜子仍在冷却",
                           left, 289, (226, 197, 245), small=True)
            if isinstance(env.boss, FurnaceHydra):
                self._text("已用阀门会发热；发红后离开", left, 289,
                           (255, 191, 142), small=True)
            if isinstance(env.boss, VoidAngler):
                self._text("已被钩住：下一动作不能移动，可近战／射击" if env.hooked_actions else
                           "钩锁瞄准时离开紫格；命中会被拉近", left, 289,
                           (220, 174, 255), small=True)
            if isinstance(env.boss, MirrorSeraph):
                self._text(f"镜锁 {len(env.boss.broken_locks)}/3 · 先借 Boss 射线破锁",
                           left, 289, (255, 210, 242), small=True)
                if env.boss.exposed_rounds:
                    self._text("镜翼蓄势：远射可能落空；近身 EMP 可干扰" if
                               env.boss.evade_ready and not env.boss.emp_jammed else
                               "护盾已破：现在攻击 Boss 本体！", left, 312,
                               (179, 255, 200), small=True)
                else:
                    self._text("右→左锁  左→右锁  下→下锁", left, 312,
                               (255, 210, 242), small=True)
                if env.boss.copied_action:
                    directions = {"n": "上", "s": "下", "e": "右", "w": "左"}
                    original = directions.get(env.boss.copied_action[-1], "—")
                    mirrored = directions.get(env.boss.mirrored_direction, "—")
                    self._text(f"正在复制你的{original}动作：向{mirrored}发射", left, 335,
                               (255, 235, 253), small=True)
                if env.boss.echo_target:
                    self._text(f"紫色十字下轮爆炸：{env.boss.echo_damage} 伤害", left, 358,
                               (251, 167, 242), small=True)
            if isinstance(env.boss, SiegeLeviathan):
                self._text(f"装甲锁 {len(env.boss.broken_locks)}/4 · 炮击掩体破锁",
                           left, 289, (255, 214, 168), small=True)
                if env.boss.rail_target is not None:
                    lane = "列" if env.boss.rail_axis == "v" else "行"
                    self._text(f"轨道炮：{lane} {env.boss.rail_target} · 蓄力 {env.boss.charge}",
                               left, 312, (255, 214, 168), small=True)
                else:
                    self._text("站掩体后引导炮线；破盾后攻击本体", left, 312,
                               (255, 214, 168), small=True)
            if isinstance(env.boss, NullWeaver) and not env.boss.exposed_rounds:
                labels = {"move": "普通移动", "melee": "近战", "ranged": "远程", "skill": "技能"}
                blocked = labels.get(env.boss.blocked_kind, "无")
                self._text(f"本轮封锁：{blocked} · 下个节点 {min(4, env.boss.node_index + 1)}",
                           left, 289, (183, 223, 255), small=True)
                if env.boss.erase_targets:
                    self._text(f"删格倒计时：{env.boss.erase_countdown}", left, 312,
                               (249, 170, 215), small=True)
            if isinstance(env.boss, ApexArbiter) and not env.boss.exposed_rounds:
                names = {"cage": "熔锁牢笼", "barrage": "雷幕弹雨", "charge": "镜面冲撞",
                         "gravity": "坍缩漩涡", "verdict": "终审判词",
                         "cage_barrage": "熔锁雷幕", "charge_gravity": "镜冲引力"}
                self._text(f"当前法则：{names[env.boss.kind]} · 预警 {env.boss.countdown}",
                           left, 289, (255, 215, 167), small=True)
                if env.boss.kind == "verdict" and env.boss.countdown:
                    self._text("踩白色上诉位反弹判词；黄格会受伤", left, 312,
                               (219, 255, 210), small=True)
                elif env.boss.kind == "barrage" and env.boss.barrage_volley == 2:
                    self._text("第二波弹幕：前一波安全格现已危险", left, 312,
                               (211, 233, 255), small=True)
                elif env.boss.kind == "cage_barrage" and env.boss.countdown:
                    self._text("打碎白门后找安全格；第二波将变向", left, 312,
                               (211, 233, 255), small=True)
                elif env.boss.kind == "charge_gravity" and env.boss.countdown:
                    self._text("紫色冲撞＋绿色爆心：离开两区", left, 312,
                               (219, 255, 210), small=True)
        self._text(f"智能体：{AGENT_NAMES.get(agent, agent)}", left, 55, colors["muted"])
        self._text(f"动作：{ACTION_NAMES.get(action, action)}", left, 80, colors["text"])
        self._text(f"推理耗时：{latency_ms:.1f} 毫秒", left, 105, colors["muted"])
        self._text(f"决策间隔：{decision_ms} 毫秒", left, 130, colors["muted"], small=True)
        if selection_reason:
            self._text(f"决策依据：{REASON_NAMES.get(selection_reason, selection_reason)}", left, 152,
                       colors["muted"], small=True)
        difficulty = (f"难度：第 {level} 关 Boss · 机制破盾" if env.boss else
                      f"难度：敌 {env.config.enemies}  火 {env.config.fires}  刺 {env.config.spikes}  "
                      f"坑 {env.config.pits}  敌生命 +{env.config.enemy_hp_bonus}")
        self._text(difficulty, left, 174, colors["muted"], small=True)
        dash_cd = env.player.cooldowns.get("dash", 0)
        emp_cd = env.player.cooldowns.get("emp", 0)
        self._text(f"技能：冲刺 {'就绪' if not dash_cd else dash_cd}  EMP {'就绪' if not emp_cd else emp_cd}",
                   left, 196, colors["selected"] if not dash_cd and not emp_cd else colors["muted"], small=True)
        loadout = env.player.loadout
        self._text(f"武器：弓 {'未获得' if not loadout.bow else f'{loadout.arrows} 箭'}  "
                   f"手枪 {'未获得' if not loadout.pistol else f'{loadout.energy} 发'}",
                   left, 218, colors["muted"], small=True)
        y = 390 if isinstance(env.boss, MirrorSeraph) else 365 if env.boss else 245
        for name, probability in sorted(probabilities.items(), key=lambda item: item[1], reverse=True)[:7 if env.boss else 11]:
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
                     "enemy_razor_hound",
                     "enemy_furnace_hatchling", "enemy_vine_hunter",
                     "gem", "fire", "medkit", "wall", "item_bow", "item_pulse_pistol",
                     "ammo_arrows", "ammo_energy_cell", "barrel", "spike", "pit",
                     "projectile_enemy_laser", "projectile_player_pulse", "projectile_player_arrow",
                     "effect_player_slash", "effect_enemy_claw", "boss_prism_warden",
                     "projectile_boss_prism", "effect_boss_prism_burst", "boss_furnace_hydra",
                     "projectile_boss_fireball", "effect_boss_magma_wave", "boss_storm_choir",
                     "effect_boss_chain_lightning", "boss_chrono_mantis",
                     "effect_boss_temporal_slash", "boss_void_angler",
                     "effect_boss_gravity_vortex", "boss_iron_gardener",
                     "effect_boss_plasma_thorns", "boss_mirror_seraph",
                     "effect_boss_mirror_shards", "boss_siege_leviathan",
                     "effect_boss_railgun", "boss_null_weaver",
                     "effect_boss_grid_fracture", "boss_apex_arbiter", "projectile_boss_barrage",
                     "effect_boss_law_convergence"):
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
            elif name in ("boss_storm_choir", "effect_boss_chain_lightning"):
                limit = round(self.CELL * (3.0 if name == "boss_storm_choir" else 1.6))
            elif name in ("boss_chrono_mantis", "effect_boss_temporal_slash"):
                limit = round(self.CELL * (3.0 if name == "boss_chrono_mantis" else 2.0))
            elif name in ("boss_void_angler", "effect_boss_gravity_vortex"):
                limit = round(self.CELL * (3.2 if name == "boss_void_angler" else 2.2))
            elif name in ("boss_iron_gardener", "effect_boss_plasma_thorns"):
                limit = round(self.CELL * (3.2 if name == "boss_iron_gardener" else 2.0))
            elif name in ("boss_mirror_seraph", "effect_boss_mirror_shards"):
                limit = round(self.CELL * (3.2 if name == "boss_mirror_seraph" else 2.1))
            elif name in ("boss_siege_leviathan", "effect_boss_railgun"):
                limit = round(self.CELL * (4.2 if name == "boss_siege_leviathan" else 2.2))
            elif name in ("boss_null_weaver", "effect_boss_grid_fracture"):
                limit = round(self.CELL * (3.8 if name == "boss_null_weaver" else 2.3))
            elif name in ("boss_apex_arbiter", "effect_boss_law_convergence"):
                limit = round(self.CELL * (4.2 if name == "boss_apex_arbiter" else 2.5))
            elif name == "projectile_boss_barrage":
                limit = round(self.CELL * .8)
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
        if not intent or intent.kind in (IntentType.MOVE, IntentType.SPRINT, IntentType.WAIT):
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
        if intent.kind == IntentType.SPRINT:
            points = (enemy.position,) + intent.path
            centers = [(x * self.CELL + self.CELL // 2, y * self.CELL + self.CELL // 2)
                       for x, y in points]
            if len(centers) > 1:
                self.pg.draw.lines(self.screen, (76, 210, 231), False, centers, 3)
                for index, (x, y) in enumerate(intent.path, 1):
                    rect = self.pg.Rect(x * self.CELL + 4, y * self.CELL + 4,
                                        self.CELL - 8, self.CELL - 8)
                    self.pg.draw.rect(self.screen, (76, 210, 231), rect, 2 + int(index == len(intent.path)),
                                      border_radius=6)
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
        if isinstance(env.boss, ApexArbiter):
            if not env.boss.countdown:
                return
            overlay = self.pg.Surface(self.screen.get_size(), self.pg.SRCALPHA)
            palette = {"cage": (255, 137, 63), "barrage": (98, 197, 255),
                       "charge": (222, 133, 255), "gravity": (112, 237, 157),
                       "verdict": (255, 216, 112), "cage_barrage": (98, 197, 255),
                       "charge_gravity": (222, 133, 255)}
            color = palette[env.boss.kind]
            for x, y in env.boss.danger:
                self.pg.draw.rect(overlay, (*color, 115 if env.boss.countdown == 1 else 65),
                                  (x * self.CELL + 2, y * self.CELL + 2,
                                   self.CELL - 4, self.CELL - 4), border_radius=4)
            if env.boss.gate:
                x, y = env.boss.gate
                self.pg.draw.rect(overlay, (255, 245, 196, 220),
                                  (x * self.CELL + 3, y * self.CELL + 3,
                                   self.CELL - 6, self.CELL - 6), 3, border_radius=4)
            if env.boss.appeal:
                x, y = env.boss.appeal
                center = (x * self.CELL + self.CELL // 2, y * self.CELL + self.CELL // 2)
                self.pg.draw.circle(overlay, (248, 255, 220, 240), center, 14, 4)
                self.pg.draw.circle(overlay, (128, 255, 216, 200), center, 7)
            if env.boss.kind == "charge_gravity" and env.boss.target:
                x, y = env.boss.target
                center = (x * self.CELL + self.CELL // 2, y * self.CELL + self.CELL // 2)
                self.pg.draw.circle(overlay, (128, 255, 169, 210), center, self.CELL + 3, 3)
            self.screen.blit(overlay, (0, 0))
            return
        if isinstance(env.boss, NullWeaver):
            if not env.boss.erase_targets and not env.boss.warp_target:
                return
            overlay = self.pg.Surface(self.screen.get_size(), self.pg.SRCALPHA)
            urgent = env.boss.erase_countdown == 1
            for x, y in env.boss.erase_targets:
                rect = (x * self.CELL + 2, y * self.CELL + 2,
                        self.CELL - 4, self.CELL - 4)
                self.pg.draw.rect(overlay, (255, 84, 144, 150) if urgent else (133, 177, 255, 95),
                                  rect, border_radius=5)
                self.pg.draw.rect(overlay, (255, 212, 230, 230) if urgent else (205, 231, 255, 190),
                                  rect, 2, border_radius=5)
                if (x, y) in env.boss.fracture_cells:
                    self.pg.draw.line(overlay, (90, 9, 118, 245),
                                      (rect[0] + 5, rect[1] + 5),
                                      (rect[0] + rect[2] - 5, rect[1] + rect[3] - 5), 4)
            if env.boss.warp_target:
                x, y = env.boss.warp_target
                center = (x * self.CELL + self.CELL // 2, y * self.CELL + self.CELL // 2)
                self.pg.draw.circle(overlay, (91, 236, 255, 215), center, self.CELL // 2 - 3, 3)
                self.pg.draw.circle(overlay, (211, 255, 255, 240), center, 8, 2)
            self.screen.blit(overlay, (0, 0))
            return
        if isinstance(env.boss, SiegeLeviathan):
            path = env.rail_path()
            blast = env.siege_blast_cells()
            charge = env.siege_charge_path()
            if not path and not blast and not charge:
                return
            overlay = self.pg.Surface(self.screen.get_size(), self.pg.SRCALPHA)
            alpha = 80 if env.boss.charge == 2 else 145
            cover = env.rail_cover()
            for x, y in path:
                safe = cover and (y > cover[1] if env.boss.rail_axis == "v" else x > cover[0])
                color = (107, 215, 223, 82) if safe else (255, 147, 68, alpha)
                self.pg.draw.rect(overlay, color,
                                  (x * self.CELL + 2, y * self.CELL + 2,
                                   self.CELL - 4, self.CELL - 4), border_radius=5)
            for x, y in blast:
                rect = (x * self.CELL + 2, y * self.CELL + 2,
                        self.CELL - 4, self.CELL - 4)
                self.pg.draw.rect(overlay, (255, 74, 40, 135), rect, border_radius=5)
                self.pg.draw.rect(overlay, (255, 230, 110, 230), rect, 2, border_radius=5)
            for x, y in charge:
                rect = (x * self.CELL + 2, y * self.CELL + 2,
                        self.CELL - 4, self.CELL - 4)
                self.pg.draw.rect(overlay, (255, 72, 88, 150), rect, border_radius=5)
                self.pg.draw.rect(overlay, (255, 240, 180, 230), rect, 2, border_radius=5)
            self.screen.blit(overlay, (0, 0))
            return
        if isinstance(env.boss, MirrorSeraph):
            path = env.mirror_ray()
            clones = [enemy for enemy in env.enemies if enemy.summoned_by and
                      enemy.summoned_by.startswith("mirror_")]
            if not path and not env.boss.echo_target and not env.boss.rush_target and not clones:
                return
            overlay = self.pg.Surface(self.screen.get_size(), self.pg.SRCALPHA)
            if path:
                locked = path[-1] in env.mirror_locks and path[-1] not in env.boss.broken_locks
                ray_cells = set(path) if locked else {
                    cell for point in path for cell in
                    ((point[0] + dx, point[1] + dy) for dx, dy in
                     ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1))) if env.in_bounds(cell)}
                for x, y in ray_cells:
                    self.pg.draw.rect(overlay, (255, 174, 235, 92) if locked else (255, 91, 165, 125),
                                      (x * self.CELL + 2, y * self.CELL + 2,
                                       self.CELL - 4, self.CELL - 4), border_radius=5)
                end = path[-1]
                self.pg.draw.line(overlay, (255, 235, 253, 210),
                                  ((env.boss.position[0] + .5) * self.CELL,
                                   (env.boss.position[1] + .5) * self.CELL),
                                  ((end[0] + .5) * self.CELL, (end[1] + .5) * self.CELL), 3)
            if env.boss.echo_target:
                x, y = env.boss.echo_target
                for cell in env.mirror_echo_cells():
                    area = (cell[0] * self.CELL + 2, cell[1] * self.CELL + 2,
                            self.CELL - 4, self.CELL - 4)
                    self.pg.draw.rect(overlay, (209, 71, 238, 85), area, border_radius=5)
                    self.pg.draw.rect(overlay, (251, 177, 255, 210), area, 2, border_radius=5)
                self.pg.draw.circle(overlay, (255, 223, 255, 240),
                                    (int((x + .5) * self.CELL), int((y + .5) * self.CELL)), 6, 2)
            for x, y in env.mirror_rush_cells():
                self.pg.draw.rect(overlay, (255, 69, 123, 150),
                                  (x * self.CELL + 2, y * self.CELL + 2,
                                   self.CELL - 4, self.CELL - 4), border_radius=5)
            for clone in clones:
                cx, cy = clone.position
                self.pg.draw.circle(overlay, (181, 92, 241, 135),
                                    (int((cx + .5) * self.CELL), int((cy + .5) * self.CELL)),
                                    self.CELL * 2, 2)
            self.screen.blit(overlay, (0, 0))
            return
        if isinstance(env.boss, IronGardener):
            boss = env.boss
            if boss.target is None:
                return
            overlay = self.pg.Surface(self.screen.get_size(), self.pg.SRCALPHA)
            if boss.attack_kind == "flame":
                cells = [(boss.target[0], y) for y in range(8, 17)]
            else:
                x, y = boss.target
                radius = 2 if boss.attack_kind == "bloom" else 1
                cells = [(x + dx, y + dy) for dx in range(-radius, radius + 1)
                         for dy in range(-radius, radius + 1) if abs(dx) + abs(dy) <= radius]
            for x, y in cells:
                if not env.in_bounds((x, y)):
                    continue
                safe = (boss.attack_kind == "flame" and (x, y) == boss.target and
                        env.iron_root_ready(x))
                self.pg.draw.rect(overlay, (95, 236, 166, 125) if safe else (255, 131, 76, 115),
                                  (x * self.CELL + 2, y * self.CELL + 2,
                                   self.CELL - 4, self.CELL - 4), border_radius=5)
            start = ((boss.position[0] + .5) * self.CELL, (boss.position[1] + .5) * self.CELL)
            end = ((boss.target[0] + .5) * self.CELL, (boss.target[1] + .5) * self.CELL)
            self.pg.draw.line(overlay, (255, 197, 100, 220), start, end, 3)
            self.screen.blit(overlay, (0, 0))
            return
        if isinstance(env.boss, VoidAngler):
            boss = env.boss
            if boss.target is None:
                return
            overlay = self.pg.Surface(self.screen.get_size(), self.pg.SRCALPHA)
            radius = 2 if boss.attack_kind == "mine" else 1
            for dx in range(-radius, radius + 1):
                for dy in range(-radius + abs(dx), radius - abs(dx) + 1):
                    x, y = boss.target[0] + dx, boss.target[1] + dy
                    if env.in_bounds((x, y)):
                        color = ((155, 104, 241, 90) if boss.attack_kind == "mine" else
                                 (255, 80, 139, 120))
                        self.pg.draw.rect(overlay, color,
                                          (x * self.CELL + 2, y * self.CELL + 2,
                                           self.CELL - 4, self.CELL - 4), border_radius=5)
            target = boss.target
            center = ((target[0] + .5) * self.CELL, (target[1] + .5) * self.CELL)
            start = ((boss.position[0] + .5) * self.CELL, (boss.position[1] + .5) * self.CELL)
            self.pg.draw.line(overlay, (224, 155, 255, 205), start, center, 3)
            if boss.attack_kind == "mine" and target in env.gravity_nodes - boss.drained_nodes:
                self.pg.draw.circle(overlay, (145, 255, 212, 240), center, 19, 3)
            self.screen.blit(overlay, (0, 0))
            return
        if isinstance(env.boss, ChronoMantis):
            boss = env.boss
            overlay = self.pg.Surface(self.screen.get_size(), self.pg.SRCALPHA)
            if boss.retreat_target:
                target = boss.retreat_target
                cells = [(target[0] + dx, target[1]) for dx in (-1, 0, 1)]
                color = (212, 124, 255, 135)
            elif boss.phase == "slash" and boss.slash_target:
                target = boss.slash_target
                cells = [(target[0] + dx, target[1] + dy)
                         for dx, dy in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1))]
                color = (255, 105, 138, 125)
            elif boss.phase == "leap" and boss.leap_target:
                target = boss.leap_target
                cells = [target]
                color = (208, 130, 255, 125)
            else:
                return
            for cell in cells:
                if env.in_bounds(cell):
                    tint = ((255, 178, 84, 135) if boss.phase == "slash" else
                            (116, 239, 188, 135)) if cell in env.time_anchors else color
                    self.pg.draw.rect(overlay, tint,
                                      (cell[0] * self.CELL + 2, cell[1] * self.CELL + 2,
                                       self.CELL - 4, self.CELL - 4), border_radius=5)
            if boss.phase == "leap":
                if boss.slash_target:
                    echo = boss.slash_target
                    self.pg.draw.rect(overlay, (255, 105, 138, 115),
                                      (echo[0] * self.CELL + 3, echo[1] * self.CELL + 3,
                                       self.CELL - 6, self.CELL - 6), 3, border_radius=5)
                start = ((boss.position[0] + .5) * self.CELL, (boss.position[1] + .5) * self.CELL)
                end = ((target[0] + .5) * self.CELL, (target[1] + .5) * self.CELL)
                self.pg.draw.line(overlay, (243, 191, 255, 225), start, end, 4)
            self.screen.blit(overlay, (0, 0))
            return
        if isinstance(env.boss, StormChoir):
            boss = env.boss
            if boss.target is None and boss.net_target is None:
                return
            overlay = self.pg.Surface(self.screen.get_size(), self.pg.SRCALPHA)
            if boss.net_target:
                x, y = boss.net_target
                for dx, dy in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)):
                    if env.in_bounds((x + dx, y + dy)):
                        self.pg.draw.rect(overlay, (204, 80, 255, 135),
                                          ((x + dx) * self.CELL + 2, (y + dy) * self.CELL + 2,
                                           self.CELL - 4, self.CELL - 4), 3, border_radius=5)
            if boss.target is None:
                self.screen.blit(overlay, (0, 0))
                return
            nodes = env.storm_chain() if boss.attack_kind == "chain" else (boss.position, boss.target)
            for start, end in zip(nodes, nodes[1:]):
                a = (start[0] * self.CELL + self.CELL // 2, start[1] * self.CELL + self.CELL // 2)
                b = (end[0] * self.CELL + self.CELL // 2, end[1] * self.CELL + self.CELL // 2)
                self.pg.draw.line(overlay, (75, 173, 255, 55), a, b, 14)
                self.pg.draw.line(overlay, (157, 222, 255, 210), a, b, 3)
            target = boss.target
            safe = (boss.attack_kind == "chain" and target in env.relay_pads and
                    target != boss.last_ground_pad and len(nodes) == 6)
            if boss.attack_kind == "surge":
                for dx in range(-1, 2):
                    for dy in range(-1 + abs(dx), 2 - abs(dx)):
                        x, y = target[0] + dx, target[1] + dy
                        if env.in_bounds((x, y)):
                            self.pg.draw.rect(overlay, (255, 92, 116, 110),
                                              (x * self.CELL + 2, y * self.CELL + 2,
                                               self.CELL - 4, self.CELL - 4), border_radius=5)
            center = (target[0] * self.CELL + self.CELL // 2,
                      target[1] * self.CELL + self.CELL // 2)
            self.pg.draw.circle(overlay, (86, 235, 255, 220) if safe else (255, 105, 72, 225),
                                center, 17 if boss.attack_kind == "chain" else 28, 3)
            self.screen.blit(overlay, (0, 0))
            return
        if isinstance(env.boss, FurnaceHydra):
            boss = env.boss
            if boss.target is None:
                return
            overlay = self.pg.Surface(self.screen.get_size(), self.pg.SRCALPHA)
            if boss.attack_kind in ("wave", "triple"):
                columns = boss.wave_columns or ((boss.head_x - 1, boss.head_x, boss.head_x + 1)
                                                if boss.hp <= boss.max_hp // 2 else (boss.head_x,))
                cells = [(x, y) for x in columns for y in range(8, 17)]
            else:
                x, y = boss.target
                cells = [(x, y), (x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
            for x, y in cells:
                if not env.in_bounds((x, y)):
                    continue
                safe = (boss.attack_kind in ("wave", "triple") and not boss.wave_rapid and
                        (x, y) in env.coolant_valves and
                        x not in boss.valves_opened and
                        (boss.attack_kind == "triple" or x == boss.head_x))
                edge = boss.attack_kind == "wave" and x != boss.head_x
                self.pg.draw.rect(overlay, (70, 215, 250, 115) if safe else
                                  (255, 146, 70, 85) if edge else (255, 98, 34, 110),
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
        rays = env.prism_rays()
        if not rays or not env.boss:
            return
        path = rays[0]
        overlay = self.pg.Surface(self.screen.get_size(), self.pg.SRCALPHA)
        for cell in set().union(*rays):
            center = (cell[0] * self.CELL + self.CELL // 2,
                      cell[1] * self.CELL + self.CELL // 2)
            self.pg.draw.rect(overlay, (214, 65, 246, 75),
                              (cell[0] * self.CELL + 2, cell[1] * self.CELL + 2,
                               self.CELL - 4, self.CELL - 4), border_radius=5)
            self.pg.draw.circle(overlay, (255, 222, 255, 215), center, 4)
        if env.boss.sweep:
            for cell in env.prism_attack_cells() - set(path):
                self.pg.draw.rect(overlay, (255, 102, 205, 125),
                                  (cell[0] * self.CELL + 2, cell[1] * self.CELL + 2,
                                   self.CELL - 4, self.CELL - 4), 3, border_radius=5)
        start = (env.boss.position[0] * self.CELL + self.CELL // 2,
                 env.boss.position[1] * self.CELL + self.CELL // 2)
        end = (path[-1][0] * self.CELL + self.CELL // 2,
               path[-1][1] * self.CELL + self.CELL // 2)
        for ray in rays:
            if ray:
                end = (ray[-1][0] * self.CELL + self.CELL // 2,
                       ray[-1][1] * self.CELL + self.CELL // 2)
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
            elif event.startswith("hound_sprint:"): labels.append("迅猛兽沿预警路线冲刺！")
            elif event.startswith("archer_shot:"): labels.append("敌方能量激光！")
            elif event == "boss_aim": labels.append("棱镜守卫锁定目标！可推动未充能镜柱改反射线。")
            elif event == "boss_sweep_aim": labels.append("棱镜横扫预警：亮格也会受到伤害！")
            elif event == "boss_cover_aim": labels.append("棱镜守卫正在锁定可破坏掩体！")
            elif event == "boss_prism_phase_two": labels.append("棱镜守卫进入第二阶段！")
            elif event.startswith("boss_cover_break:"): labels.append("棱镜光束摧毁了掩体！")
            elif event.startswith("boss_prism_shot:"): labels.append("棱镜弹发射！")
            elif event.startswith("boss_reflect:"): labels.append("镜柱反射：护盾松动！")
            elif event.startswith("boss_prism_followup:"): labels.append("反射后追击光束！不要久站镜柱后！")
            elif event.startswith("boss_lunge_aim:"): labels.append("Boss 突进预警：躲开红色区域！")
            elif event == "boss_lunge": labels.append("Boss 突进！")
            elif event == "boss_shield_break": labels.append("护盾破裂！攻击核心！")
            elif event == "boss_shield": labels.append("护盾阻挡攻击")
            elif event.startswith("boss_hit:"): labels.append(f"核心受到 {event.split(':')[1]} 点伤害")
            elif event == "boss_defeated": labels.append("Boss 已击败！")
            elif event.startswith("furnace_aim:wave:"): labels.append("熔岩波预警：蓝色冷却阀可挡火！")
            elif event.startswith("furnace_aim:triple:"): labels.append("三炉头同时喷火！站冷却阀可回灌。")
            elif event.startswith("furnace_aim:fireball:"): labels.append("火球锁定：离开橙色落点！")
            elif event == "furnace_phase_two": labels.append("熔炉过热：熔岩波变宽，火球会点燃地面！")
            elif event.startswith("furnace_ignite:"): labels.append("火球留下短暂燃烧格！")
            elif event.startswith("furnace_wave:"): labels.append("熔岩波来袭！")
            elif event == "furnace_rapid_combo": labels.append("熔炉连发！这次冷却阀不安全，马上离开预警列！")
            elif event.startswith("furnace_fireball:"): labels.append("熔炉火球发射！")
            elif event.startswith("furnace_valve_strike:"): labels.append("从侧面敲开冷却阀：离开即将喷发的火线！")
            elif event.startswith("furnace_valve:"): labels.append("冷却阀反制成功！")
            elif event == "furnace_restock_arrows": labels.append("熔炉冷却，回收箭矢 +3！")
            elif event.startswith("storm_aim:chain:"): labels.append("连锁雷网预警：导电位接地，或靠近导电位用 EMP 短接！")
            elif event.startswith("storm_aim:surge:"): labels.append("高压雷爆锁定：离开目标周围！")
            elif event == "storm_overdrive_charge": labels.append("风暴合唱环自身充能：即将高速追击！")
            elif event.startswith("storm_overdrive_move:"): labels.append("高速乱流逼近，近战伤害较低！")
            elif event == "storm_overdrive_exhausted": labels.append("充能耗尽，进入短暂疲惫期！")
            elif event == "storm_phase_two": labels.append("风暴合唱环进入二阶段：雷爆更强！")
            elif event == "storm_relay_shift": labels.append("导电位移到两侧：去蓝色新落点！")
            elif event.startswith("storm_chain:"): labels.append("连锁闪电穿过接地柱！")
            elif event.startswith("storm_surge:"): labels.append("高压雷爆！")
            elif event.startswith("storm_net_place:"): labels.append("身边突然铺开雷网：下轮前离开紫色十字！")
            elif event.startswith("storm_net_fire:"): labels.append("雷网通电！")
            elif event == "storm_grounded": labels.append("四柱接地回灌：核心开放！")
            elif event == "storm_emp_ground": labels.append("EMP 短接雷链：核心短暂开放！")
            elif event.startswith("chrono_slash_aim:"): labels.append("时序近斩预警：离开红色范围！")
            elif event == "chrono_phase_two": labels.append("时序螳螂进入二阶段！")
            elif event == "chrono_anchor_shift": labels.append("时间锚移到后方通道！")
            elif event.startswith("chrono_slash:"): labels.append("时序螳螂近斩！")
            elif event == "chrono_anchor_guard": labels.append("时间锚缓冲近斩，仍会受伤；接跳可反制！")
            elif event.startswith("chrono_leap_aim:"): labels.append("跳杀锁定当前位置！离开紫格，或用时间锚反制！")
            elif event.startswith("chrono_leap_charge:"): labels.append("跃迁蓄力：落点不再改变！")
            elif event.startswith("chrono_leap:"): labels.append("时序螳螂跃迁！")
            elif event == "chrono_fatigued": labels.append("高速连招后疲惫，趁机调整站位！")
            elif event == "chrono_retreat": labels.append("螳螂拉开距离，准备残影扫射！")
            elif event.startswith("chrono_retreat_aim:"): labels.append("紫色三格残影已锁定：横向闪开！")
            elif event.startswith("chrono_retreat_burst:"): labels.append("残影扫射爆发！")
            elif event.startswith("damage:chrono_leap:"): labels.append("跳杀命中！")
            elif event == "chrono_anchor": labels.append("时间锚已启动！")
            elif event.startswith("chrono_anchor_prime:"): labels.append("时间锚已预置：离开落点，诱螳螂撞上残影！")
            elif event == "chrono_echo_replay": labels.append("残影回放击中 Boss！")
            elif event.startswith("null_node_strike:"): labels.append("近战激活逻辑节点，无需站到节点上！")
            elif event.startswith("void_aim:mine:"): labels.append("引力雷预警：站节点吸甲，或从邻格近战截断！")
            elif event.startswith("void_aim:beam:"): labels.append("虚空光束锁定：离开红色格！")
            elif event.startswith("void_aim:hook:"): labels.append("虚空钩锁定：离开紫色落点！")
            elif event.startswith("void_aim:pulse:"): labels.append("引力脉冲锁定：离开紫色区域，随后会接光束！")
            elif event.startswith("void_pulse:"): labels.append("引力脉冲爆发：靠近中心会被拉向 Boss！")
            elif event == "void_warp": labels.append("虚空钓手拉开距离，准备远程攻击！")
            elif event.startswith("void_mine:"): labels.append("引力雷爆发！")
            elif event.startswith("void_drain:"): labels.append("外层装甲被引力雷吸离！")
            elif event.startswith("void_pull:"): labels.append("被引力牵引！")
            elif event.startswith("void_hook:"): labels.append("被钩到 Boss 身边，下一动作不能移动！")
            elif event == "void_hook_release": labels.append("钩锁解除，可以移动！")
            elif event.startswith("void_beam:"): labels.append("虚空光束发射！")
            elif event.startswith("iron_aim:flame:"): labels.append("焚烧线预警：根盘引火，或从邻格截断成熟藤蔓！")
            elif event.startswith("iron_aim:thorn:"): labels.append("荆棘落点锁定：离开橙色区域！")
            elif event.startswith("iron_aim:bloom:"): labels.append("荆棘绽放预警：两格范围即将爆发！")
            elif event.startswith("iron_vine_grow:"): labels.append("机械藤蔓长成荆棘墙！")
            elif event.startswith("iron_vine_burn:"): labels.append("藤蔓被焚烧！")
            elif event.startswith("iron_reflux:"): labels.append("热回流击中 Boss 装甲！")
            elif event.startswith("iron_flame:"): labels.append("焚烧射线来袭！")
            elif event.startswith("iron_thorn:"): labels.append("荆棘爆发！")
            elif event.startswith("iron_bloom:"): labels.append("荆棘大范围绽放！")
            elif event.startswith("boss_summon:furnace:"): labels.append("熔炉孵出爆裂幼体！")
            elif event.startswith("boss_summon:iron:"): labels.append("园丁放出藤蔓猎兽！")
            elif event.startswith("mirror_aim:"): labels.append("镜像动作已预告：注意实际方向！")
            elif event.startswith("mirror_shard:"): labels.append("镜像碎片射线！")
            elif event.startswith("mirror_echo_aim:"): labels.append("双重镜片十字锁定旧位置及侧翼：下一轮离开紫色区域！")
            elif event.startswith("mirror_echo:"): labels.append("双重镜片十字爆裂！")
            elif event.startswith("mirror_rush_aim:"): labels.append("镜像炽天使突脸预警：离开红色近战区！")
            elif event.startswith("mirror_rush:"): labels.append("镜像炽天使高速突袭！")
            elif event.startswith("mirror_clone_spawn:"): labels.append("镜像主角现身：攻击分身会反噬本体！")
            elif event.startswith("mirror_clone_strike:"): labels.append("镜像主角近身共振！")
            elif event.startswith("mirror_link:"): labels.append("镜像受击，本体承受部分伤害！")
            elif event == "mirror_evade": labels.append("镜翼侧闪：子弹打中了残影！")
            elif event == "mirror_emp_jam": labels.append("EMP 干扰镜翼：本次破盾无法侧闪！")
            elif event.startswith("mirror_lock_break:"): labels.append("镜锁被反射碎片击碎！")
            elif event == "boss_reflectors_recharged": labels.append("棱镜重新充能，可再次反射！")
            elif event.startswith("prism_reflector_shove:"): labels.append("镜柱已推动：反射路径改变！")
            elif event.startswith("void_node_cut:"): labels.append("近战截断引力雷，装甲被吸离！")
            elif event.startswith("iron_vine_cut:"): labels.append("截断焚烧藤蔓，热量回流！")
            elif event.startswith("siege_charge_recoil:"): labels.append("冲撞击中移动掩体，侧甲反噬！")
            elif event == "apex_appeal_mark": labels.append("射击打断终审：上诉反击已准备！")
            elif event == "mirror_silence": labels.append("镜像 EMP：局部沉默一回合")
            elif event == "mirror_heal_echo": labels.append("镜像治疗：不消耗玩家药包")
            elif event.startswith("rail_aim:"): labels.append("轨道炮开始两轮蓄力：寻找可移动掩体！")
            elif event.startswith("rail_charge:"): labels.append("轨道炮即将贯穿预警线！")
            elif event.startswith("rail_fire:"): labels.append("轨道炮贯穿！")
            elif event.startswith("siege_blast_aim:shrapnel:"): labels.append("破片预警：击中掩体会向邻格反弹，避开亮格！")
            elif event.startswith("siege_blast_aim:"): labels.append("轰炸锁定：离开亮起的预警区域！")
            elif event.startswith("siege_blast:"): labels.append("利维坦轰炸：掩体可能被摧毁！")
            elif event.startswith("siege_pit_open:"): labels.append("地板塌陷成深坑，稍后会恢复！")
            elif event.startswith("siege_charge_aim:"): labels.append("利维坦冲撞锁定路径：侧移，或让移动掩体挡撞反噬侧甲！")
            elif event.startswith("siege_charge:"): labels.append("利维坦高速冲撞！")
            elif event.startswith("rail_cover_break:"): labels.append("掩体挡住炮击并碎裂！")
            elif event.startswith("rail_cover_rebuild:"): labels.append("掩体重新生成！")
            elif event.startswith("rail_cover_shove:"): labels.append("推动了掩体！")
            elif event.startswith("rail_lock_break:"): labels.append("折射爆炸击碎装甲锁！")
            elif event.startswith("null_node:"): labels.append("封印节点已激活：按编号继续！")
            elif event == "null_node_reset": labels.append("节点顺序错误：从①重新开始！")
            elif event.startswith("null_mark:"): labels.append("地板正在删除：两回合后离开标记格！")
            elif event.startswith("null_countdown:"): labels.append("地板即将消失：立刻离开粉色格！")
            elif event.startswith("null_fracture:"): labels.append("地板断裂！")
            elif event == "apex_barrage_second_aim": labels.append("第二波弹幕变向：刚才的安全格也会受击！")
            elif event == "null_displace": labels.append("被断裂地板弹开！")
            elif event == "null_reverse_write": labels.append("逆向写入完成：核心开放！")
            elif event.startswith("null_block:"): labels.append("Boss 封锁了一类动作，查看右侧提示！")
            elif event.startswith("null_warp_aim:"): labels.append("虚空跃迁预警：蓝色圆环是落点！")
            elif event == "null_warp": labels.append("虚空织者跃迁！")
            elif event.startswith("apex_aim:verdict:"): labels.append("终审判词：踩白色上诉位，或射击 Boss 主动上诉！")
            elif event.startswith("apex_aim:cage_barrage:"): labels.append("熔锁雷幕：破白门，躲蓝色弹幕！")
            elif event.startswith("apex_aim:charge_gravity:"): labels.append("镜冲引力：紫色路径和绿色爆心都危险！")
            elif event.startswith("apex_aim:"): labels.append("裁决法则锁定：按地面预警走位！")
            elif event.startswith("apex_cage:"): labels.append("牢笼成形：击碎白色闸门后离开！")
            elif event == "apex_gate_break": labels.append("闸门击碎：火流即将反向回灌！")
            elif event.startswith("apex_fire:"): labels.append("裁决攻击爆发！")
            elif event.startswith("apex_seal:"): labels.append("封印充能！")
            elif event == "apex_appeal": labels.append("终审上诉成功：判词反弹，核心开放！")
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
