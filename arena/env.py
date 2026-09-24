from __future__ import annotations

import copy
import random
from collections import deque
from dataclasses import dataclass

from .entities import (DIRECTIONS, ENEMY_MELEE_BONUS, ENEMY_MOVE_DELAY, ENEMY_SPEED_LEVEL, Action,
                       Enemy, EnemyType, Intent, IntentType, Player, PlayerLoadout)
from .boss import ApexArbiter, ChronoMantis, FurnaceHydra, IronGardener, MirrorSeraph, NullWeaver, PrismWarden, SiegeLeviathan, StormChoir, VoidAngler, ray_cells


@dataclass(frozen=True)
class ArenaConfig:
    difficulty_level: int = 1
    width: int = 20
    height: int = 20
    max_ticks: int = 500
    walls: int = 35
    enemies: int = 3
    gems: int = 6
    fires: int = 10
    spikes: int = 0
    pits: int = 0
    barrels: int = 0
    medkits: int = 2
    bow_pickups: int = 0
    pistol_pickups: int = 0
    arrow_bundles: int = 0
    energy_cells: int = 0
    enemy_damage: int = 5
    enemy_hp_bonus: int = 0
    fire_damage: int = 10
    spike_damage: int = 12
    attack_damage: int = 20
    heal_amount: int = 35
    enemy_move_interval: int = 1
    charger_ratio: float = 0.25
    bomber_ratio: float = 0.15
    archer_ratio: float = 0.0
    charger_range: int = 4
    charger_damage: int = 10
    collision_damage: int = 15
    bomber_damage: int = 20
    bomber_radius: int = 2
    archer_damage: int = 12
    archer_countdown: int = 2
    spawn_protection_rounds: int = 0
    bow_damage: int = 15
    bow_range: int = 6
    pistol_damage: int = 12
    pistol_range: int = 8
    dash_cooldown: int = 3
    emp_cooldown: int = 4
    emp_radius: int = 1
    barrel_damage: int = 25
    barrel_radius: int = 2
    action_points: int = 1
    dash_ap_cost: int = 1
    emp_ap_cost: int = 1
    finish_on_all_gems: bool = False


def campaign_config(level: int) -> ArenaConfig:
    if level < 1:
        raise ValueError("level must be positive")
    if level in (10, 20, 30, 40, 50, 60, 70, 80, 90, 100):
        return ArenaConfig(difficulty_level=level, max_ticks=400 if level == 100 else 300, walls=0, enemies=0, gems=0,
                           fires=0, spikes=0, pits=0, barrels=0, medkits=0,
                           enemy_hp_bonus=level - 1, action_points=2, spawn_protection_rounds=2,
                           finish_on_all_gems=True)
    return ArenaConfig(
        difficulty_level=level,
        max_ticks=min(300, 160 + level * 20),
        walls=min(55, 24 + level * 3),
        enemies=min(8, 1 + (level + 1) // 2),
        gems=min(8, 2 + (level + 1) // 2),
        fires=min(24, 4 + level * 2),
        spikes=min(8, max(0, level - 1) * 2),
        pits=min(4, max(0, level - 2)),
        barrels=min(4, level // 2),
        medkits=max(1, 3 - level // 3) + int(level >= 6 and level % 4 == 0),
        bow_pickups=1,
        pistol_pickups=1 if level >= 3 else 0,
        arrow_bundles=1 + int(level >= 6 and level % 2 == 0) if level >= 2 else 0,
        energy_cells=1 + int(level % 5 == 0) if level >= 4 else 0,
        enemy_damage=4 + (level - 1) // 3,
        enemy_hp_bonus=level - 1,
        fire_damage=9 + (level - 1) // 3,
        spike_damage=12 + (level - 1) // 4,
        charger_damage=10 + (level - 1) // 4,
        collision_damage=15 + (level - 1) // 4,
        bomber_damage=20 + (level - 1) // 3,
        archer_damage=12 + (level - 1) // 4,
        barrel_damage=25 + (level - 1) // 3,
        enemy_move_interval=2,
        charger_ratio=min(0.45, 0.2 + level * 0.03),
        bomber_ratio=min(0.25, 0.1 + level * 0.02),
        archer_ratio=min(0.2, 0.08 + level * 0.02),
        spawn_protection_rounds=2,
        action_points=2,
        finish_on_all_gems=True,
    )


@dataclass(frozen=True)
class StepResult:
    reward: float
    done: bool
    events: tuple[str, ...]


class ArenaEnv:
    def __init__(self, config: ArenaConfig | None = None, loadout: PlayerLoadout | None = None):
        self.config = config or ArenaConfig()
        self.loadout = copy.deepcopy(loadout or PlayerLoadout())
        self.reset(0)

    def reset(self, seed: int = 0) -> dict:
        if (self.config.difficulty_level < 1 or self.config.enemy_move_interval < 1 or self.config.charger_range < 1 or
                self.config.archer_countdown < 1 or self.config.action_points < 1 or
                not 1 <= self.config.dash_ap_cost <= self.config.action_points or
                not 1 <= self.config.emp_ap_cost <= self.config.action_points):
            raise ValueError("enemy intervals, ranges, and AP costs must be valid")
        ratios = (self.config.charger_ratio, self.config.bomber_ratio, self.config.archer_ratio)
        if any(not 0 <= ratio <= 1 for ratio in ratios) or sum(ratios) > 1:
            raise ValueError("enemy type ratios must be between zero and one and sum to at most one")
        self.seed = seed
        self.rng = random.Random(
            f"{seed}:{self.config.difficulty_level}"
            if self.config.finish_on_all_gems and self.config.difficulty_level > 1 else seed)
        self.tick = self.score = self.gems_collected = self.kills = self.environment_kills = 0
        self.round, self.ap_remaining = 1, self.config.action_points
        self.damage_taken = 0
        self.previous_player_position: tuple[int, int] | None = None
        self.last_action: str | None = None
        self.last_non_wait_action: str | None = None
        self.done = False
        for _ in range(20):
            self._generate_map()
            self._plan_enemy_intents()
            if self._map_is_playable():
                return self.observation()
        raise RuntimeError("could not generate a playable map after 20 attempts")

    def _generate_map(self) -> None:
        self.boss: PrismWarden | FurnaceHydra | StormChoir | ChronoMantis | VoidAngler | IronGardener | MirrorSeraph | SiegeLeviathan | NullWeaver | ApexArbiter | None = None
        self.reflectors: set[tuple[int, int]] = set()
        self.breakable_walls: set[tuple[int, int]] = set()
        self.coolant_valves: set[tuple[int, int]] = set()
        self.grounding_pylons: set[tuple[int, int]] = set()
        self.relay_pads: set[tuple[int, int]] = set()
        self.time_anchors: set[tuple[int, int]] = set()
        self.gravity_nodes: set[tuple[int, int]] = set()
        self.root_plates: set[tuple[int, int]] = set()
        self.mirror_locks: set[tuple[int, int]] = set()
        self.rail_locks: set[tuple[int, int]] = set()
        self.rail_covers: dict[tuple[int, int], tuple[int, int]] = {}
        self.rail_rebuilds: dict[tuple[int, int], int] = {}
        self.null_nodes: tuple[tuple[int, int], ...] = ()
        self.null_void: set[tuple[int, int]] = set()
        self.apex_seals: tuple[tuple[int, int], ...] = ()
        self.apex_cage: set[tuple[int, int]] = set()
        self.vine_seeds: dict[tuple[int, int], int] = {}
        self.vine_walls: set[tuple[int, int]] = set()
        self.forge_floor: set[tuple[int, int]] = set()
        self.furnace_burns: dict[tuple[int, int], int] = {}
        if self.config.difficulty_level in (10, 20, 30, 40, 50, 60, 70, 80, 90, 100) and self.config.finish_on_all_gems:
            furnace = self.config.difficulty_level == 20
            storm = self.config.difficulty_level == 30
            chrono = self.config.difficulty_level == 40
            void = self.config.difficulty_level == 50
            iron = self.config.difficulty_level == 60
            mirror = self.config.difficulty_level == 70
            siege = self.config.difficulty_level == 80
            null = self.config.difficulty_level == 90
            apex = self.config.difficulty_level == 100
            self.player = Player((10, 16 if furnace or storm or chrono or void or iron or mirror or siege or null or apex else 15), loadout=self.loadout)
            if furnace:
                self.player.loadout.bow = True
                self.player.loadout.arrows = max(10, self.player.loadout.arrows)
                room = {(x, y) for y in range(20)
                        for x in range(4 if y in (0, 1, 18, 19) else
                                       2 if y in (2, 3, 16, 17) else 0,
                                       16 if y in (0, 1, 18, 19) else
                                       18 if y in (2, 3, 16, 17) else 20)}
            elif storm or chrono or void:
                self.player.loadout.pistol = True
                self.player.loadout.energy = max(12, self.player.loadout.energy)
                top = (3, 17) if storm else (4, 16) if chrono else (5, 15)
                bottom = (3, 17) if storm else (2, 18) if chrono else (4, 16)
                room = {(x, y) for y in range(20)
                        for x in range(top[0] if y in (0, 1) else
                                       bottom[0] if y in (18, 19) else
                                       1 if y in (2, 3, 16, 17) else 0,
                                       top[1] if y in (0, 1) else
                                       bottom[1] if y in (18, 19) else
                                       19 if y in (2, 3, 16, 17) else 20)}
            elif iron or mirror:
                self.player.loadout.pistol = True
                self.player.loadout.energy = max(12, self.player.loadout.energy)
                room = {(x, y) for y in range(20)
                        for x in range((4 if iron else 3) if y in (0, 1, 2) else
                                       (4 if iron else 5) if y in (17, 18, 19) else
                                       2 if y in (3, 4, 15, 16) else 0,
                                       (16 if iron else 17) if y in (0, 1, 2) else
                                       (16 if iron else 15) if y in (17, 18, 19) else
                                       18 if y in (3, 4, 15, 16) else 20)}
            elif siege:
                self.player.loadout.pistol = True
                self.player.loadout.energy = max(18, self.player.loadout.energy)
                room = {(x, y) for y in range(20)
                        for x in range(5 if y <= 2 else 3 if y <= 5 else
                                       2 if y >= 15 and y <= 17 else 6 if y >= 18 else 1,
                                       15 if y <= 2 else 18 if y <= 5 else
                                       18 if y >= 15 and y <= 17 else 14 if y >= 18 else 19)}
            elif null:
                self.player.loadout.pistol = True
                self.player.loadout.energy = max(24, self.player.loadout.energy)
                room = {(x, y) for y in range(20)
                        for x in range(6 if y <= 1 else 3 if y <= 4 else
                                       2 if y >= 16 and y <= 17 else 5 if y >= 18 else 0,
                                       14 if y <= 1 else 17 if y <= 4 else
                                       18 if y >= 16 and y <= 17 else 15 if y >= 18 else 20)}
            elif apex:
                self.player.loadout.pistol = True
                self.player.loadout.energy = max(24, self.player.loadout.energy)
                room = {(x, y) for y in range(20)
                        for x in range(6 if y <= 1 else 2 if y <= 3 else
                                       1 if y <= 16 else 3 if y <= 18 else 6,
                                       14 if y <= 1 else 18 if y <= 3 else
                                       19 if y <= 16 else 17 if y <= 18 else 14)}
            else:
                self.player.loadout.pistol = True
                self.player.loadout.energy = max(12, self.player.loadout.energy)
                room = {(x, y) for y in range(1, 19)
                        for x in range(3 if y in (1, 2, 17, 18) else 1,
                                       17 if y in (1, 2, 17, 18) else 19)}
            self.walls = {cell for cell in room
                          if any(self.add(cell, direction) not in room for direction in DIRECTIONS)}
            self.enemies, self.gems, self.fires, self.spikes = [], set(), set(), set()
            self.pits, self.barrels = set(), set()
            if self.config.difficulty_level == 10:
                self.breakable_walls = {(6, 10), (14, 10), (6, 13), (14, 13)}
                self.walls |= self.breakable_walls
                self.medkits, self.energy_cells = {(7, 14), (13, 14)}, {(7, 15), (13, 15)}
                self.bow_pickups, self.pistol_pickups = set(), {(10, 14)}
                self.arrow_bundles = set()
                self.reflectors = {(9, 12), (11, 12), (8, 13), (12, 13)}
                self.boss = PrismWarden()
            elif furnace:
                self.walls |= {(5, 8), (14, 8), (5, 9), (14, 9),
                               (8, 10), (12, 10), (6, 14), (14, 14)}
                self.forge_floor = room - self.walls
                self.fires = {(4, 11), (15, 11), (4, 13), (15, 13)}
                self.medkits, self.energy_cells = {(5, 16), (15, 16)}, set()
                self.bow_pickups, self.pistol_pickups = {(10, 15)}, set()
                self.arrow_bundles = {(6, 16), (14, 16)}
                self.coolant_valves = {(x, 12) for x in (7, 10, 13)}
                self.boss = FurnaceHydra()
            elif storm:
                self.walls |= {(5, 7), (14, 7), (5, 11), (14, 11),
                               (4, 14), (15, 14), (6, 16), (13, 16)}
                self.pits = {(3, 9), (16, 9), (4, 16), (15, 16)}
                self.medkits, self.energy_cells = {(5, 15)}, {(7, 15), (13, 15), (10, 14)}
                self.bow_pickups, self.pistol_pickups = set(), set()
                self.arrow_bundles = set()
                self.grounding_pylons = {(8, 9), (12, 9), (11, 13), (7, 13)}
                self.relay_pads = {(8, 8), (10, 8), (12, 8)}
                self.boss = StormChoir()
            elif chrono:
                self.walls |= {(4, 8), (15, 8), (5, 10), (12, 10),
                               (4, 13), (15, 13), (6, 15), (13, 15)}
                self.pits = {(3, 11), (16, 11), (5, 16), (14, 16)}
                self.medkits, self.energy_cells = {(6, 14), (13, 14), (11, 16)}, {(7, 15)}
                self.bow_pickups, self.pistol_pickups = set(), {(10, 15)}
                self.arrow_bundles = set()
                self.time_anchors = {(9, 11), (14, 11)}
                self.boss = ChronoMantis()
            elif void:
                self.walls |= {(5, 8), (14, 8), (6, 12), (14, 12),
                               (4, 14), (15, 14)}
                self.pits = {(3, 10), (16, 10), (5, 15), (15, 15)}
                self.medkits, self.energy_cells = {(6, 15)}, {(7, 16), (13, 16)}
                self.bow_pickups, self.pistol_pickups = set(), {(10, 15)}
                self.arrow_bundles = set()
                self.gravity_nodes = {(7, 11), (10, 12), (13, 11)}
                self.boss = VoidAngler()
            elif iron:
                self.walls |= {(4, 8), (15, 8), (5, 14), (15, 14)}
                self.pits = {(3, 11), (16, 11), (4, 16), (15, 16)}
                self.medkits, self.energy_cells = {(6, 15), (14, 15)}, {(8, 16)}
                self.bow_pickups, self.pistol_pickups = set(), set()
                self.arrow_bundles = set()
                self.root_plates = {(x, 12) for x in (6, 9, 12, 15)}
                self.vine_seeds = {(x, 10): 2 for x in (6, 9, 12, 15)}
                self.boss = IronGardener()
            elif mirror:
                self.walls |= {(4, 9), (16, 9), (6, 12), (14, 12), (5, 15), (15, 15)}
                self.pits = {(3, 11), (17, 11)}
                self.medkits, self.energy_cells = {(6, 15), (14, 15), (10, 14)}, {(8, 16), (12, 16)}
                self.bow_pickups, self.pistol_pickups = set(), set()
                self.arrow_bundles = set()
                self.mirror_locks = {(6, 6), (14, 6), (10, 11)}
                self.boss = MirrorSeraph()
            elif siege:
                self.walls |= {(5, 8), (15, 8), (4, 12), (16, 12), (6, 15), (14, 15)}
                self.pits = {(3, 9), (17, 9), (4, 16), (16, 16)}
                self.medkits, self.energy_cells = {(5, 16), (15, 16), (10, 17)}, {(10, 14)}
                self.bow_pickups, self.pistol_pickups = {(5, 13)}, set()
                self.arrow_bundles = {(4, 15), (16, 15)}
                self.rail_locks = {(x, 7) for x in (7, 8, 12, 13)}
                self.rail_covers = {origin: origin for origin in ((7, 11), (10, 13), (13, 11))}
                self.boss = SiegeLeviathan()
            elif null:
                self.walls |= {(4, 8), (16, 8), (7, 9), (13, 9),
                               (5, 13), (15, 13), (8, 16), (12, 16)}
                self.pits = {(3, 10), (17, 10), (4, 15), (16, 15)}
                self.medkits, self.energy_cells = {(5, 16), (15, 16)}, {(7, 15), (10, 14), (13, 15)}
                self.bow_pickups, self.pistol_pickups = {(4, 12)}, set()
                self.arrow_bundles = {(16, 12)}
                self.null_nodes = ((5, 11), (15, 11), (6, 14), (11, 10))
                self.boss = NullWeaver()
            else:
                self.walls |= {(3, 8), (17, 8), (4, 13), (16, 13), (7, 16), (13, 15)}
                self.pits = {(2, 10), (18, 10), (4, 16), (16, 16)}
                self.medkits = {(6, 15), (15, 14), (5, 9)}
                self.energy_cells = {(8, 15), (14, 11), (6, 7)}
                self.bow_pickups, self.arrow_bundles = {(15, 12)}, {(5, 12), (12, 17)}
                self.pistol_pickups = set()
                self.apex_seals = ((5, 11), (15, 11), (10, 11), (6, 9))
                self.boss = ApexArbiter()
            return
        cells = [(x, y) for y in range(self.config.height) for x in range(self.config.width)]
        self.rng.shuffle(cells)
        needed = (1 + self.config.walls + self.config.enemies + self.config.gems + self.config.fires +
                  self.config.spikes + self.config.pits + self.config.barrels +
                  self.config.medkits + self.config.bow_pickups + self.config.pistol_pickups +
                  self.config.arrow_bundles + self.config.energy_cells)
        if needed > len(cells):
            raise ValueError("map contains more entities than cells")

        take = iter(cells)
        self.player = Player(next(take), loadout=self.loadout)
        self.walls = {next(take) for _ in range(self.config.walls)}
        self.enemies = []
        for _ in range(self.config.enemies):
            roll = self.rng.random()
            enemy_type = (EnemyType.BOMBER if roll < self.config.bomber_ratio else
                          EnemyType.CHARGER if roll < self.config.bomber_ratio + self.config.charger_ratio else
                          EnemyType.ARCHER if roll < sum((self.config.bomber_ratio,
                                                         self.config.charger_ratio,
                                                         self.config.archer_ratio)) else
                          EnemyType.CHASER)
            enemy = Enemy(next(take), enemy_type=enemy_type)
            enemy.hp += self.config.enemy_hp_bonus
            enemy.max_hp += self.config.enemy_hp_bonus
            self.enemies.append(enemy)
        self.gems = {next(take) for _ in range(self.config.gems)}
        self.fires = {next(take) for _ in range(self.config.fires)}
        self.spikes = {next(take) for _ in range(self.config.spikes)}
        self.pits = {next(take) for _ in range(self.config.pits)}
        self.barrels = {next(take) for _ in range(self.config.barrels)}
        self.medkits = {next(take) for _ in range(self.config.medkits)}
        self.bow_pickups = {next(take) for _ in range(self.config.bow_pickups)}
        self.pistol_pickups = {next(take) for _ in range(self.config.pistol_pickups)}
        self.arrow_bundles = {next(take) for _ in range(self.config.arrow_bundles)}
        self.energy_cells = {next(take) for _ in range(self.config.energy_cells)}
        self._ensure_pickups_reachable()

    def _reachable_cells(self) -> set[tuple[int, int]]:
        blocked = (self.walls | self.fires | self.spikes | self.pits | self.barrels |
                   set(self.rail_covers.values()) | self.null_void | self.apex_cage)
        reachable = {self.player.position}
        frontier = [self.player.position]
        while frontier:
            position = frontier.pop()
            for direction in DIRECTIONS:
                target = self.add(position, direction)
                if self.in_bounds(target) and target not in blocked and target not in reachable:
                    reachable.add(target)
                    frontier.append(target)
        return reachable

    def _map_is_playable(self) -> bool:
        possible_exits = sum(self.in_bounds(self.add(self.player.position, direction))
                             for direction in DIRECTIONS)
        exits = sum(self.in_bounds(self.add(self.player.position, direction)) and
                    self.add(self.player.position, direction) not in self.walls and
                    self.add(self.player.position, direction) not in self.pits and
                    self.add(self.player.position, direction) not in self.barrels and
                    self.add(self.player.position, direction) not in self.fires and
                    self.add(self.player.position, direction) not in self.spikes and
                    not self.enemy_at(self.add(self.player.position, direction))
                    for direction in DIRECTIONS)
        if exits < min(2, possible_exits) or not (self.gems | self.medkits) <= self._reachable_cells():
            return False
        for action in self.legal_actions():
            simulation = self.clone()
            simulation.step(action)
            if not simulation.done and simulation.ap_remaining < simulation.config.action_points:
                simulation.step(Action.WAIT)
            if simulation.player.hp > 0:
                return True
        return False

    def clone(self) -> ArenaEnv:
        return copy.deepcopy(self)

    def in_bounds(self, position: tuple[int, int]) -> bool:
        x, y = position
        return 0 <= x < self.config.width and 0 <= y < self.config.height

    def add(self, position: tuple[int, int], direction: str) -> tuple[int, int]:
        dx, dy = DIRECTIONS[direction]
        return position[0] + dx, position[1] + dy

    def enemy_at(self, position: tuple[int, int]) -> Enemy | None:
        return next((enemy for enemy in self.enemies if enemy.position == position), None)

    def boss_at(self, position: tuple[int, int]) -> PrismWarden | FurnaceHydra | StormChoir | ChronoMantis | VoidAngler | IronGardener | MirrorSeraph | SiegeLeviathan | NullWeaver | ApexArbiter | None:
        return self.boss if self.boss and self.boss.position == position else None

    def boss_ray(self) -> tuple[tuple[int, int], ...]:
        if not isinstance(self.boss, PrismWarden) or self.boss.target is None:
            return ()
        cells = ray_cells(self.boss.position, self.boss.target)
        stop = next((index for index, cell in enumerate(cells)
                     if cell in self.walls or cell in self.reflectors - self.boss.used_reflectors), None)
        return cells[:stop + 1] if stop is not None else cells

    def prism_attack_cells(self) -> set[tuple[int, int]]:
        path = self.boss_ray()
        if not path:
            return set()
        cells = set(path)
        if self.boss.sweep and path[-1] not in self.reflectors | self.walls:
            x, y = path[-1]
            cells.update((x + dx, y + dy) for dx, dy in DIRECTIONS.values()
                         if self.in_bounds((x + dx, y + dy)))
        return cells

    def prism_baits(self) -> set[tuple[int, int]]:
        if not isinstance(self.boss, PrismWarden) or self.boss.position[1] != 8:
            return set()
        unused = self.reflectors - self.boss.used_reflectors
        baits = set()
        for y in range(14, 17):
            for x in range(4, 16):
                if (x, y) in self.walls | self.fires | self.spikes | self.pits:
                    continue
                for cell in ray_cells(self.boss.position, (x, y)):
                    if cell in self.walls:
                        break
                    if cell in unused:
                        baits.add((x, y))
                        break
        return baits

    def iron_root_ready(self, x: int) -> bool:
        seed = (x, 10)
        return seed in self.vine_walls or self.vine_seeds.get(seed) == 1

    def storm_chain(self) -> tuple[tuple[int, int], ...]:
        if not isinstance(self.boss, StormChoir) or self.boss.target is None:
            return ()
        nodes = [self.boss.position, self.boss.target]
        remaining = set(self.grounding_pylons)
        while remaining:
            nearest = min(remaining, key=lambda cell: (self._distance(nodes[-1], cell), cell))
            if self._distance(nodes[-1], nearest) > 5:
                break
            nodes.append(nearest)
            remaining.remove(nearest)
        return tuple(nodes)

    def chrono_landing_x(self) -> int:
        boss = self.boss
        if not isinstance(boss, ChronoMantis):
            raise ValueError("chrono landing requires ChronoMantis")
        if boss.leap_target:
            return boss.leap_target[0]
        if boss.phase == "slash":
            return 9 if boss.position[0] >= 12 else 14
        player_x = self.player.position[0]
        if player_x <= 7:
            return 9
        if player_x >= 13:
            return 14
        return 9 if boss.position[0] >= 10 else 14

    def _ensure_pickups_reachable(self) -> None:
        reachable = self._reachable_cells()
        pickup_sets = (self.bow_pickups, self.pistol_pickups, self.arrow_bundles, self.energy_cells)
        occupied = ({self.player.position} | self.walls | self.gems | self.fires | self.spikes | self.pits |
                    self.barrels | self.medkits |
                    {enemy.position for enemy in self.enemies} | set().union(*pickup_sets))
        free = sorted(reachable - occupied, key=lambda p: self._distance(self.player.position, p))
        for pickups in pickup_sets:
            for position in list(pickups - reachable):
                if not free:
                    return
                pickups.remove(position)
                pickups.add(free.pop(0))

    def legal_actions(self) -> list[Action]:
        actions: list[Action] = []
        ranged: list[tuple[int, Action]] = []
        for direction in DIRECTIONS:
            target = self.add(self.player.position, direction)
            if (self.in_bounds(target) and target not in self.walls and target not in self.pits and
                    target not in self.barrels and target not in self.rail_covers.values() and
                    target not in self.null_void and target not in self.apex_cage and
                    not self.enemy_at(target) and not self.boss_at(target)):
                actions.append(Action(f"move_{direction}"))
            if self.enemy_at(target) or self.boss_at(target) or target in self.barrels or target == getattr(self.boss, "gate", None) and target in self.apex_cage:
                actions.append(Action(f"attack_{direction}"))
            if self.enemy_at(target):
                if self.in_bounds(self.add(target, direction)):
                    actions.append(Action(f"shove_{direction}"))
            if target in self.rail_covers.values():
                destination = self.add(target, direction)
                if (self.in_bounds(destination) and destination not in self.walls | self.pits | self.barrels and
                        destination not in self.rail_covers.values() and
                        destination != self.player.position and not self.boss_at(destination)):
                    actions.append(Action(f"shove_{direction}"))
            destination = self.add(target, direction)
            if (not self.player.cooldowns.get("dash", 0) and self.in_bounds(target) and
                    self.in_bounds(destination) and target not in self.walls and destination not in self.walls and
                    target not in self.pits and destination not in self.pits and
                    target not in self.barrels and destination not in self.barrels and
                    target not in self.null_void and destination not in self.null_void and
                    target not in self.apex_cage and destination not in self.apex_cage and
                    target not in self.rail_covers.values() and destination not in self.rail_covers.values() and
                    not self.enemy_at(target) and not self.enemy_at(destination) and
                    not self.boss_at(target) and not self.boss_at(destination)):
                actions.append(Action(f"dash_{direction}"))
            bow_distance = self._ranged_target_distance(direction, self.config.bow_range)
            pistol_distance = self._ranged_target_distance(direction, self.config.pistol_range)
            silenced = (isinstance(self.boss, MirrorSeraph) and self.boss.silence_rounds and
                        self._distance(self.player.position, self.boss.position) <= 4)
            if not silenced and self.player.loadout.bow and self.player.loadout.arrows and bow_distance and (bow_distance > 1 or self._apex_gate_distance(direction, self.config.bow_range)):
                ranged.append((bow_distance, Action(f"shoot_bow_{direction}")))
            if not silenced and self.player.loadout.pistol and self.player.loadout.energy and pistol_distance and (pistol_distance > 1 or self._apex_gate_distance(direction, self.config.pistol_range)):
                ranged.append((pistol_distance, Action(f"shoot_pistol_{direction}")))
        if self.player.medkits and self.player.hp < 100:
            actions.append(Action.HEAL)
        if (not self.player.cooldowns.get("emp", 0) and
                (any(self._distance(self.player.position, enemy.position) <= self.config.emp_radius
                     for enemy in self.enemies) or
                 isinstance(self.boss, MirrorSeraph) and
                 self._distance(self.player.position, self.boss.position) <= self.config.emp_radius)):
            actions.append(Action.EMP)
        ranged.sort(key=lambda item: item[0])
        actions.extend(action for _, action in ranged[:max(0, 11 - len(actions))])
        actions.append(Action.WAIT)
        locked = self.boss.blocked_kind if isinstance(self.boss, NullWeaver) else None
        return [action for action in actions if self._action_cost(action) <= self.ap_remaining and
                not (locked == "move" and action.value.startswith("move_") or
                     locked == "melee" and action.value.startswith(("attack_", "shove_")) or
                     locked == "ranged" and action.value.startswith("shoot_") or
                     locked == "skill" and (action.value.startswith("dash_") or action == Action.EMP))]

    def _action_cost(self, action: Action) -> int:
        if action.value.startswith("dash_"):
            return self.config.dash_ap_cost
        if action == Action.EMP:
            return self.config.emp_ap_cost
        return 1

    def step(self, action: Action | str) -> StepResult:
        if self.done:
            raise RuntimeError("episode is already finished")
        action = Action(action)
        if action not in self.legal_actions():
            raise ValueError(f"illegal action: {action}")
        self.player.invulnerable = action.value.startswith("dash_")
        if self.ap_remaining == self.config.action_points:
            self._tick_cooldowns()
        origin = self.player.position
        reward = 0.05
        events: list[str] = []
        if action.value.startswith("move_"):
            self.player.position = self.add(self.player.position, action.value[-1])
            reward += self._collect(events)
            if self.player.position in self.spikes:
                reward += self._damage_entity(self.player, self.config.spike_damage, events, "spike") - 3
        elif action.value.startswith("dash_"):
            direction = action.value[-1]
            traversed = []
            for _ in range(2):
                self.player.position = self.add(self.player.position, direction)
                traversed.append(self.player.position)
            self.player.cooldowns["dash"] = self.config.dash_cooldown
            events.append(f"dash:{direction}")
            reward += self._collect(events)
            for position in traversed:
                if position != self.player.position and position in self.fires:
                    reward += self._damage_entity(self.player, self.config.fire_damage, events, "fire")
                if position in self.spikes:
                    reward += self._damage_entity(self.player, self.config.spike_damage, events, "spike")
        elif action.value.startswith("attack_"):
            target = self.add(self.player.position, action.value[-1])
            events.append("attack")
            if isinstance(self.boss, ApexArbiter) and target == self.boss.gate and target in self.apex_cage:
                self.apex_cage.remove(target)
                self.boss.gate_broken = True
                events.append("apex_gate_break")
            elif target in self.barrels:
                reward += self._explode_barrel(target, events)
            else:
                enemy = self.enemy_at(target) or self.boss_at(target)
                assert enemy is not None
                reward += self._damage_entity(enemy, self.config.attack_damage, events, "attack")
        elif action.value.startswith("shove_"):
            direction = action.value[-1]
            enemy = self.enemy_at(self.add(self.player.position, direction))
            if enemy is not None:
                reward += self._push_entity(enemy, direction, events)
            else:
                position = self.add(self.player.position, direction)
                origin = next(origin for origin, cover in self.rail_covers.items() if cover == position)
                self.rail_covers[origin] = self.add(position, direction)
                events.append(f"rail_cover_shove:{position[0]}:{position[1]}:{direction}")
        elif action.value.startswith("shoot_"):
            weapon = "bow" if action.value.startswith("shoot_bow_") else "pistol"
            reward += self._player_shoot(weapon, action.value[-1], events)
        elif action == Action.HEAL:
            self.player.medkits -= 1
            self.player.hp = min(100, self.player.hp + self.config.heal_amount)
            events.append("heal")
        elif action == Action.EMP:
            affected = [enemy for enemy in self.enemies
                        if self._distance(self.player.position, enemy.position) <= self.config.emp_radius]
            for enemy in affected:
                enemy.stunned += 1
            self.player.cooldowns["emp"] = self.config.emp_cooldown
            events.append(f"emp:{len(affected)}")

        if self.player.hp > 0 and self.player.position in self.fires:
            reward += self._damage_entity(self.player, self.config.fire_damage, events, "fire")
            if action.value.startswith("move_"):
                reward -= 3

        if action != Action.WAIT:
            self.last_non_wait_action = action.value
        self.ap_remaining -= self._action_cost(action)
        if self.player.hp > 0 and self.ap_remaining == 0:
            reward += self._resolve_enemy_intents(events)
            reward += self._resolve_boss(events)
            events.append("round_end")
            if self.player.hp > 0:
                self.round += 1
                self.ap_remaining = self.config.action_points
        self.previous_player_position = origin
        self.last_action = action.value
        self.tick += 1
        if self.player.hp <= 0:
            reward -= 30
            self.done = True
            events.append("death")
        elif self.config.finish_on_all_gems and not self.gems and self.boss is None:
            reward += 25
            self.score += 25
            self.done = True
            events.append("level_complete")
        elif self.tick >= self.config.max_ticks:
            reward += 10
            self.done = True
            events.append("completed")
        self.player.invulnerable = False
        return StepResult(round(reward, 4), self.done, tuple(events))

    def _tick_cooldowns(self) -> None:
        for skill, remaining in self.player.cooldowns.items():
            self.player.cooldowns[skill] = max(0, remaining - 1)

    def _collect(self, events: list[str]) -> float:
        reward = 0.0
        if self.player.position in self.gems:
            self.gems.remove(self.player.position)
            self.gems_collected += 1
            self.score += 10
            reward += 10
            events.append("gem")
        if self.player.position in self.medkits:
            self.medkits.remove(self.player.position)
            self.player.medkits += 1
            self.score += 3
            reward += 3
            events.append("medkit")
        if self.player.position in self.bow_pickups:
            self.bow_pickups.remove(self.player.position)
            self.player.loadout.bow = True
            self.player.loadout.arrows = min(12, self.player.loadout.arrows + 3)
            events.append("pickup_bow")
        if self.player.position in self.pistol_pickups:
            self.pistol_pickups.remove(self.player.position)
            self.player.loadout.pistol = True
            self.player.loadout.energy = min(24, self.player.loadout.energy + 6)
            events.append("pickup_pistol")
        if self.player.position in self.arrow_bundles:
            self.arrow_bundles.remove(self.player.position)
            self.player.loadout.arrows = min(12, self.player.loadout.arrows + 3)
            events.append("pickup_arrows")
        if self.player.position in self.energy_cells:
            self.energy_cells.remove(self.player.position)
            self.player.loadout.energy = min(24, self.player.loadout.energy + 6)
            events.append("pickup_energy")
        if isinstance(self.boss, NullWeaver) and not self.boss.exposed_rounds and self.player.position in self.null_nodes:
            expected = self.null_nodes[self.boss.node_index]
            if self.player.position == expected:
                self.boss.node_index += 1
                reward += 15
                events.append(f"null_node:{self.boss.node_index}")
                if self.boss.node_index == len(self.null_nodes):
                    self.boss.exposed_rounds = 6
                    self.boss.blocked_kind = None
                    self.boss.erase_targets.clear()
                    self.boss.erase_countdown = 0
                    self.boss.warp_target = None
                    self.null_void.clear()
                    events.append("boss_shield_break")
                    events.append("null_reverse_write")
            else:
                self.boss.node_index = 0
                events.append("null_node_reset")
        return reward

    def _damage_entity(self, entity: Player | Enemy | PrismWarden | FurnaceHydra | StormChoir | ChronoMantis | VoidAngler | IronGardener | MirrorSeraph | SiegeLeviathan | NullWeaver | ApexArbiter,
                       amount: int, events: list[str], source: str) -> float:
        if entity is self.player and self.player.invulnerable:
            events.append(f"invulnerable:{source}")
            return 0.0
        actual = min(amount, entity.hp)
        entity.hp -= actual
        if entity is self.player:
            self.damage_taken += actual
            events.append(f"damage:{source}:{actual}")
            return -0.1 * actual
        if entity is self.boss:
            if not entity.exposed_rounds:
                entity.hp += actual
                events.append("boss_shield")
                return 0.0
            events.append(f"boss_hit:{actual}")
            if isinstance(entity, PrismWarden) and entity.hp > 0 and entity.hp <= entity.max_hp * 2 // 3 < entity.hp + actual:
                events.append("boss_prism_phase_two")
            if isinstance(entity, FurnaceHydra) and entity.hp > 0 and entity.hp <= entity.max_hp // 2 < entity.hp + actual:
                events.append("furnace_phase_two")
            if isinstance(entity, StormChoir) and entity.hp > 0 and entity.hp <= entity.max_hp * 2 // 3 < entity.hp + actual:
                events.append("storm_phase_two")
            if isinstance(entity, ChronoMantis) and entity.hp > 0 and entity.hp <= entity.max_hp * 2 // 3 < entity.hp + actual:
                events.append("chrono_phase_two")
            if entity.hp <= 0:
                self.apex_cage.clear()
                self.boss = None
                self.kills += 1
                self.score += 50
                events.append("boss_defeated")
                return 50.0
            return actual * 0.5
        events.append(f"enemy_damage:{source}:{actual}")
        if entity.hp > 0:
            return 0.0
        self.enemies.remove(entity)
        self.kills += 1
        self.score += 20
        if source not in ("attack", "bow", "pistol"):
            self.environment_kills += 1
            events.append("environment_kill")
        events.append("kill")
        return 20.0 if source in ("attack", "bow", "pistol") else 10.0

    def _ray_target(self, origin: tuple[int, int], direction: str,
                    range_: int) -> tuple[Enemy | PrismWarden | FurnaceHydra | StormChoir | ChronoMantis | VoidAngler | IronGardener | MirrorSeraph | SiegeLeviathan | NullWeaver | ApexArbiter, int] | None:
        target = origin
        for distance in range(1, range_ + 1):
            target = self.add(target, direction)
            if not self.in_bounds(target) or target in self.walls | self.apex_cage or target in self.rail_covers.values():
                return None
            enemy = self.enemy_at(target) or self.boss_at(target)
            if enemy:
                return enemy, distance
        return None

    def _barrel_target(self, origin: tuple[int, int], direction: str, range_: int) -> tuple[tuple[int, int], int] | None:
        target = origin
        for distance in range(1, range_ + 1):
            target = self.add(target, direction)
            if not self.in_bounds(target) or target in self.walls or self.enemy_at(target) or self.boss_at(target):
                return None
            if target in self.barrels:
                return target, distance
        return None

    def _ranged_target_distance(self, direction: str, range_: int) -> int | None:
        enemy = self._ray_target(self.player.position, direction, range_)
        barrel = self._barrel_target(self.player.position, direction, range_)
        distances = [target[1] for target in (enemy, barrel) if target]
        gate = self._apex_gate_distance(direction, range_)
        if gate:
            distances.append(gate)
        return min(distances) if distances else None

    def _apex_gate_distance(self, direction: str, range_: int) -> int | None:
        if not isinstance(self.boss, ApexArbiter) or not self.boss.gate:
            return None
        target = self.player.position
        for distance in range(1, range_ + 1):
            target = self.add(target, direction)
            if target == self.boss.gate and target in self.apex_cage:
                return distance
            if target in self.walls | self.apex_cage:
                return None
        return None

    def _player_shoot(self, weapon: str, direction: str, events: list[str]) -> float:
        loadout = self.player.loadout
        if weapon == "bow":
            loadout.arrows -= 1
            damage, range_ = self.config.bow_damage, self.config.bow_range
        else:
            loadout.energy -= 1
            damage, range_ = self.config.pistol_damage, self.config.pistol_range
        gate_distance = self._apex_gate_distance(direction, range_)
        if gate_distance is not None and isinstance(self.boss, ApexArbiter):
            self.apex_cage.remove(self.boss.gate)
            self.boss.gate_broken = True
            events.extend((f"shoot_{weapon}:{direction}:{gate_distance}", "apex_gate_break"))
            return 0.0
        enemy_target = self._ray_target(self.player.position, direction, range_)
        barrel_target = self._barrel_target(self.player.position, direction, range_)
        barrel_first = bool(barrel_target and (not enemy_target or barrel_target[1] < enemy_target[1]))
        target = barrel_target if barrel_first else enemy_target
        assert target is not None
        distance = target[1]
        events.append(f"shoot_{weapon}:{direction}:{distance}")
        if barrel_first:
            assert barrel_target is not None
            return self._explode_barrel(barrel_target[0], events)
        assert enemy_target is not None
        enemy, _ = enemy_target
        reward = self._damage_entity(enemy, damage, events, weapon)
        if weapon == "bow" and enemy in self.enemies:
            reward += self._push_entity(enemy, direction, events)
        return reward

    def _resolve_boss(self, events: list[str]) -> float:
        boss = self.boss
        if boss is None or self.round < self.config.spawn_protection_rounds:
            return 0.0
        if isinstance(boss, FurnaceHydra):
            return self._resolve_furnace(boss, events)
        if isinstance(boss, StormChoir):
            return self._resolve_storm(boss, events)
        if isinstance(boss, ChronoMantis):
            return self._resolve_chrono(boss, events)
        if isinstance(boss, VoidAngler):
            return self._resolve_void(boss, events)
        if isinstance(boss, IronGardener):
            return self._resolve_iron(boss, events)
        if isinstance(boss, MirrorSeraph):
            return self._resolve_mirror(boss, events)
        if isinstance(boss, SiegeLeviathan):
            return self._resolve_siege(boss, events)
        if isinstance(boss, NullWeaver):
            return self._resolve_null(boss, events)
        if isinstance(boss, ApexArbiter):
            return self._resolve_apex(boss, events)
        if boss.exposed_rounds:
            boss.exposed_rounds -= 1
            if boss.hp <= boss.max_hp * 2 // 3 and boss.exposed_rounds == 1:
                previous = boss.position
                boss.position = (11 if previous[0] <= 10 else 9, previous[1])
                events.append(f"boss_move:{previous[0]}:{previous[1]}:{boss.position[0]}:{boss.position[1]}")
            if not boss.exposed_rounds:
                boss.reflections = 0
                boss.used_reflectors.clear()
                boss.lunge_used = False
                boss.target = None
                events.append("boss_shield_restored")
            return 0.0
        if boss.lunge_target:
            destination = boss.lunge_target
            boss.lunge_target = None
            reward = 0.0
            if self._distance(self.player.position, destination) <= 1:
                reward += self._damage_entity(self.player, boss.lunge_damage, events, "boss_lunge")
            if self.player.position != destination:
                previous = boss.position
                boss.position = destination
                events.append(f"boss_move:{previous[0]}:{previous[1]}:{destination[0]}:{destination[1]}")
            boss.returning = True
            events.append("boss_lunge")
            return reward
        if boss.returning:
            previous = boss.position
            boss.position = (10, 8)
            boss.returning = False
            boss.target = None
            events.append(f"boss_move:{previous[0]}:{previous[1]}:10:8")
        if boss.target is None:
            phase_two = boss.hp <= boss.max_hp * 2 // 3
            cover_attack = phase_two and self.breakable_walls and boss.shots_fired % 3 == 0
            boss.target = (min(self.breakable_walls, key=lambda cell: (self._distance(boss.position, cell), cell))
                           if cover_attack else self.player.position)
            boss.sweep = phase_two and not cover_attack and boss.shots_fired % 3 == 1
            events.append("boss_cover_aim" if cover_attack else "boss_sweep_aim" if boss.sweep else "boss_aim")
            return 0.0
        path = self.boss_ray()
        endpoint = path[-1] if path else boss.position
        reflected = endpoint in self.reflectors - boss.used_reflectors
        events.append(f"boss_prism_shot:{boss.position[0]}:{boss.position[1]}:"
                      f"{endpoint[0]}:{endpoint[1]}:{int(reflected)}")
        if boss.sweep and not reflected:
            events.append("boss_prism_sweep")
        reward = 0.0
        if reflected:
            boss.used_reflectors.add(endpoint)
            boss.reflections += 1
            events.append(f"boss_reflect:{boss.reflections}")
            if boss.reflections == 3:
                boss.exposed_rounds = 3
                events.append("boss_shield_break")
                reward += 15
        elif endpoint in self.breakable_walls:
            self.breakable_walls.remove(endpoint)
            self.walls.remove(endpoint)
            events.append(f"boss_cover_break:{endpoint[0]}:{endpoint[1]}")
        elif self.player.position in self.prism_attack_cells():
            reward += self._damage_entity(self.player, boss.beam_damage, events, "boss_prism")
        boss.shots_fired += 1
        if boss.exposed_rounds == 0:
            if boss.reflections == 2 and not boss.lunge_used:
                target = (max(5, min(14, self.player.position[0])),
                          max(9, min(15, self.player.position[1] - 1)))
                if target in self.walls or target in self.reflectors:
                    target = self.player.position
                boss.lunge_target = target
                boss.lunge_used = True
                boss.target = None
                events.append(f"boss_lunge_aim:{target[0]}:{target[1]}")
                return reward
            previous = boss.position
            x = (10, 9, 11)[boss.shots_fired % 3]
            boss.position = (x, previous[1])
            boss.target = None
            if previous != boss.position:
                events.append(f"boss_move:{previous[0]}:{previous[1]}:{x}:{previous[1]}")
        else:
            boss.target = None
        return reward

    def _resolve_furnace(self, boss: FurnaceHydra, events: list[str]) -> float:
        for cell, rounds in list(self.furnace_burns.items()):
            if rounds == 1:
                del self.furnace_burns[cell]
                self.fires.remove(cell)
                events.append(f"furnace_burn_end:{cell[0]}:{cell[1]}")
            else:
                self.furnace_burns[cell] = rounds - 1
        if boss.exposed_rounds:
            boss.exposed_rounds -= 1
            if boss.hp <= boss.max_hp // 2 and boss.exposed_rounds == 1:
                previous = boss.position
                boss.position = (11 if previous[0] <= 10 else 9, 7)
                events.append(f"boss_move:{previous[0]}:{previous[1]}:{boss.position[0]}:7")
            if not boss.exposed_rounds:
                boss.valves_opened.clear()
                events.append("boss_shield_restored")
            return 0.0
        if boss.target is None:
            boss.attack_kind = "wave" if boss.attacks % 2 == 0 else "fireball"
            if boss.attack_kind == "wave":
                remaining = [x for x in (10, 7, 13) if x not in boss.valves_opened]
                boss.head_x = remaining[0]
                boss.target = (boss.head_x, 16)
            else:
                boss.target = self.player.position
            events.append(f"furnace_aim:{boss.attack_kind}:{boss.target[0]}:{boss.target[1]}")
            return 0.0
        reward = 0.0
        if boss.attack_kind == "wave":
            events.append(f"furnace_wave:{boss.head_x}")
            if self.player.position == (boss.head_x, 12):
                boss.valves_opened.add(boss.head_x)
                events.append(f"furnace_valve:{boss.head_x}")
                reward += 15
            elif 8 <= self.player.position[1] <= 16:
                offset = abs(self.player.position[0] - boss.head_x)
                if offset == 0 or (boss.hp <= boss.max_hp // 2 and offset == 1):
                    damage = boss.wave_damage if offset == 0 else 12
                    reward += self._damage_entity(self.player, damage, events, "furnace_wave")
        else:
            events.append(f"furnace_fireball:{boss.position[0]}:{boss.position[1]}:"
                          f"{boss.target[0]}:{boss.target[1]}")
            if self._distance(self.player.position, boss.target) <= 1:
                reward += self._damage_entity(self.player, boss.fireball_damage, events, "furnace_fireball")
            burn_cell = ((boss.target[0] + 1, boss.target[1])
                         if boss.target in self.coolant_valves else boss.target)
            if (boss.hp <= boss.max_hp // 2 and burn_cell not in self.walls | self.coolant_valves |
                    self.fires | self.medkits | self.bow_pickups | self.arrow_bundles):
                self.fires.add(burn_cell)
                self.furnace_burns[burn_cell] = 3
                events.append(f"furnace_ignite:{burn_cell[0]}:{burn_cell[1]}")
        boss.attacks += 1
        boss.target = None
        if len(boss.valves_opened) == 3:
            boss.exposed_rounds = 3
            boss.position = (10, 7)
            events.append("boss_shield_break")
        else:
            previous = boss.position
            boss.position = ((9, 10, 11)[boss.attacks % 3], 7)
            if previous != boss.position:
                events.append(f"boss_move:{previous[0]}:{previous[1]}:{boss.position[0]}:7")
        return reward

    def _resolve_storm(self, boss: StormChoir, events: list[str]) -> float:
        if boss.exposed_rounds:
            boss.exposed_rounds -= 1
            if not boss.exposed_rounds:
                if boss.hp <= boss.max_hp * 2 // 3 and self.relay_pads != {(7, 8), (13, 8)}:
                    self.relay_pads = {(7, 8), (13, 8)}
                    events.append("storm_relay_shift")
                events.append("boss_shield_restored")
            return 0.0
        if boss.target is None:
            boss.attack_kind = "chain" if boss.attacks % 2 == 0 else "surge"
            boss.target = self.player.position
            events.append(f"storm_aim:{boss.attack_kind}:{boss.target[0]}:{boss.target[1]}")
            return 0.0
        reward = 0.0
        if boss.attack_kind == "chain":
            chain = self.storm_chain()
            events.append("storm_chain:" + ":".join(f"{x}:{y}" for x, y in chain))
            grounded = (boss.target in self.relay_pads and self.player.position == boss.target and
                        len(chain) == 6)
            if grounded:
                boss.exposed_rounds = 4
                events.append("storm_grounded")
                events.append("boss_shield_break")
                reward += 15
            elif self.player.position == boss.target or self.player.position in chain[2:]:
                reward += self._damage_entity(self.player, boss.arc_damage, events, "storm_arc")
        else:
            events.append(f"storm_surge:{boss.target[0]}:{boss.target[1]}")
            if self._distance(self.player.position, boss.target) <= 1:
                damage = boss.surge_damage + (4 if boss.hp <= boss.max_hp * 2 // 3 else 0)
                reward += self._damage_entity(self.player, damage, events, "storm_surge")
        boss.attacks += 1
        boss.target = None
        if not boss.exposed_rounds:
            previous = boss.position
            boss.position = ((10, 11, 9)[boss.attacks % 3], 5)
            if previous != boss.position:
                events.append(f"boss_move:{previous[0]}:{previous[1]}:{boss.position[0]}:5")
        return reward

    def _resolve_chrono(self, boss: ChronoMantis, events: list[str]) -> float:
        if boss.exposed_rounds:
            boss.exposed_rounds -= 1
            if boss.hp <= boss.max_hp * 2 // 3 and boss.exposed_rounds == 2:
                previous = boss.position
                boss.position = (boss.position[0] + (1 if boss.position[0] <= 10 else -1), 7)
                events.append(f"boss_move:{previous[0]}:{previous[1]}:{boss.position[0]}:7")
            if not boss.exposed_rounds:
                if boss.hp <= boss.max_hp * 2 // 3 and self.time_anchors != {(9, 13), (14, 13)}:
                    self.time_anchors = {(9, 13), (14, 13)}
                    events.append("chrono_anchor_shift")
                boss.phase = "flank"
                events.append("boss_shield_restored")
            return 0.0
        if boss.phase == "flank":
            previous = boss.position
            launch_x = 12 if self.chrono_landing_x() == 9 else 11
            delta = max(-2, min(2, launch_x - boss.position[0]))
            x = boss.position[0] + delta
            y = (8 if boss.position[1] == 7 else 7) if not delta else boss.position[1]
            if (x, y) == self.player.position:
                y = 8 if y == 7 else 7
            boss.position = (x, y)
            boss.moves += 1
            if x != launch_x:
                events.append(f"boss_move:{previous[0]}:{previous[1]}:{x}:{y}")
                return 0.0
            boss.slash_target = self.player.position
            boss.phase = "slash"
            events.extend((f"boss_move:{previous[0]}:{previous[1]}:{x}:{y}",
                           f"chrono_slash_aim:{boss.slash_target[0]}:{boss.slash_target[1]}"))
            return 0.0
        if boss.phase == "slash":
            target = boss.slash_target
            reward = 0.0
            events.append(f"chrono_slash:{target[0]}:{target[1]}")
            if self._distance(self.player.position, target) <= 1:
                reward += self._damage_entity(self.player, boss.slash_damage, events, "chrono_slash")
            boss.leap_target = (self.chrono_landing_x(), 7)
            boss.leap_countdown = 2
            boss.phase = "leap"
            events.append(f"chrono_leap_aim:{boss.leap_target[0]}:{boss.leap_target[1]}")
            return reward
        if boss.leap_countdown > 1:
            boss.leap_countdown -= 1
            previous = boss.position
            boss.position = (boss.position[0], 8)
            if previous != boss.position:
                events.append(f"boss_move:{previous[0]}:{previous[1]}:{boss.position[0]}:8")
            events.append(f"chrono_leap_charge:{boss.leap_target[0]}:{boss.leap_target[1]}")
            return 0.0
        if self.player.position == boss.leap_target:
            boss.phase = "flank"
            boss.slash_target = boss.leap_target = None
            boss.leap_countdown = 0
            events.append("chrono_leap_cancel")
            return 0.0
        previous = boss.position
        boss.position = boss.leap_target
        boss.moves += 1
        events.append(f"boss_move:{previous[0]}:{previous[1]}:{boss.position[0]}:7")
        events.append(f"chrono_leap:{boss.position[0]}:7")
        reward = 0.0
        if self.player.position in self.time_anchors and self.player.position[0] == boss.position[0]:
            boss.exposed_rounds = 4
            events.extend(("chrono_anchor", "chrono_echo_replay", "boss_shield_break"))
            reward += 15
        elif self.player.position == boss.slash_target:
            reward += self._damage_entity(self.player, boss.echo_damage, events, "chrono_echo")
            events.append(f"chrono_echo:{boss.slash_target[0]}:{boss.slash_target[1]}")
        boss.phase = "flank"
        boss.slash_target = None
        boss.leap_target = None
        boss.leap_countdown = 0
        return reward

    def _resolve_void(self, boss: VoidAngler, events: list[str]) -> float:
        if boss.exposed_rounds:
            boss.exposed_rounds -= 1
            if not boss.exposed_rounds:
                boss.drained_nodes.clear()
                events.append("boss_shield_restored")
            return 0.0
        if boss.target is None:
            boss.attack_kind = "mine" if boss.attacks % 2 == 0 else "beam"
            remaining = self.gravity_nodes - boss.drained_nodes
            boss.target = (min(remaining, key=lambda cell: (self._distance(self.player.position, cell), cell))
                           if boss.attack_kind == "mine" and remaining else self.player.position)
            events.append(f"void_aim:{boss.attack_kind}:{boss.target[0]}:{boss.target[1]}")
            return 0.0
        target = boss.target
        reward = 0.0
        if boss.attack_kind == "mine":
            events.append(f"void_mine:{target[0]}:{target[1]}")
            if target in self.gravity_nodes - boss.drained_nodes and self.player.position == target:
                boss.drained_nodes.add(target)
                events.append(f"void_drain:{len(boss.drained_nodes)}")
                reward += 15
                if len(boss.drained_nodes) == 3:
                    boss.exposed_rounds = 4
                    events.append("boss_shield_break")
            elif self._distance(self.player.position, target) <= 2 and not self.player.invulnerable:
                x, y = self.player.position
                dx = (target[0] > x) - (target[0] < x)
                dy = (target[1] > y) - (target[1] < y)
                for destination in ((x + dx, y), (x, y + dy)):
                    if (destination != (x, y) and self.in_bounds(destination) and
                            destination not in self.walls | self.pits | self.fires | self.spikes and
                            not self.boss_at(destination)):
                        self.player.position = destination
                        events.append(f"void_pull:{x}:{y}:{destination[0]}:{destination[1]}")
                        break
        else:
            events.append(f"void_beam:{boss.position[0]}:{boss.position[1]}:{target[0]}:{target[1]}")
            if self._distance(self.player.position, target) <= 1:
                reward += self._damage_entity(self.player, boss.beam_damage, events, "void_beam")
        boss.attacks += 1
        boss.target = None
        if not boss.exposed_rounds:
            previous = boss.position
            boss.position = ((10, 11, 9)[boss.attacks % 3], 6)
            if previous != boss.position:
                events.append(f"boss_move:{previous[0]}:{previous[1]}:{boss.position[0]}:6")
        return reward

    def _resolve_iron(self, boss: IronGardener, events: list[str]) -> float:
        for seed, rounds in list(self.vine_seeds.items()):
            if rounds > 1 or self.player.position == seed:
                self.vine_seeds[seed] = max(1, rounds - 1)
            else:
                del self.vine_seeds[seed]
                self.vine_walls.add(seed)
                self.walls.add(seed)
                events.append(f"iron_vine_grow:{seed[0]}:{seed[1]}")
        if boss.exposed_rounds:
            boss.exposed_rounds -= 1
            if not boss.exposed_rounds:
                boss.refluxed_roots.clear()
                for vine in self.vine_walls:
                    self.walls.remove(vine)
                self.vine_walls.clear()
                self.vine_seeds = {(x, 10): 2 for x in (6, 9, 12, 15)}
                events.append("boss_shield_restored")
            return 0.0
        if boss.target is None:
            boss.attack_kind = "flame" if boss.attacks % 2 == 0 else "thorn"
            remaining = self.root_plates - {(x, 12) for x in boss.refluxed_roots}
            boss.target = (min(remaining, key=lambda cell: (self._distance(self.player.position, cell), cell))
                           if boss.attack_kind == "flame" and remaining else self.player.position)
            events.append(f"iron_aim:{boss.attack_kind}:{boss.target[0]}:{boss.target[1]}")
            return 0.0
        target = boss.target
        reward = 0.0
        if boss.attack_kind == "flame":
            x = target[0]
            vine = (x, 10)
            grown = vine in self.vine_walls
            events.append(f"iron_flame:{x}")
            if grown:
                self.vine_walls.remove(vine)
                self.walls.remove(vine)
                events.append(f"iron_vine_burn:{x}:10")
            if grown and self.player.position == target:
                boss.refluxed_roots.add(x)
                events.append(f"iron_reflux:{len(boss.refluxed_roots)}")
                reward += 15
                if len(boss.refluxed_roots) == 4:
                    boss.exposed_rounds = 4
                    events.append("boss_shield_break")
            elif self.player.position[0] == x and 8 <= self.player.position[1] <= 16:
                reward += self._damage_entity(self.player, boss.flame_damage, events, "iron_flame")
            if not (grown and self.player.position == target):
                self.vine_seeds[vine] = 2
        else:
            events.append(f"iron_thorn:{target[0]}:{target[1]}")
            if self._distance(self.player.position, target) <= 1:
                reward += self._damage_entity(self.player, boss.thorn_damage, events, "iron_thorn")
        boss.attacks += 1
        boss.target = None
        if not boss.exposed_rounds:
            previous = boss.position
            boss.position = ((10, 11, 9)[boss.attacks % 3], 6)
            if previous != boss.position:
                events.append(f"boss_move:{previous[0]}:{previous[1]}:{boss.position[0]}:6")
        return reward

    def mirror_ray(self) -> tuple[tuple[int, int], ...]:
        boss = self.boss
        if not isinstance(boss, MirrorSeraph) or not boss.mirrored_direction:
            return ()
        position = boss.position
        cells = []
        while True:
            position = self.add(position, boss.mirrored_direction)
            if not self.in_bounds(position) or position in self.walls:
                break
            cells.append(position)
            if position in self.mirror_locks:
                break
        return tuple(cells)

    def _resolve_mirror(self, boss: MirrorSeraph, events: list[str]) -> float:
        if boss.silence_rounds:
            boss.silence_rounds -= 1
        if boss.exposed_rounds:
            boss.exposed_rounds -= 1
            if not boss.exposed_rounds:
                boss.broken_locks.clear()
                events.append("boss_shield_restored")
            return 0.0
        if boss.copied_action is None:
            action = self.last_non_wait_action or "move_n"
            boss.copied_action = action
            direction = action[-1] if action[-1] in "nsew" else None
            boss.mirrored_direction = {"e": "w", "w": "e"}.get(direction, direction)
            # Sidestep to line up the mirrored shot; never shift a locked warning.
            desired_x = 11 if boss.mirrored_direction == "w" else 9 if boss.mirrored_direction == "e" else 10
            destination = (boss.position[0] + (desired_x > boss.position[0]) - (desired_x < boss.position[0]), 6)
            if destination != boss.position and destination != self.player.position:
                previous = boss.position
                boss.position = destination
                events.append(f"boss_move:{previous[0]}:{previous[1]}:{destination[0]}:{destination[1]}")
            path = self.mirror_ray()
            boss.target = path[-1] if path else None
            events.append(f"mirror_aim:{action}:{boss.mirrored_direction or '-'}")
            return 0.0
        action = boss.copied_action
        path = self.mirror_ray()
        reward = 0.0
        if action == "emp":
            boss.silence_rounds = 1
            events.append("mirror_silence")
        elif action == "heal":
            events.append("mirror_heal_echo")
        elif path:
            target = path[-1]
            events.append(f"mirror_shard:{action}:{target[0]}:{target[1]}")
            if target in self.mirror_locks and target not in boss.broken_locks:
                boss.broken_locks.add(target)
                reward += 15
                events.append(f"mirror_lock_break:{len(boss.broken_locks)}")
                if len(boss.broken_locks) == len(self.mirror_locks):
                    boss.exposed_rounds = 4
                    events.append("boss_shield_break")
            elif self.player.position in path:
                damage = boss.dash_damage if action.startswith("dash_") else boss.shard_damage
                reward += self._damage_entity(self.player, damage, events, "mirror_shard")
        boss.copied_action = boss.mirrored_direction = boss.target = None
        return reward

    def rail_path(self) -> tuple[tuple[int, int], ...]:
        boss = self.boss
        if not isinstance(boss, SiegeLeviathan) or boss.rail_target is None:
            return ()
        return (tuple((boss.rail_target, y) for y in range(self.config.height)) if boss.rail_axis == "v"
                else tuple((x, boss.rail_target) for x in range(self.config.width)))

    def rail_cover(self) -> tuple[int, int] | None:
        boss = self.boss
        if not isinstance(boss, SiegeLeviathan) or boss.rail_target is None:
            return None
        matches = (cover for cover in self.rail_covers.values()
                   if cover[0 if boss.rail_axis == "v" else 1] == boss.rail_target)
        return min(matches, key=lambda cell: cell[1 if boss.rail_axis == "v" else 0], default=None)

    def rail_threatens(self, position: tuple[int, int]) -> bool:
        boss = self.boss
        if not isinstance(boss, SiegeLeviathan) or boss.rail_target is None:
            return False
        if position[0 if boss.rail_axis == "v" else 1] != boss.rail_target:
            return False
        cover = self.rail_cover()
        return cover is None or (position[1] < cover[1] if boss.rail_axis == "v" else position[0] < cover[0])

    def _resolve_siege(self, boss: SiegeLeviathan, events: list[str]) -> float:
        for origin, rounds in list(self.rail_rebuilds.items()):
            if rounds > 1 or origin in self.rail_covers.values() or origin == self.player.position:
                self.rail_rebuilds[origin] = max(1, rounds - 1)
            else:
                del self.rail_rebuilds[origin]
                self.rail_covers[origin] = origin
                events.append(f"rail_cover_rebuild:{origin[0]}:{origin[1]}")
        if boss.exposed_rounds:
            boss.exposed_rounds -= 1
            if not boss.exposed_rounds:
                boss.broken_locks.clear()
                events.append("boss_shield_restored")
            return 0.0
        if boss.rail_target is None:
            # Heavy lateral repositioning only between rail volleys.
            desired_x = min(12, max(8, self.player.position[0]))
            destination = (boss.position[0] + (desired_x > boss.position[0]) - (desired_x < boss.position[0]), 5)
            if destination != boss.position and destination != self.player.position:
                previous = boss.position
                boss.position = destination
                events.append(f"boss_move:{previous[0]}:{previous[1]}:{destination[0]}:{destination[1]}")
            boss.rail_axis = "h" if boss.shots % 4 == 3 else "v"
            boss.rail_target = self.player.position[1 if boss.rail_axis == "h" else 0]
            boss.charge = 2
            events.append(f"rail_aim:{boss.rail_axis}:{boss.rail_target}:2")
            return 0.0
        boss.charge -= 1
        if boss.charge:
            events.append(f"rail_charge:{boss.rail_axis}:{boss.rail_target}:1")
            return 0.0
        target, axis = boss.rail_target, boss.rail_axis
        cover = self.rail_cover()
        reward = 0.0
        events.append(f"rail_fire:{axis}:{target}")
        if self.rail_threatens(self.player.position):
            reward += self._damage_entity(self.player, boss.rail_damage, events, "railgun")
        if cover:
            origin = next(origin for origin, position in self.rail_covers.items() if position == cover)
            del self.rail_covers[origin]
            self.rail_rebuilds[origin] = 2
            events.append(f"rail_cover_break:{cover[0]}:{cover[1]}")
            lock = (target, 7)
            if axis == "v" and lock in self.rail_locks and lock not in boss.broken_locks:
                boss.broken_locks.add(lock)
                reward += 15
                events.append(f"rail_lock_break:{len(boss.broken_locks)}")
                if len(boss.broken_locks) == len(self.rail_locks):
                    boss.exposed_rounds = 6
                    events.append("boss_shield_break")
        boss.shots += 1
        boss.rail_target = None
        for origin, position in list(self.rail_covers.items()):
            x = ((7 + boss.shots % 2) if origin[0] == 7 else
                 (13 - boss.shots % 2) if origin[0] == 13 else 10 + boss.shots % 2)
            destination = (x, origin[1])
            if destination != self.player.position and destination not in self.rail_covers.values():
                self.rail_covers[origin] = destination
                if destination != position:
                    events.append(f"rail_cover_move:{position[0]}:{position[1]}:{x}:{origin[1]}")
        return reward

    def _resolve_null(self, boss: NullWeaver, events: list[str]) -> float:
        if self.null_void:
            self.null_void.clear()
            events.append("null_floor_restore")
        if boss.exposed_rounds:
            boss.exposed_rounds -= 1
            if not boss.exposed_rounds:
                boss.node_index = 0
                events.append("boss_shield_restored")
            return 0.0
        if boss.warp_target:
            destination = boss.warp_target
            boss.warp_target = None
            if destination != self.player.position:
                previous = boss.position
                boss.position = destination
                events.extend((f"boss_move:{previous[0]}:{previous[1]}:{destination[0]}:{destination[1]}",
                               "null_warp"))
        elif boss.cycles % 3 == 0:
            anchors = ((8, 5), (12, 5), (8, 7), (12, 7))
            choices = (cell for cell in anchors if cell != boss.position and cell != self.player.position)
            boss.warp_target = min(choices, key=lambda cell: (self._distance(cell, self.player.position), cell))
            events.append(f"null_warp_aim:{boss.warp_target[0]}:{boss.warp_target[1]}")
        boss.blocked_kind = ("move", "melee", "ranged", "skill")[boss.cycles % 4]
        boss.cycles += 1
        events.append(f"null_block:{boss.blocked_kind}")
        if not boss.erase_targets:
            x, y = self.player.position
            occupied = (self.walls | self.pits | set(self.null_nodes) | self.medkits |
                        self.energy_cells | self.bow_pickups | self.pistol_pickups |
                        self.arrow_bundles | self.barrels)
            boss.erase_targets = {(x + dx, y) for dx in (-1, 0, 1)
                                  if self.in_bounds((x + dx, y)) and (x + dx, y) not in occupied}
            boss.erase_countdown = 2
            events.append(f"null_mark:{y}:2")
            return 0.0
        boss.erase_countdown -= 1
        if boss.erase_countdown:
            events.append(f"null_countdown:{boss.erase_countdown}")
            return 0.0
        self.null_void = set(boss.erase_targets)
        boss.erase_targets.clear()
        reward = 0.0
        if self.player.position in self.null_void:
            reward += self._damage_entity(self.player, boss.fracture_damage, events, "null_fracture")
            safe = [self.add(self.player.position, direction) for direction in DIRECTIONS
                    if self.in_bounds(self.add(self.player.position, direction)) and
                    self.add(self.player.position, direction) not in self.null_void | self.walls | self.pits and
                    not self.boss_at(self.add(self.player.position, direction))]
            if safe:
                self.player.position = min(safe)
                events.append(f"null_displace:{self.player.position[0]}:{self.player.position[1]}")
        events.append(f"null_fracture:{len(self.null_void)}")
        return reward

    def _apex_charge_path(self, boss: ApexArbiter, target: tuple[int, int]) -> tuple[tuple[int, int], ...]:
        path = []
        for cell in ray_cells(boss.position, target):
            if cell in self.walls | self.pits:
                break
            path.append(cell)
        return tuple(path)

    def _apex_reposition(self, boss: ApexArbiter, events: list[str]) -> None:
        # Each phase has a different objective: flank, recenter for reflection,
        # then pursue. This happens before the danger cells are locked.
        if boss.seals < 3:
            desired = ((9, 5), (11, 5), (10, 5))[boss.seals]
        else:
            x, y = self.player.position
            desired = (min(12, max(8, x)), min(10, max(5, y - 2)))
        x, y = boss.position
        dx = (desired[0] > x) - (desired[0] < x)
        dy = (desired[1] > y) - (desired[1] < y)
        candidates = ((x + dx, y), (x, y + dy)) if boss.seals < 3 else ((x, y + dy), (x + dx, y))
        for destination in candidates:
            if (destination != boss.position and destination != self.player.position and
                    self.in_bounds(destination) and destination not in self.walls | self.pits | self.apex_cage):
                boss.position = destination
                events.append(f"boss_move:{x}:{y}:{destination[0]}:{destination[1]}")
                break

    def _resolve_apex(self, boss: ApexArbiter, events: list[str]) -> float:
        if boss.exposed_rounds:
            boss.exposed_rounds -= 1
            if not boss.exposed_rounds:
                events.append("boss_shield_restored")
            return 0.0
        if not boss.countdown:
            self._apex_reposition(boss, events)
            boss.kind = (("cage", "barrage", "charge", "gravity")[boss.seals]
                         if boss.seals < 4 else ("cage_barrage", "charge_gravity", "verdict")[boss.finale_cycles % 3])
            boss.target = self.player.position
            boss.gate_broken = False
            boss.appeal = None
            if boss.kind in ("cage", "cage_barrage"):
                directions = ("n", "e", "w", "s")
                boss.gate = next((self.add(boss.target, d) for d in directions
                                  if self.in_bounds(self.add(boss.target, d)) and
                                  self.add(boss.target, d) not in self.walls | self.pits | self.barrels and
                                  not self.boss_at(self.add(boss.target, d))), None)
                boss.danger = {boss.target}
                if boss.kind == "cage_barrage" and boss.gate:
                    safe_column = boss.gate[0] % 3
                    boss.danger |= {(x, y) for y in range(5, 18) for x in range(2, 18)
                                    if x % 3 != safe_column and (x, y) not in self.walls | self.pits}
                boss.countdown = 2
            elif boss.kind == "barrage":
                safe_column = 0 if boss.seals < 4 else (boss.finale_cycles // 2 + 1) % 3
                boss.danger = {(x, y) for y in range(5, 18) for x in range(2, 18)
                               if x % 3 != safe_column and (x, y) not in self.walls | self.pits}
                boss.countdown = 1
            elif boss.kind in ("charge", "charge_gravity"):
                boss.danger = set(self._apex_charge_path(boss, boss.target))
                if boss.kind == "charge_gravity":
                    x, y = boss.target
                    boss.danger |= {(x + dx, y + dy) for dx in range(-1, 2) for dy in range(-1, 2)
                                    if abs(dx) + abs(dy) <= 1}
                boss.countdown = 2 if boss.kind == "charge_gravity" else 1
            elif boss.kind == "verdict":
                boss.appeal = next((self.add(boss.target, direction) for direction in ("n", "e", "w", "s")
                                    if self.in_bounds(self.add(boss.target, direction)) and
                                    self.add(boss.target, direction) not in self.walls | self.pits | self.barrels and
                                    not self.boss_at(self.add(boss.target, direction))), None)
                x, y = boss.target
                boss.danger = {(x + dx, y + dy) for dx in range(-2, 3) for dy in range(-2, 3)
                               if abs(dx) + abs(dy) <= 2 and (x + dx, y + dy) != boss.appeal and
                               self.in_bounds((x + dx, y + dy))}
                boss.countdown = 2
            else:
                boss.danger = {(boss.target[0] + dx, boss.target[1] + dy)
                               for dx in range(-1, 2) for dy in range(-1, 2)
                               if abs(dx) + abs(dy) <= 1}
                boss.countdown = 2
            events.append(f"apex_aim:{boss.kind}:{boss.countdown}")
            return 0.0
        if boss.countdown == 2:
            boss.countdown = 1
            if boss.kind in ("cage", "cage_barrage") and boss.target and boss.gate:
                x, y = boss.target
                self.apex_cage = {(x + dx, y + dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)
                                  if (dx or dy) and self.in_bounds((x + dx, y + dy)) and
                                  (x + dx, y + dy) not in self.walls | self.pits | self.barrels and
                                  not self.boss_at((x + dx, y + dy))}
                # A doorway always remains breakable even when the ring touches existing terrain.
                events.append(f"apex_cage:{boss.gate[0]}:{boss.gate[1]}")
            elif boss.kind in ("gravity", "charge_gravity") and boss.target and self.player.position not in self.apex_seals:
                dx = (boss.target[0] > self.player.position[0]) - (boss.target[0] < self.player.position[0])
                dy = (boss.target[1] > self.player.position[1]) - (boss.target[1] < self.player.position[1])
                direction = (dx, 0) if dx else (0, dy)
                pulled = (self.player.position[0] + direction[0], self.player.position[1] + direction[1])
                if (direction != (0, 0) and self.in_bounds(pulled) and
                        pulled not in self.walls | self.pits | self.barrels | self.apex_cage and
                        not self.boss_at(pulled)):
                    self.player.position = pulled
                    events.append(f"apex_pull:{pulled[0]}:{pulled[1]}")
            events.append(f"apex_charge:{boss.kind}:1")
            return 0.0
        reward = 0.0
        if boss.kind == "cage":
            if not boss.gate_broken and self.player.position in boss.danger:
                reward += self._damage_entity(self.player, 26, events, "apex_cage")
            if boss.gate_broken and boss.seals == 0:
                boss.seals += 1
                events.append("apex_seal:1")
            self.apex_cage.clear()
        elif boss.kind == "barrage":
            if self.player.position in boss.danger:
                reward += self._damage_entity(self.player, 16, events, "apex_barrage")
            if boss.seals == 1 and self.player.position == self.apex_seals[1]:
                boss.seals += 1
                events.append("apex_seal:2")
        elif boss.kind == "charge":
            reflected = self.apex_seals[2] in boss.danger and boss.seals == 2
            if self.player.position in boss.danger and not reflected:
                reward += self._damage_entity(self.player, 32, events, "apex_charge")
            if reflected:
                boss.seals += 1
                events.append("apex_seal:3")
        elif boss.kind == "gravity":
            if self.player.position in boss.danger and self.player.position != self.apex_seals[3]:
                reward += self._damage_entity(self.player, 24, events, "apex_gravity")
            if boss.seals == 3 and boss.target == self.apex_seals[3] and self.player.position == boss.target:
                boss.seals += 1
                events.append("apex_seal:4")
        elif boss.kind == "cage_barrage":
            safe_column = boss.gate[0] % 3 if boss.gate else boss.target[0] % 3
            cage_hit = not boss.gate_broken and self.player.position == boss.target
            barrage_hit = (self.player.position[0] % 3 != safe_column and
                           5 <= self.player.position[1] < 18)
            if cage_hit or barrage_hit:
                reward += self._damage_entity(self.player, min(36, 26 * cage_hit + 16 * barrage_hit),
                                              events, "apex_cage_barrage")
            self.apex_cage.clear()
        elif boss.kind == "charge_gravity":
            path = set(self._apex_charge_path(boss, boss.target))
            charge_hit = self.player.position in path
            gravity_hit = self._distance(self.player.position, boss.target) <= 1
            if charge_hit or gravity_hit:
                reward += self._damage_entity(self.player, min(36, 32 * charge_hit + 24 * gravity_hit),
                                              events, "apex_charge_gravity")
        else:
            if boss.appeal and self.player.position == boss.appeal:
                events.append("apex_appeal")
            elif self.player.position in boss.danger:
                reward += self._damage_entity(self.player, 36, events, "apex_verdict")
        safe_column = (boss.gate[0] % 3 if boss.gate else boss.target[0] % 3)
        charge_path = (self._apex_charge_path(boss, boss.target)
                       if boss.kind in ("charge", "charge_gravity") else ())
        endpoint = charge_path[-1] if charge_path else (boss.position if boss.kind in ("charge", "charge_gravity") else boss.target)
        events.append(f"apex_fire:{boss.kind}:{boss.target[0]}:{boss.target[1]}:{safe_column}:{endpoint[0]}:{endpoint[1]}")
        if charge_path:
            # Rush along the warned trace, stopping short of the player or a
            # reflecting seal. The new position persists after the animation.
            reflecting = "apex_seal:3" in events
            stop = next((index for index, cell in enumerate(charge_path)
                         if cell == self.player.position or
                         (reflecting and cell == self.apex_seals[2])), len(charge_path))
            # Early law keeps the reflector lane readable; the finale unleashes
            # a longer rush. A reflected charge may reach the pylon doorstep.
            rush_limit = 5 if boss.seals >= 4 or reflecting else 2
            traversed = charge_path[:min(stop, rush_limit)]
            if traversed:
                destination = traversed[-1]
                previous = boss.position
                boss.position = destination
                events.append(f"boss_move:{previous[0]}:{previous[1]}:{destination[0]}:{destination[1]}")
        boss.countdown = 0
        boss.danger.clear()
        boss.target = boss.gate = boss.appeal = None
        if boss.seals == 4:
            if "apex_seal:4" in events or "apex_appeal" in events:
                if "apex_appeal" in events:
                    boss.finale_cycles += 1
                boss.exposed_rounds = 6
                events.append("boss_shield_break")
            else:
                boss.finale_cycles += 1
        return reward

    def _push_entity(self, enemy: Enemy, direction: str, events: list[str]) -> float:
        destination = self.add(enemy.position, direction)
        events.append(f"shove:{direction}")
        if destination in self.walls:
            return self._damage_entity(enemy, self.config.collision_damage, events, "wall")
        if destination in self.pits:
            events.append("pit_fall")
            return self._damage_entity(enemy, enemy.hp, events, "pit")
        blocker = self.enemy_at(destination)
        if blocker:
            reward = self._damage_entity(blocker, self.config.collision_damage, events, "collision")
            if enemy in self.enemies:
                reward += self._damage_entity(enemy, self.config.collision_damage, events, "collision")
            return reward
        if destination in self.barrels:
            return self._explode_barrel(destination, events)
        enemy.position = destination
        if destination in self.fires:
            return self._damage_entity(enemy, self.config.fire_damage, events, "fire")
        if destination in self.spikes:
            return self._damage_entity(enemy, self.config.spike_damage, events, "spike")
        return 0.0

    def _plan_enemy_intents(self, enemies: list[Enemy] | None = None) -> None:
        occupied = {enemy.position for enemy in self.enemies}
        for enemy in enemies or self.enemies:
            if enemy.stunned:
                enemy.intent = Intent(IntentType.WAIT, countdown=1)
                continue
            distance = self._distance(enemy.position, self.player.position)
            if enemy.enemy_type == EnemyType.BOMBER and distance <= self.config.bomber_radius + 1:
                enemy.intent = Intent(IntentType.EXPLODE, countdown=2, power=self.config.bomber_damage)
                continue
            if distance == 1:
                direction = next(name for name, delta in DIRECTIONS.items()
                                 if self.add(enemy.position, name) == self.player.position)
                enemy.intent = Intent(IntentType.MELEE, direction,
                                      self.config.enemy_move_interval + ENEMY_MOVE_DELAY[enemy.enemy_type],
                                      max(1, self.config.enemy_damage + ENEMY_MELEE_BONUS[enemy.enemy_type]))
                continue
            dx = self.player.position[0] - enemy.position[0]
            dy = self.player.position[1] - enemy.position[1]
            if enemy.enemy_type == EnemyType.CHARGER and (dx == 0 or dy == 0):
                direction = "e" if dx > 0 else "w" if dx < 0 else "s" if dy > 0 else "n"
                if self._clear_shot_to_player(enemy.position, direction, self.fires | self.spikes):
                    enemy.intent = Intent(IntentType.CHARGE, direction,
                                          max(1, self._enemy_move_interval(enemy) - 1),
                                          self.config.charger_damage)
                    continue
            if (enemy.enemy_type == EnemyType.ARCHER and self.round > self.config.spawn_protection_rounds
                    and (dx == 0 or dy == 0)):
                direction = "e" if dx > 0 else "w" if dx < 0 else "s" if dy > 0 else "n"
                if self._clear_shot_to_player(enemy.position, direction):
                    countdown = max(1, self.config.archer_countdown -
                                    int(self.config.difficulty_level >= ENEMY_SPEED_LEVEL[EnemyType.ARCHER]))
                    enemy.intent = Intent(IntentType.SHOOT, direction, countdown,
                                          self.config.archer_damage)
                    continue
            direction = self._enemy_path_direction(enemy.position, occupied - {enemy.position})
            enemy.intent = (Intent(IntentType.MOVE, direction, self._enemy_move_interval(enemy))
                            if direction else Intent(IntentType.WAIT))

    def _enemy_path_direction(self, start: tuple[int, int], occupied: set[tuple[int, int]]) -> str | None:
        queue = deque((self.add(start, direction), direction) for direction in DIRECTIONS)
        visited = {start}
        while queue:
            position, first = queue.popleft()
            if (position in visited or not self.in_bounds(position) or position in self.walls or position in self.fires or
                    position in self.spikes or position in self.pits or
                    position in self.barrels or position in occupied or position == self.player.position):
                continue
            if self._distance(position, self.player.position) == 1:
                return first
            visited.add(position)
            queue.extend((self.add(position, direction), first) for direction in DIRECTIONS)
        return None

    def _enemy_move_interval(self, enemy: Enemy) -> int:
        threshold = ENEMY_SPEED_LEVEL[enemy.enemy_type]
        speedup = 0 if self.config.difficulty_level < threshold else 1 + (self.config.difficulty_level - threshold) // 12
        return max(1, self.config.enemy_move_interval + ENEMY_MOVE_DELAY[enemy.enemy_type] - speedup)

    def _resolve_enemy_intents(self, events: list[str]) -> float:
        reward = 0.0
        occupied = {enemy.position for enemy in self.enemies}
        resolved: list[Enemy] = []
        for enemy in list(self.enemies):
            if enemy not in self.enemies:
                continue
            intent = enemy.intent
            if intent is None:
                resolved.append(enemy)
                continue
            intent.countdown -= 1
            if intent.countdown > 0:
                continue
            if enemy.stunned:
                enemy.stunned -= 1
            elif intent.kind == IntentType.MELEE and intent.direction:
                if self.add(enemy.position, intent.direction) == self.player.position:
                    reward += self._damage_entity(self.player, intent.power, events, "enemy")
                    events.append("enemy_melee")
                    events.append(f"enemy_attack_at:{enemy.position[0]}:{enemy.position[1]}:"
                                  f"{self.player.position[0]}:{self.player.position[1]}")
            elif intent.kind == IntentType.MOVE and intent.direction:
                target = self.add(enemy.position, intent.direction)
                if (self.in_bounds(target) and target not in self.walls and target not in self.fires and
                        target not in self.spikes and target not in self.pits and
                        target not in self.barrels and target not in occupied and
                        target != self.player.position):
                    occupied.remove(enemy.position)
                    enemy.position = target
                    occupied.add(target)
                    events.append("enemy_move")
            elif intent.kind == IntentType.CHARGE and intent.direction:
                reward += self._resolve_charge(enemy, intent.direction, occupied, events)
            elif intent.kind == IntentType.SHOOT and intent.direction:
                reward += self._resolve_shot(enemy, intent.direction, intent.power, events)
            elif intent.kind == IntentType.EXPLODE:
                reward += self._resolve_explosion(enemy, intent.power, events)
                occupied.discard(enemy.position)
            if enemy in self.enemies:
                resolved.append(enemy)
        if resolved:
            self._plan_enemy_intents(resolved)
        return reward

    def _explode_barrel(self, center: tuple[int, int], events: list[str]) -> float:
        if center not in self.barrels:
            return 0.0
        self.barrels.remove(center)
        events.append("barrel_explode")
        events.append(f"explosion_at:{center[0]}:{center[1]}:{self.config.barrel_radius}")
        reward = 0.0
        if self._distance(center, self.player.position) <= self.config.barrel_radius:
            reward += self._damage_entity(self.player, self.config.barrel_damage, events, "barrel")
        for enemy in list(self.enemies):
            if enemy not in self.enemies:
                continue
            if self._distance(center, enemy.position) <= self.config.barrel_radius:
                if enemy.enemy_type == EnemyType.BOMBER:
                    reward += self._resolve_explosion(enemy, self.config.bomber_damage, events)
                else:
                    reward += self._damage_entity(enemy, self.config.barrel_damage, events, "barrel")
        for barrel in list(self.barrels):
            if self._distance(center, barrel) <= self.config.barrel_radius:
                reward += self._explode_barrel(barrel, events)
        return reward

    def _clear_shot_to_player(self, origin: tuple[int, int], direction: str,
                              blocked: set[tuple[int, int]] | None = None) -> bool:
        target = origin
        while True:
            target = self.add(target, direction)
            if not self.in_bounds(target) or target in self.walls or (blocked and target in blocked):
                return False
            if target == self.player.position:
                return True

    def _resolve_shot(self, archer: Enemy, direction: str, damage: int,
                      events: list[str]) -> float:
        origin = archer.position
        target = archer.position
        while True:
            target = self.add(target, direction)
            if not self.in_bounds(target) or target in self.walls:
                events.append(f"archer_shot:{direction}:{origin[0]}:{origin[1]}:{target[0]}:{target[1]}")
                events.append("shot_blocked")
                return 0.0
            if target in self.barrels:
                events.append(f"archer_shot:{direction}:{origin[0]}:{origin[1]}:{target[0]}:{target[1]}")
                events.append("archer_barrel_hit")
                return self._explode_barrel(target, events)
            victim = self.enemy_at(target)
            if victim:
                events.append(f"archer_shot:{direction}:{origin[0]}:{origin[1]}:{target[0]}:{target[1]}")
                reward = self._damage_entity(victim, damage, events, "shot")
                events.append("archer_friendly_fire")
                return reward
            if target == self.player.position:
                events.append(f"archer_shot:{direction}:{origin[0]}:{origin[1]}:{target[0]}:{target[1]}")
                events.append("archer_hit")
                return self._damage_entity(self.player, damage, events, "shot")

    def _resolve_explosion(self, bomber: Enemy, damage: int, events: list[str]) -> float:
        center = bomber.position
        self.enemies.remove(bomber)
        events.append("bomber_explode")
        events.append(f"explosion_at:{center[0]}:{center[1]}:{self.config.bomber_radius}")
        reward = 0.0
        if self._distance(center, self.player.position) <= self.config.bomber_radius:
            reward += self._damage_entity(self.player, damage, events, "explosion")
        for enemy in list(self.enemies):
            if enemy not in self.enemies:
                continue
            if self._distance(center, enemy.position) <= self.config.bomber_radius:
                reward += self._damage_entity(enemy, damage, events, "explosion")
        for barrel in list(self.barrels):
            if self._distance(center, barrel) <= self.config.bomber_radius:
                reward += self._explode_barrel(barrel, events)
        return reward

    def _resolve_charge(self, enemy: Enemy, direction: str, occupied: set[tuple[int, int]],
                        events: list[str]) -> float:
        reward = 0.0
        for _ in range(self.config.charger_range):
            target = self.add(enemy.position, direction)
            if not self.in_bounds(target) or target in self.walls or target in self.fires:
                events.append("charge_blocked")
                break
            if target in self.spikes:
                events.append("charge_blocked")
                break
            if target in self.pits:
                occupied.discard(enemy.position)
                events.append("pit_fall")
                reward += self._damage_entity(enemy, enemy.hp, events, "pit")
                break
            if target in self.barrels:
                reward += self._explode_barrel(target, events)
                break
            victim = self.enemy_at(target)
            if victim:
                reward += self._damage_entity(victim, self.config.collision_damage, events, "collision")
                events.append(f"enemy_collision:{self.config.collision_damage}")
                if victim not in self.enemies:
                    occupied.discard(target)
                    events.append("friendly_fire_kill")
                break
            if target == self.player.position:
                reward += self._damage_entity(self.player, self.config.charger_damage, events, "charge")
                events.append("charger_hit")
                events.append(f"enemy_attack_at:{enemy.position[0]}:{enemy.position[1]}:"
                              f"{self.player.position[0]}:{self.player.position[1]}")
                break
            occupied.remove(enemy.position)
            enemy.position = target
            occupied.add(target)
            events.append("charger_move")
        return reward

    @staticmethod
    def _distance(a: tuple[int, int], b: tuple[int, int]) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def imminent_threats(self, position: tuple[int, int] | None = None) -> tuple[tuple[str, int], ...]:
        position = position or self.player.position
        threats: list[tuple[str, int]] = []
        if isinstance(self.boss, PrismWarden) and self.boss.target:
            path = self.boss_ray()
            if path and path[-1] not in self.walls | self.reflectors and position in self.prism_attack_cells():
                threats.append(("prism_warden/beam", self.boss.beam_damage))
        if isinstance(self.boss, PrismWarden) and self.boss.lunge_target and self._distance(position, self.boss.lunge_target) <= 1:
            threats.append(("prism_warden/lunge", self.boss.lunge_damage))
        if isinstance(self.boss, FurnaceHydra) and self.boss.target:
            if self.boss.attack_kind == "wave" and 8 <= position[1] <= 16:
                offset = abs(position[0] - self.boss.head_x)
                if position != (self.boss.head_x, 12) and (
                        offset == 0 or (self.boss.hp <= self.boss.max_hp // 2 and offset == 1)):
                    threats.append(("furnace_hydra/wave", self.boss.wave_damage if offset == 0 else 12))
            elif self.boss.attack_kind == "fireball" and self._distance(position, self.boss.target) <= 1:
                threats.append(("furnace_hydra/fireball", self.boss.fireball_damage))
        if isinstance(self.boss, StormChoir) and self.boss.target:
            if self.boss.attack_kind == "chain":
                chain = self.storm_chain()
                if ((position == self.boss.target or position in chain[2:]) and
                        not (position == self.boss.target and position in self.relay_pads and len(chain) == 6)):
                    threats.append(("storm_choir/arc", self.boss.arc_damage))
            elif self._distance(position, self.boss.target) <= 1:
                damage = self.boss.surge_damage + (4 if self.boss.hp <= self.boss.max_hp * 2 // 3 else 0)
                threats.append(("storm_choir/surge", damage))
        if isinstance(self.boss, ChronoMantis):
            if self.boss.phase == "slash" and self._distance(position, self.boss.slash_target) <= 1:
                threats.append(("chrono_mantis/slash", self.boss.slash_damage))
            if self.boss.phase == "leap" and position == self.boss.slash_target:
                threats.append(("chrono_mantis/echo", self.boss.echo_damage))
        if isinstance(self.boss, VoidAngler) and self.boss.attack_kind == "beam" and self.boss.target:
            if self._distance(position, self.boss.target) <= 1:
                threats.append(("void_angler/beam", self.boss.beam_damage))
        if isinstance(self.boss, IronGardener) and self.boss.target:
            if self.boss.attack_kind == "flame" and position[0] == self.boss.target[0] and 8 <= position[1] <= 16:
                if not (position == self.boss.target and self.iron_root_ready(position[0])):
                    threats.append(("iron_gardener/flame", self.boss.flame_damage))
            elif self.boss.attack_kind == "thorn" and self._distance(position, self.boss.target) <= 1:
                threats.append(("iron_gardener/thorn", self.boss.thorn_damage))
        if isinstance(self.boss, MirrorSeraph) and self.boss.copied_action and position in self.mirror_ray():
            path = self.mirror_ray()
            if path[-1] not in self.mirror_locks or path[-1] in self.boss.broken_locks:
                damage = (self.boss.dash_damage if self.boss.copied_action.startswith("dash_")
                          else self.boss.shard_damage)
                threats.append(("mirror_seraph/shard", damage))
        if isinstance(self.boss, SiegeLeviathan) and self.boss.rail_target is not None and self.boss.charge <= 1:
            if self.rail_threatens(position):
                threats.append(("siege_leviathan/railgun", self.boss.rail_damage))
        if isinstance(self.boss, NullWeaver) and self.boss.erase_countdown == 1:
            if position in self.boss.erase_targets:
                threats.append(("null_weaver/fracture", self.boss.fracture_damage))
        if isinstance(self.boss, ApexArbiter) and self.boss.countdown == 1:
            if position in self.boss.danger:
                power = {"cage": 26, "barrage": 16, "charge": 32, "gravity": 24,
                         "cage_barrage": 36, "charge_gravity": 36, "verdict": 36}[self.boss.kind]
                if not (self.boss.kind == "cage" and self.boss.gate_broken or
                        self.boss.kind == "cage_barrage" and self.boss.gate_broken and
                        self.boss.gate is not None and
                        position[0] % 3 == self.boss.gate[0] % 3 or
                        self.boss.kind == "charge" and self.apex_seals[2] in self.boss.danger or
                        self.boss.kind == "gravity" and position == self.apex_seals[3]):
                    threats.append((f"apex_arbiter/{self.boss.kind}", power))
        for enemy in self.enemies:
            intent = enemy.intent
            if not intent or intent.countdown > 1:
                continue
            label = f"{enemy.enemy_type.value}@{enemy.position[0]},{enemy.position[1]}"
            if intent.kind == IntentType.EXPLODE:
                if self._distance(enemy.position, position) <= self.config.bomber_radius:
                    threats.append((f"{label}/blast", intent.power))
            elif intent.kind == IntentType.MELEE and intent.direction:
                if self.add(enemy.position, intent.direction) == position:
                    threats.append((f"{label}/melee", intent.power))
            elif intent.kind == IntentType.CHARGE and intent.direction:
                target = enemy.position
                for _ in range(self.config.charger_range):
                    target = self.add(target, intent.direction)
                    if not self.in_bounds(target) or target in self.walls or target in self.pits:
                        break
                    if target == position:
                        threats.append((f"{label}/charge", intent.power))
                        break
                    if self.enemy_at(target):
                        break
            elif intent.kind == IntentType.SHOOT and intent.direction:
                target = enemy.position
                while True:
                    target = self.add(target, intent.direction)
                    if not self.in_bounds(target) or target in self.walls:
                        break
                    if target == position:
                        threats.append((f"{label}/shot", intent.power))
                        break
                    if self.enemy_at(target):
                        break
        return tuple(threats)

    def observation(self) -> dict:
        return {
            "seed": self.seed,
            "tick": self.tick,
            "round": self.round,
            "ap_remaining": self.ap_remaining,
            "hp": self.player.hp,
            "score": self.score,
            "position": self.player.position,
            "previous_position": self.previous_player_position,
            "last_action": self.last_action,
            "last_non_wait_action": self.last_non_wait_action,
            "medkits_carried": self.player.medkits,
            "loadout": (self.player.loadout.bow, self.player.loadout.pistol,
                        self.player.loadout.arrows, self.player.loadout.energy),
            "cooldowns": tuple(sorted(self.player.cooldowns.items())),
            "walls": tuple(sorted(self.walls)),
            "enemies": tuple((enemy.position, enemy.hp) for enemy in self.enemies),
            "boss": (("prism", self.boss.position, self.boss.hp, self.boss.reflections,
                      tuple(sorted(self.boss.used_reflectors)), self.boss.shots_fired,
                      self.boss.exposed_rounds, self.boss.target, self.boss.lunge_target,
                      self.boss.returning) if isinstance(self.boss, PrismWarden) else
                     ("furnace", self.boss.position, self.boss.hp, tuple(sorted(self.boss.valves_opened)),
                      self.boss.exposed_rounds, self.boss.attack_kind, self.boss.target,
                      self.boss.attacks) if isinstance(self.boss, FurnaceHydra) else
                     ("storm", self.boss.position, self.boss.hp, self.boss.exposed_rounds,
                      self.boss.attack_kind, self.boss.target, self.boss.attacks) if isinstance(self.boss, StormChoir) else
                     ("chrono", self.boss.position, self.boss.hp, self.boss.exposed_rounds,
                      self.boss.phase, self.boss.slash_target, self.boss.leap_target,
                      self.boss.leap_countdown, self.boss.moves) if isinstance(self.boss, ChronoMantis) else
                     ("void", self.boss.position, self.boss.hp, self.boss.exposed_rounds,
                      tuple(sorted(self.boss.drained_nodes)), self.boss.attack_kind,
                      self.boss.target, self.boss.attacks) if isinstance(self.boss, VoidAngler) else
                     ("iron", self.boss.position, self.boss.hp, self.boss.exposed_rounds,
                      tuple(sorted(self.boss.refluxed_roots)), self.boss.attack_kind,
                      self.boss.target, self.boss.attacks) if isinstance(self.boss, IronGardener) else
                     ("mirror", self.boss.position, self.boss.hp, self.boss.exposed_rounds,
                      tuple(sorted(self.boss.broken_locks)), self.boss.copied_action,
                      self.boss.mirrored_direction, self.boss.target) if isinstance(self.boss, MirrorSeraph) else
                     ("siege", self.boss.position, self.boss.hp, self.boss.exposed_rounds,
                      tuple(sorted(self.boss.broken_locks)), self.boss.rail_axis,
                      self.boss.rail_target, self.boss.charge, self.boss.shots) if isinstance(self.boss, SiegeLeviathan) else
                     ("null", self.boss.position, self.boss.hp, self.boss.exposed_rounds,
                      self.boss.node_index, self.boss.blocked_kind,
                      tuple(sorted(self.boss.erase_targets)), self.boss.erase_countdown,
                      self.boss.cycles, self.boss.warp_target) if isinstance(self.boss, NullWeaver) else
                     ("apex", self.boss.position, self.boss.hp, self.boss.exposed_rounds,
                      self.boss.seals, self.boss.kind, self.boss.target, self.boss.countdown,
                      tuple(sorted(self.boss.danger)), self.boss.gate, self.boss.gate_broken,
                      self.boss.finale_cycles, self.boss.appeal) if self.boss else None),
            "reflectors": tuple(sorted(self.reflectors)),
            "coolant_valves": tuple(sorted(self.coolant_valves)),
            "grounding_pylons": tuple(sorted(self.grounding_pylons)),
            "relay_pads": tuple(sorted(self.relay_pads)),
            "time_anchors": tuple(sorted(self.time_anchors)),
            "gravity_nodes": tuple(sorted(self.gravity_nodes)),
            "root_plates": tuple(sorted(self.root_plates)),
            "mirror_locks": tuple(sorted(self.mirror_locks)),
            "rail_locks": tuple(sorted(self.rail_locks)),
            "rail_covers": tuple(sorted(self.rail_covers.items())),
            "rail_rebuilds": tuple(sorted(self.rail_rebuilds.items())),
            "null_nodes": self.null_nodes,
            "null_void": tuple(sorted(self.null_void)),
            "apex_seals": self.apex_seals,
            "apex_cage": tuple(sorted(self.apex_cage)),
            "vine_seeds": tuple(sorted(self.vine_seeds.items())),
            "vine_walls": tuple(sorted(self.vine_walls)),
            "enemy_intents": tuple((enemy.enemy_type.value, enemy.position, enemy.hp,
                                    enemy.intent.kind.value if enemy.intent else None,
                                    enemy.intent.direction if enemy.intent else None,
                                    enemy.intent.countdown if enemy.intent else 0,
                                    enemy.intent.power if enemy.intent else 0,
                                    enemy.stunned) for enemy in self.enemies),
            "gems": tuple(sorted(self.gems)),
            "fires": tuple(sorted(self.fires)),
            "spikes": tuple(sorted(self.spikes)),
            "pits": tuple(sorted(self.pits)),
            "barrels": tuple(sorted(self.barrels)),
            "medkits": tuple(sorted(self.medkits)),
            "bow_pickups": tuple(sorted(self.bow_pickups)),
            "pistol_pickups": tuple(sorted(self.pistol_pickups)),
            "arrow_bundles": tuple(sorted(self.arrow_bundles)),
            "energy_cells": tuple(sorted(self.energy_cells)),
            "gems_collected": self.gems_collected,
            "kills": self.kills,
            "environment_kills": self.environment_kills,
            "damage_taken": self.damage_taken,
            "done": self.done,
        }
