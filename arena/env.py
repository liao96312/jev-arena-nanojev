from __future__ import annotations

import copy
import random
from collections import deque
from dataclasses import dataclass

from .entities import (DIRECTIONS, ENEMY_MELEE_BONUS, ENEMY_MOVE_DELAY, ENEMY_SPEED_LEVEL, Action,
                       Enemy, EnemyType, Intent, IntentType, Player, PlayerLoadout)
from .boss import ChronoMantis, FurnaceHydra, PrismWarden, StormChoir, ray_cells


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
    if level in (10, 20, 30, 40):
        return ArenaConfig(difficulty_level=level, max_ticks=300, walls=0, enemies=0, gems=0,
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
        medkits=max(1, 3 - level // 3),
        bow_pickups=1,
        pistol_pickups=1 if level >= 3 else 0,
        arrow_bundles=1 if level >= 2 else 0,
        energy_cells=1 if level >= 4 else 0,
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
        self.rng = random.Random(seed)
        self.tick = self.score = self.gems_collected = self.kills = self.environment_kills = 0
        self.round, self.ap_remaining = 1, self.config.action_points
        self.damage_taken = 0
        self.previous_player_position: tuple[int, int] | None = None
        self.last_action: str | None = None
        self.done = False
        for _ in range(20):
            self._generate_map()
            self._plan_enemy_intents()
            if self._map_is_playable():
                return self.observation()
        raise RuntimeError("could not generate a playable map after 20 attempts")

    def _generate_map(self) -> None:
        self.boss: PrismWarden | FurnaceHydra | StormChoir | ChronoMantis | None = None
        self.reflectors: set[tuple[int, int]] = set()
        self.breakable_walls: set[tuple[int, int]] = set()
        self.coolant_valves: set[tuple[int, int]] = set()
        self.grounding_pylons: set[tuple[int, int]] = set()
        self.relay_pads: set[tuple[int, int]] = set()
        self.time_anchors: set[tuple[int, int]] = set()
        self.forge_floor: set[tuple[int, int]] = set()
        self.furnace_burns: dict[tuple[int, int], int] = {}
        if self.config.difficulty_level in (10, 20, 30, 40) and self.config.finish_on_all_gems:
            furnace = self.config.difficulty_level == 20
            storm = self.config.difficulty_level == 30
            chrono = self.config.difficulty_level == 40
            self.player = Player((10, 16 if furnace or storm or chrono else 15), loadout=self.loadout)
            if furnace:
                self.player.loadout.bow = True
                self.player.loadout.arrows = max(10, self.player.loadout.arrows)
                room = {(x, y) for y in range(20)
                        for x in range(4 if y in (0, 1, 18, 19) else
                                       2 if y in (2, 3, 16, 17) else 0,
                                       16 if y in (0, 1, 18, 19) else
                                       18 if y in (2, 3, 16, 17) else 20)}
            elif storm or chrono:
                self.player.loadout.pistol = True
                self.player.loadout.energy = max(12, self.player.loadout.energy)
                room = {(x, y) for y in range(20)
                        for x in range(3 if y in (0, 1, 18, 19) else
                                       1 if y in (2, 3, 16, 17) else 0,
                                       17 if y in (0, 1, 18, 19) else
                                       19 if y in (2, 3, 16, 17) else 20)}
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
                self.medkits, self.energy_cells = {(5, 15), (14, 15)}, {(7, 15), (13, 15)}
                self.bow_pickups, self.pistol_pickups = set(), {(10, 15)}
                self.arrow_bundles = set()
                self.grounding_pylons = {(8, 9), (12, 9), (11, 13), (7, 13)}
                self.relay_pads = {(8, 8), (10, 8), (12, 8)}
                self.boss = StormChoir()
            else:
                self.walls |= {(4, 8), (15, 8), (5, 10), (12, 10),
                               (4, 13), (15, 13), (6, 15), (13, 15)}
                self.pits = {(3, 11), (16, 11), (5, 16), (14, 16)}
                self.medkits, self.energy_cells = {(6, 14), (13, 14)}, {(7, 15), (12, 15)}
                self.bow_pickups, self.pistol_pickups = set(), {(10, 15)}
                self.arrow_bundles = set()
                self.time_anchors = {(9, 11), (14, 11)}
                self.boss = ChronoMantis()
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
        blocked = self.walls | self.fires | self.spikes | self.pits | self.barrels
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

    def boss_at(self, position: tuple[int, int]) -> PrismWarden | FurnaceHydra | StormChoir | ChronoMantis | None:
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
                    target not in self.barrels and
                    not self.enemy_at(target) and not self.boss_at(target)):
                actions.append(Action(f"move_{direction}"))
            if self.enemy_at(target) or self.boss_at(target) or target in self.barrels:
                actions.append(Action(f"attack_{direction}"))
            if self.enemy_at(target):
                if self.in_bounds(self.add(target, direction)):
                    actions.append(Action(f"shove_{direction}"))
            destination = self.add(target, direction)
            if (not self.player.cooldowns.get("dash", 0) and self.in_bounds(target) and
                    self.in_bounds(destination) and target not in self.walls and destination not in self.walls and
                    target not in self.pits and destination not in self.pits and
                    target not in self.barrels and destination not in self.barrels and
                    not self.enemy_at(target) and not self.enemy_at(destination) and
                    not self.boss_at(target) and not self.boss_at(destination)):
                actions.append(Action(f"dash_{direction}"))
            bow_distance = self._ranged_target_distance(direction, self.config.bow_range)
            pistol_distance = self._ranged_target_distance(direction, self.config.pistol_range)
            if self.player.loadout.bow and self.player.loadout.arrows and bow_distance and bow_distance > 1:
                ranged.append((bow_distance, Action(f"shoot_bow_{direction}")))
            if self.player.loadout.pistol and self.player.loadout.energy and pistol_distance and pistol_distance > 1:
                ranged.append((pistol_distance, Action(f"shoot_pistol_{direction}")))
        if self.player.medkits and self.player.hp < 100:
            actions.append(Action.HEAL)
        if (not self.player.cooldowns.get("emp", 0) and
                any(self._distance(self.player.position, enemy.position) <= self.config.emp_radius
                    for enemy in self.enemies)):
            actions.append(Action.EMP)
        ranged.sort(key=lambda item: item[0])
        actions.extend(action for _, action in ranged[:max(0, 11 - len(actions))])
        actions.append(Action.WAIT)
        return [action for action in actions if self._action_cost(action) <= self.ap_remaining]

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
            if target in self.barrels:
                reward += self._explode_barrel(target, events)
            else:
                enemy = self.enemy_at(target) or self.boss_at(target)
                assert enemy is not None
                reward += self._damage_entity(enemy, self.config.attack_damage, events, "attack")
        elif action.value.startswith("shove_"):
            direction = action.value[-1]
            enemy = self.enemy_at(self.add(self.player.position, direction))
            assert enemy is not None
            reward += self._push_entity(enemy, direction, events)
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
        return reward

    def _damage_entity(self, entity: Player | Enemy | PrismWarden | FurnaceHydra | StormChoir | ChronoMantis,
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
            if entity.hp <= 0:
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
                    range_: int) -> tuple[Enemy | PrismWarden | FurnaceHydra | StormChoir | ChronoMantis, int] | None:
        target = origin
        for distance in range(1, range_ + 1):
            target = self.add(target, direction)
            if not self.in_bounds(target) or target in self.walls:
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
        return min(distances) if distances else None

    def _player_shoot(self, weapon: str, direction: str, events: list[str]) -> float:
        loadout = self.player.loadout
        if weapon == "bow":
            loadout.arrows -= 1
            damage, range_ = self.config.bow_damage, self.config.bow_range
        else:
            loadout.energy -= 1
            damage, range_ = self.config.pistol_damage, self.config.pistol_range
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
                reward += self._damage_entity(self.player, boss.surge_damage, events, "storm_surge")
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
            if not boss.exposed_rounds:
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
        if self.player.position == (boss.position[0], 11):
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
                threats.append(("storm_choir/surge", self.boss.surge_damage))
        if isinstance(self.boss, ChronoMantis):
            if self.boss.phase == "slash" and self._distance(position, self.boss.slash_target) <= 1:
                threats.append(("chrono_mantis/slash", self.boss.slash_damage))
            if self.boss.phase == "leap" and position == self.boss.slash_target:
                threats.append(("chrono_mantis/echo", self.boss.echo_damage))
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
                      self.boss.leap_countdown, self.boss.moves) if self.boss else None),
            "reflectors": tuple(sorted(self.reflectors)),
            "coolant_valves": tuple(sorted(self.coolant_valves)),
            "grounding_pylons": tuple(sorted(self.grounding_pylons)),
            "relay_pads": tuple(sorted(self.relay_pads)),
            "time_anchors": tuple(sorted(self.time_anchors)),
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
