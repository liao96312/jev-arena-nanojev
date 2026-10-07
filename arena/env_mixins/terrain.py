"""Map generation, reachability checks and terrain hazards (pushes, barrels).

Mixin for :class:`arena.env.ArenaEnv`; methods were moved verbatim from ``arena/env.py``.
"""
from __future__ import annotations

from ..boss import ApexArbiter, ChronoMantis, FurnaceHydra, IronGardener, MirrorSeraph, NullWeaver, PrismWarden, SiegeLeviathan, StormChoir, VoidAngler
from ..entities import Action, DIRECTIONS, Enemy, EnemyType, Player


class TerrainMixin:
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
        self.siege_pits: dict[tuple[int, int], int] = {}
        self.null_nodes: tuple[tuple[int, int], ...] = ()
        self.null_void: set[tuple[int, int]] = set()
        self.apex_seals: tuple[tuple[int, int], ...] = ()
        self.apex_cage: set[tuple[int, int]] = set()
        self.apex_fast_volley_resolved = False
        self.vine_seeds: dict[tuple[int, int], int] = {}
        self.vine_walls: set[tuple[int, int]] = set()
        self.iron_spores: dict[tuple[int, int], int] = {}
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
                shift = self.rng.choice((-1, 0, 1))
                self.reflectors = {(9 + shift, 12), (11 + shift, 12),
                                   (8 + shift, 13), (12 + shift, 13)}
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
        hounds = min(self.config.enemies, 1 + int(self.config.difficulty_level >= 15)) if (
            self.config.finish_on_all_gems and self.config.difficulty_level >= 5) else 0
        for index in range(self.config.enemies):
            roll = self.rng.random()
            enemy_type = (EnemyType.RAZOR_HOUND if index < hounds else
                          EnemyType.BOMBER if roll < self.config.bomber_ratio else
                          EnemyType.CHARGER if roll < self.config.bomber_ratio + self.config.charger_ratio else
                          EnemyType.ARCHER if roll < sum((self.config.bomber_ratio,
                                                         self.config.charger_ratio,
                                                         self.config.archer_ratio)) else
                          EnemyType.CHASER)
            enemy = Enemy(next(take), enemy_type=enemy_type)
            bonus = min(10, self.config.enemy_hp_bonus) if enemy_type == EnemyType.RAZOR_HOUND else self.config.enemy_hp_bonus
            enemy.hp += bonus
            enemy.max_hp += bonus
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
