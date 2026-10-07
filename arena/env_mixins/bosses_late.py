"""Siege Leviathan (80), Null Weaver (90) and Apex Arbiter (100).

Mixin for :class:`arena.env.ArenaEnv`; methods were moved verbatim from ``arena/env.py``.
"""
from __future__ import annotations

from ..boss import ApexArbiter, NullWeaver, SiegeLeviathan, ray_cells
from ..entities import DIRECTIONS


class LateBossMixin:
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

    def siege_blast_cells(self, target: tuple[int, int] | None = None,
                         kind: str | None = None) -> set[tuple[int, int]]:
        boss = self.boss
        if not isinstance(boss, SiegeLeviathan):
            return set()
        if target is None and boss.blast_target is not None and boss.blast_cells:
            return set(boss.blast_cells)
        target = target or boss.blast_target
        if target is None:
            return set()
        kind = kind or boss.blast_kind
        x, y = target
        if kind == "shrapnel":
            ring = {(x + dx, y + dy) for dx in range(-2, 3) for dy in range(-2, 3)
                    if max(abs(dx), abs(dy)) == 2 and self.in_bounds((x + dx, y + dy))}
            reflected = {self.add(cover, direction) for cover in self.rail_covers.values()
                         if cover in ring for direction in DIRECTIONS}
            return ring | {cell for cell in reflected if self.in_bounds(cell) and
                           cell not in self.walls | self.pits}
        radius = 2 if kind == "cross" else 1
        return {(x + offset, y) for offset in range(-radius, radius + 1)
                if self.in_bounds((x + offset, y))} | {
                (x, y + offset) for offset in range(-radius, radius + 1)
                if self.in_bounds((x, y + offset))}

    def siege_charge_path(self) -> tuple[tuple[int, int], ...]:
        boss = self.boss
        if not isinstance(boss, SiegeLeviathan) or boss.charge_target is None:
            return ()
        path = []
        for cell in ray_cells(boss.position, boss.charge_target):
            if cell in self.walls | self.pits:
                break
            path.append(cell)
        return tuple(path)

    def _resolve_siege(self, boss: SiegeLeviathan, events: list[str]) -> float:
        for cell, rounds in list(self.siege_pits.items()):
            if rounds <= 1:
                del self.siege_pits[cell]
                self.pits.discard(cell)
                events.append(f"siege_pit_restore:{cell[0]}:{cell[1]}")
            else:
                self.siege_pits[cell] = rounds - 1
        for origin, rounds in list(self.rail_rebuilds.items()):
            if rounds > 1 or origin in self.rail_covers.values() or origin == self.player.position:
                self.rail_rebuilds[origin] = max(1, rounds - 1)
            else:
                del self.rail_rebuilds[origin]
                self.rail_covers[origin] = origin
                events.append(f"rail_cover_rebuild:{origin[0]}:{origin[1]}")
        reward = 0.0
        if boss.blast_target:
            target, kind = boss.blast_target, boss.blast_kind
            cells = self.siege_blast_cells()
            boss.blast_target = None
            events.append(f"siege_blast:{kind}:{target[0]}:{target[1]}")
            if self.player.position in cells:
                reward += self._damage_entity(self.player, {"cross": 22, "pit": 28, "shrapnel": 18}[kind],
                                              events, "siege_blast")
            for origin, cover in list(self.rail_covers.items()):
                if cover in cells:
                    del self.rail_covers[origin]
                    self.rail_rebuilds[origin] = 7
                    events.append(f"rail_cover_break:{cover[0]}:{cover[1]}")
            if kind == "pit":
                forbidden = (self.walls | self.pits | self.rail_locks | self.medkits |
                             self.energy_cells | self.bow_pickups | self.pistol_pickups |
                             self.arrow_bundles | self.barrels | set(self.rail_covers.values()))
                if target not in forbidden and target != self.player.position:
                    self.pits.add(target)
                    self.siege_pits[target] = 3
                    events.append(f"siege_pit_open:{target[0]}:{target[1]}")
        if boss.exposed_rounds:
            boss.exposed_rounds -= 1
            if boss.exposed_rounds:
                self._step_exposed_boss(boss, (min(12, max(8, self.player.position[0])), 7), events)
            if not boss.exposed_rounds:
                boss.broken_locks.clear()
                events.append("boss_shield_restored")
            return reward
        if boss.charge_target is not None:
            path = self.siege_charge_path()
            boss.charge_target = None
            previous = boss.position
            if self.player.position in path:
                reward += self._damage_entity(self.player, 28, events, "siege_charge")
            for origin, cover in list(self.rail_covers.items()):
                if cover in path:
                    del self.rail_covers[origin]
                    self.rail_rebuilds[origin] = 7
                    events.append(f"rail_cover_break:{cover[0]}:{cover[1]}")
                    lock = (cover[0], 7)
                    if cover != origin and lock in self.rail_locks - boss.broken_locks:
                        boss.broken_locks.add(lock)
                        reward += 15
                        events.append(f"siege_charge_recoil:{lock[0]}")
            if len(boss.broken_locks) == len(self.rail_locks):
                boss.exposed_rounds = 5
                events.append("boss_shield_break")
            landing = next((cell for cell in reversed(path) if cell != self.player.position), previous)
            boss.position = landing
            events.append(f"siege_charge:{previous[0]}:{previous[1]}:{landing[0]}:{landing[1]}")
            if landing != previous:
                events.append(f"boss_move:{previous[0]}:{previous[1]}:{landing[0]}:{landing[1]}")
            return reward
        if boss.rail_target is None:
            if boss.shots and boss.shots % 3 == 0:
                boss.charge_target = self.player.position
                events.append(f"siege_charge_aim:{boss.charge_target[0]}:{boss.charge_target[1]}")
                boss.shots += 1
                return reward
            desired_x = min(12, max(8, self.player.position[0]))
            destination = (boss.position[0] + max(-2, min(2, desired_x - boss.position[0])),
                           boss.position[1] + max(-2, min(2, 7 - boss.position[1])))
            if destination != boss.position and destination != self.player.position:
                previous = boss.position
                boss.position = destination
                events.append(f"boss_move:{previous[0]}:{previous[1]}:{destination[0]}:{destination[1]}")
            boss.rail_axis = "h" if boss.shots % 4 == 3 else "v"
            boss.rail_target = self.player.position[1 if boss.rail_axis == "h" else 0]
            boss.charge = 2
            events.append(f"rail_aim:{boss.rail_axis}:{boss.rail_target}:2")
            return reward
        boss.charge -= 1
        if boss.charge:
            events.append(f"rail_charge:{boss.rail_axis}:{boss.rail_target}:1")
            return reward
        target, axis = boss.rail_target, boss.rail_axis
        cover = self.rail_cover()
        events.append(f"rail_fire:{axis}:{target}")
        if self.rail_threatens(self.player.position):
            reward += self._damage_entity(self.player, boss.rail_damage, events, "railgun")
        if cover:
            origin = next(origin for origin, position in self.rail_covers.items() if position == cover)
            del self.rail_covers[origin]
            self.rail_rebuilds[origin] = 7
            events.append(f"rail_cover_break:{cover[0]}:{cover[1]}")
            lock = (target, 7)
            if axis == "v" and lock in self.rail_locks and lock not in boss.broken_locks:
                boss.broken_locks.add(lock)
                reward += 15
                events.append(f"rail_lock_break:{len(boss.broken_locks)}")
                if len(boss.broken_locks) == len(self.rail_locks):
                    boss.exposed_rounds = 5
                    events.append("boss_shield_break")
        boss.shots += 1
        boss.rail_target = None
        if boss.shots % 2 == 0:
            boss.blasts += 1
            boss.blast_target = self.player.position
            boss.blast_kind = ("shrapnel", "pit", "cross")[(boss.blasts - 1) % 3]
            boss.blast_cells = self.siege_blast_cells(boss.blast_target, boss.blast_kind)
            events.append(f"siege_blast_aim:{boss.blast_kind}:{boss.blast_target[0]}:{boss.blast_target[1]}")
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
        if boss.exposed_rounds:
            boss.exposed_rounds -= 1
            if boss.exposed_rounds and boss.exposed_rounds % 2:
                anchors = ((8, 5), (12, 5), (8, 7), (12, 7), (9, 10), (12, 10))
                choices = sorted((cell for cell in anchors if cell != self.player.position and
                                  cell not in self.walls | self.pits | self.null_void),
                                 key=lambda cell: self._distance(cell, self.player.position), reverse=True)
                destination = self.rng.choice(choices[:3]) if choices else boss.position
                if destination != boss.position:
                    previous = boss.position
                    boss.position = destination
                    events.extend((f"boss_move:{previous[0]}:{previous[1]}:{destination[0]}:{destination[1]}",
                                   "null_warp"))
            elif boss.exposed_rounds:
                self._step_exposed_boss(boss,
                                        (max(7, min(13, self.player.position[0])),
                                         max(6, min(10, self.player.position[1] - 4))), events)
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
        elif boss.cycles % 2 == 0:
            anchors = ((8, 5), (12, 5), (8, 7), (12, 7), (9, 10), (12, 10))
            choices = sorted((cell for cell in anchors if cell != boss.position and
                              cell != self.player.position and cell not in self.null_void),
                             key=lambda cell: self._distance(cell, self.player.position))
            boss.warp_target = self.rng.choice(choices[:3]) if choices else boss.position
            events.append(f"null_warp_aim:{boss.warp_target[0]}:{boss.warp_target[1]}")
        boss.blocked_kind = ("move", "melee", "ranged", "skill")[boss.cycles % 4]
        boss.cycles += 1
        events.append(f"null_block:{boss.blocked_kind}")
        if not boss.erase_targets:
            x, y = self.player.position
            occupied = (self.walls | self.pits | set(self.null_nodes) | self.medkits |
                        self.energy_cells | self.bow_pickups | self.pistol_pickups |
                        self.arrow_bundles | self.barrels)
            boss.fracture_axis = self.rng.choice(("h", "v"))
            offsets = ({(dx, dy) for dx in range(-3, 4) for dy in (-1, 0, 1)
                        if dy == 0 or abs(dx) <= 1} if boss.fracture_axis == "h" else
                       {(dx, dy) for dy in range(-3, 4) for dx in (-1, 0, 1)
                        if dx == 0 or abs(dy) <= 1})
            boss.erase_targets = {(x + dx, y + dy) for dx, dy in offsets
                                  if self.in_bounds((x + dx, y + dy)) and
                                  (x + dx, y + dy) not in occupied}
            escapes = [self.add(self.player.position, direction) for direction in DIRECTIONS
                       if self.in_bounds(self.add(self.player.position, direction)) and
                       self.add(self.player.position, direction) not in self.walls | self.pits and
                       not self.boss_at(self.add(self.player.position, direction))]
            if escapes:
                boss.erase_targets.discard(self.rng.choice(escapes))
            candidates = sorted(boss.erase_targets - self.null_void - {self.player.position})
            self.rng.shuffle(candidates)
            boss.fracture_cells.clear()
            for cell in candidates:
                if len(boss.fracture_cells) >= 2 or len(self.null_void) >= 4:
                    break
                self.null_void.add(cell)
                if set(self.null_nodes) <= self._reachable_cells():
                    boss.fracture_cells.add(cell)
                else:
                    self.null_void.remove(cell)
            self.null_void.difference_update(boss.fracture_cells)
            boss.erase_countdown = 2
            events.append(f"null_mark:{y}:2")
            return 0.0
        boss.erase_countdown -= 1
        if boss.erase_countdown:
            events.append(f"null_countdown:{boss.erase_countdown}")
            return 0.0
        self.null_void.update(boss.fracture_cells)
        boss.fracture_cells.clear()
        hit_cells = set(boss.erase_targets)
        boss.erase_targets.clear()
        reward = 0.0
        if self.player.position in hit_cells:
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

    def _apex_barrage_cells(self, phase: int, forced_safe: tuple[int, int] | None = None,
                            forced_danger: tuple[int, int] | None = None) -> set[tuple[int, int]]:
        cells = {(x, y) for y in range(5, 18) for x in range(2, 18)
                 if (x + 2 * y + phase) % 3 != 0 and (x, y) not in self.walls | self.pits}
        if forced_safe is not None:
            cells.discard(forced_safe)
        if forced_danger is not None and forced_danger not in self.walls | self.pits:
            cells.add(forced_danger)
        return cells

    def _resolve_apex(self, boss: ApexArbiter, events: list[str]) -> float:
        if boss.exposed_rounds:
            boss.exposed_rounds -= 1
            if boss.exposed_rounds:
                self._apex_reposition(boss, events)
            if not boss.exposed_rounds:
                events.append("boss_shield_restored")
            return 0.0
        if not boss.countdown:
            self._apex_reposition(boss, events)
            boss.kind = (("cage", "barrage", "charge", "gravity")[boss.seals]
                         if boss.seals < 4 else ("cage_barrage", "charge_gravity", "verdict")[boss.finale_cycles % 3])
            boss.target = self.player.position
            boss.gate_broken = False
            boss.barrage_cells.clear()
            boss.barrage_volley = 0
            boss.appeal = None
            boss.appeal_ready = False
            if boss.kind in ("cage", "cage_barrage"):
                directions = ("n", "e", "w", "s")
                boss.gate = next((self.add(boss.target, d) for d in directions
                                  if self.in_bounds(self.add(boss.target, d)) and
                                  self.add(boss.target, d) not in self.walls | self.pits | self.barrels and
                                  not self.boss_at(self.add(boss.target, d))), None)
                boss.danger = {boss.target}
                if boss.kind == "cage_barrage" and boss.gate:
                    boss.barrage_phase = self.rng.randrange(3)
                    boss.barrage_volley = 1
                    boss.barrage_cells = self._apex_barrage_cells(boss.barrage_phase, boss.gate)
                    boss.danger |= boss.barrage_cells
                boss.countdown = 2
            elif boss.kind == "barrage":
                boss.barrage_phase = self.rng.randrange(3)
                boss.barrage_volley = 1
                boss.barrage_cells = self._apex_barrage_cells(
                    boss.barrage_phase, self.apex_seals[1] if boss.seals == 1 else None)
                boss.danger = set(boss.barrage_cells)
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
        barrage_special_safe = (self.apex_seals[1] if boss.kind == "barrage" and
                                boss.barrage_volley == 1 and boss.seals == 1 else
                                boss.gate if boss.kind == "cage_barrage" else None)
        if boss.barrage_volley:
            boss.fired_barrage_cells = set(boss.barrage_cells)
            boss.fired_barrage_volley = boss.barrage_volley
        else:
            boss.fired_barrage_cells.clear()
            boss.fired_barrage_volley = 0
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
            cage_hit = not boss.gate_broken and self.player.position == boss.target
            barrage_hit = self.player.position in boss.barrage_cells
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
            if boss.appeal_ready or boss.appeal and self.player.position == boss.appeal:
                events.append("apex_appeal")
            elif self.player.position in boss.danger:
                reward += self._damage_entity(self.player, 36, events, "apex_verdict")
        charge_path = (self._apex_charge_path(boss, boss.target)
                       if boss.kind in ("charge", "charge_gravity") else ())
        endpoint = charge_path[-1] if charge_path else (boss.position if boss.kind in ("charge", "charge_gravity") else boss.target)
        events.append(f"apex_fire:{boss.kind}:{boss.target[0]}:{boss.target[1]}:0:{endpoint[0]}:{endpoint[1]}")
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
        boss.appeal_ready = False
        if boss.seals == 4:
            if "apex_seal:4" in events or "apex_appeal" in events:
                if "apex_appeal" in events:
                    boss.finale_cycles += 1
                boss.exposed_rounds = 6
                events.append("boss_shield_break")
            elif boss.barrage_volley != 2:
                boss.finale_cycles += 1
        if boss.barrage_volley == 1:
            boss.barrage_phase = (boss.barrage_phase + 1) % 3
            boss.barrage_cells = self._apex_barrage_cells(boss.barrage_phase,
                                                           forced_danger=barrage_special_safe)
            if self.player.position in boss.barrage_cells:
                neighbors = [self.add(self.player.position, direction) for direction in DIRECTIONS
                             if self.in_bounds(self.add(self.player.position, direction)) and
                             self.add(self.player.position, direction) not in self.walls | self.pits |
                             self.barrels and not self.boss_at(self.add(self.player.position, direction))]
                if not any(cell not in boss.barrage_cells for cell in neighbors) and neighbors:
                    escape = next((cell for cell in neighbors if cell in boss.fired_barrage_cells),
                                  neighbors[0])
                    boss.barrage_cells.discard(escape)
            boss.danger = set(boss.barrage_cells)
            boss.kind = "barrage"
            boss.target = self.player.position
            boss.countdown = 1
            boss.barrage_volley = 2
            events.append("apex_barrage_second_aim")
        else:
            boss.barrage_volley = 0
        return reward
