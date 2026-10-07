"""Pickups, damage resolution and player weapons (bow / pistol targeting).

Mixin for :class:`arena.env.ArenaEnv`; methods were moved verbatim from ``arena/env.py``.
"""
from __future__ import annotations

from ..boss import ApexArbiter, ChronoMantis, FurnaceHydra, IronGardener, MirrorSeraph, NullWeaver, PrismWarden, SiegeLeviathan, StormChoir, VoidAngler
from ..entities import Enemy, Player


class CombatMixin:
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
            reward += self._activate_null_node(self.player.position, events)
        return reward

    def _activate_null_node(self, position: tuple[int, int], events: list[str]) -> float:
        boss = self.boss
        if position != self.null_nodes[boss.node_index]:
            boss.node_index = 0
            events.append("null_node_reset")
            return 0.0
        boss.node_index += 1
        events.append(f"null_node:{boss.node_index}")
        if boss.node_index == len(self.null_nodes):
            boss.exposed_rounds = 5
            boss.blocked_kind = None
            boss.erase_targets.clear()
            boss.fracture_cells.clear()
            boss.erase_countdown = 0
            boss.warp_target = None
            events.extend(("boss_shield_break", "null_reverse_write"))
        return 15.0

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
            if (isinstance(entity, ApexArbiter) and entity.kind == "verdict" and
                    entity.countdown and source in ("attack", "bow", "pistol")):
                entity.appeal_ready = True
                events.append("apex_appeal_mark")
                return 0.0
            if (isinstance(entity, MirrorSeraph) and entity.exposed_rounds >= 3 and
                    entity.evade_ready and not entity.emp_jammed and source in ("bow", "pistol")):
                x, y = entity.position
                options = ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1))
                safe = [cell for cell in options if self.in_bounds(cell) and
                        cell not in self.walls | self.pits | self.fires | self.spikes | self.mirror_locks and
                        cell != self.player.position and not self.enemy_at(cell)]
                if safe:
                    destination = max(safe, key=lambda cell: (self._distance(cell, self.player.position),
                                                              cell[0] != x, -cell[0], -cell[1]))
                    entity.hp += actual
                    entity.position = destination
                    entity.evade_ready = False
                    entity.evaded = True
                    events.extend((f"boss_move:{x}:{y}:{destination[0]}:{destination[1]}", "mirror_evade"))
                    return 0.0
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
                self.enemies = [enemy for enemy in self.enemies if not enemy.summoned_by]
                self.boss = None
                self.kills += 1
                self.score += 50
                events.append("boss_defeated")
                return 50.0
            return actual * 0.5
        events.append(f"enemy_damage:{source}:{actual}")
        if entity.summoned_by and entity.summoned_by.startswith("mirror_") and actual:
            linked = max(1, round(actual * .3))
            events.append(f"mirror_link:{linked}")
            self._damage_entity(self.player, linked, events, "mirror_link")
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
