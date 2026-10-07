"""Boss dispatch plus Prism Warden (10), Furnace Hydra (20), Storm Choir (30) and Chrono Mantis (40).

Mixin for :class:`arena.env.ArenaEnv`; methods were moved verbatim from ``arena/env.py``.
"""
from __future__ import annotations

import math
from ..boss import ApexArbiter, ChronoMantis, FurnaceHydra, IronGardener, MirrorSeraph, NullWeaver, PrismWarden, SiegeLeviathan, StormChoir, VoidAngler, ray_cells
from ..entities import DIRECTIONS, Enemy, EnemyType


class EarlyBossMixin:
    def boss_ray(self) -> tuple[tuple[int, int], ...]:
        if not isinstance(self.boss, PrismWarden) or self.boss.target is None:
            return ()
        target = self.boss.target
        if self.boss.hp <= self.boss.max_hp // 3:
            target = self._prism_spin_target(self.boss.spin_step)
        cells = tuple(cell for cell in ray_cells(self.boss.position, target) if self.in_bounds(cell))
        active = self.reflectors - self.boss.used_reflectors if not self.boss.reflector_lockout else set()
        stop = next((index for index, cell in enumerate(cells)
                     if cell in self.walls or cell in active), None)
        return cells[:stop + 1] if stop is not None else cells

    def _prism_spin_target(self, step: int) -> tuple[int, int]:
        angle = step * math.tau / 24
        return (self.boss.position[0] + round(16 * math.cos(angle)),
                self.boss.position[1] + round(16 * math.sin(angle)))

    def prism_rays(self) -> tuple[tuple[tuple[int, int], ...], ...]:
        boss = self.boss
        if not isinstance(boss, PrismWarden) or boss.target is None:
            return ()
        primary = self.boss_ray()
        if boss.hp <= boss.max_hp // 3:
            targets = [self._prism_spin_target((boss.spin_step + offset) % 24)
                       for offset in (8, 16)]
        else:
            targets = [(boss.target[0] - 3, boss.target[1]),
                       (boss.target[0] + 3, boss.target[1])]
        rays = [primary]
        active = self.reflectors - boss.used_reflectors if not boss.reflector_lockout else set()
        for target in targets:
            cells = tuple(cell for cell in ray_cells(boss.position, target) if self.in_bounds(cell))
            stop = next((index for index, cell in enumerate(cells)
                         if cell in self.walls or cell in active), None)
            rays.append(cells[:stop + 1] if stop is not None else cells)
        return tuple(rays)

    def prism_attack_cells(self) -> set[tuple[int, int]]:
        rays = self.prism_rays()
        if not rays:
            return set()
        path = rays[0]
        cells = set().union(*rays)
        if self.boss.sweep and path[-1] not in self.reflectors | self.walls:
            x, y = path[-1]
            cells.update((x + dx, y + dy) for dx, dy in DIRECTIONS.values()
                         if self.in_bounds((x + dx, y + dy)))
        return cells

    def prism_baits(self) -> set[tuple[int, int]]:
        if not isinstance(self.boss, PrismWarden) or self.boss.position[1] != 8:
            return set()
        unused = (self.reflectors - self.boss.used_reflectors
                  if not self.boss.reflector_lockout else set())
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

    def _summon_boss_minion(self, boss: FurnaceHydra | IronGardener, events: list[str]) -> None:
        interval = 4 if isinstance(boss, FurnaceHydra) else 6
        limit = 3 if isinstance(boss, FurnaceHydra) else 1
        if (self.round % interval or boss.exposed_rounds or boss.summons >= limit or
                sum(enemy.summoned_by == ("furnace" if isinstance(boss, FurnaceHydra) else "iron")
                    for enemy in self.enemies) >= (2 if isinstance(boss, FurnaceHydra) else 1)):
            return
        furnace = isinstance(boss, FurnaceHydra)
        positions = ((6, 7), (14, 7)) if furnace else ((6, 8), (14, 8))
        for position in positions[boss.summons % 2:] + positions[:boss.summons % 2]:
            if (position in self.walls | self.pits | self.fires | self.vine_walls or
                    position == self.player.position or self.enemy_at(position) or
                    self._distance(position, self.player.position) <= 3):
                continue
            kind = "furnace" if furnace else "iron"
            enemy = Enemy(position, enemy_type=EnemyType.BOMBER if furnace else EnemyType.CHASER,
                          summoned_by=kind)
            self.enemies.append(enemy)
            self._plan_enemy_intents([enemy])
            boss.summons += 1
            events.append(f"boss_summon:{kind}:{position[0]}:{position[1]}")
            return

    def _step_exposed_boss(self, boss, goal: tuple[int, int], events: list[str]) -> None:
        origin = boss.position
        for _ in range(2):
            x, y = boss.position
            dx, dy = (goal[0] > x) - (goal[0] < x), (goal[1] > y) - (goal[1] < y)
            options = ((x + dx, y + dy), (x + dx, y), (x, y + dy))
            destination = next((cell for cell in options if cell != boss.position and
                                self.in_bounds(cell) and
                                cell not in self.walls | self.pits | self.fires | self.spikes | self.barrels and
                                cell != self.player.position and not self.enemy_at(cell)), None)
            if destination is None:
                break
            boss.position = destination
        if boss.position != origin:
            events.append(f"boss_move:{origin[0]}:{origin[1]}:{boss.position[0]}:{boss.position[1]}")

    def _resolve_boss(self, events: list[str]) -> float:
        boss = self.boss
        if boss is None:
            return 0.0
        if self.round < self.config.spawn_protection_rounds:
            if isinstance(boss, ChronoMantis):
                self._step_exposed_boss(boss,
                                        (max(6, min(14, self.player.position[0])),
                                         max(7, min(11, self.player.position[1] - 4))), events)
            return 0.0
        if isinstance(boss, PrismWarden):
            if boss.reflector_lockout and (boss.reflector_lockout > 1 or boss.target is None):
                boss.reflector_lockout -= 1
            for mirror, rounds in list(boss.reflector_cooldowns.items()):
                if rounds <= 1 and boss.target is None:
                    del boss.reflector_cooldowns[mirror]
                    boss.used_reflectors.discard(mirror)
                    events.append(f"boss_reflector_recharged:{mirror[0]}:{mirror[1]}")
                elif rounds > 1:
                    boss.reflector_cooldowns[mirror] = rounds - 1
        if isinstance(boss, FurnaceHydra):
            reward = self._resolve_furnace(boss, events)
            self._summon_boss_minion(boss, events)
            return reward
        if isinstance(boss, StormChoir):
            return self._resolve_storm(boss, events)
        if isinstance(boss, ChronoMantis):
            return self._resolve_chrono(boss, events)
        if isinstance(boss, VoidAngler):
            return self._resolve_void(boss, events)
        if isinstance(boss, IronGardener):
            reward = self._resolve_iron(boss, events)
            self._summon_boss_minion(boss, events)
            return reward
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
            if boss.exposed_rounds:
                self._step_exposed_boss(boss,
                                        (12 if self.player.position[0] <= boss.position[0] else 8, 8),
                                        events)
            if not boss.exposed_rounds:
                boss.reflections = 0
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
        rays = self.prism_rays()
        active = self.reflectors - boss.used_reflectors if not boss.reflector_lockout else set()
        path = (next((ray for ray in rays if ray and ray[-1] in active),
                     rays[0]) if boss.hp <= boss.max_hp // 3 else rays[0])
        endpoint = path[-1] if path else boss.position
        reflected = endpoint in active
        for ray in rays:
            if ray:
                end = ray[-1]
                events.append(f"boss_prism_shot:{boss.position[0]}:{boss.position[1]}:"
                              f"{end[0]}:{end[1]}:{int(reflected and ray is path)}")
        if boss.sweep and not reflected:
            events.append("boss_prism_sweep")
        reward = 0.0
        side_hit = any(self.player.position in ray for ray in rays[1:])
        beam_hit = False
        if reflected:
            boss.used_reflectors.add(endpoint)
            boss.reflector_cooldowns[endpoint] = 6
            boss.reflector_lockout = 3
            boss.reflections += 1
            events.append(f"boss_reflect:{boss.reflections}")
            if (boss.reflections < 3 and self._distance(endpoint, self.player.position) <= 5 and
                    not any(cell in self.walls for cell in ray_cells(endpoint, self.player.position)[:-1]) and
                    self.rng.random() < (.75 if boss.hp <= boss.max_hp // 3 else
                                         .50 if boss.hp <= boss.max_hp * 2 // 3 else .25)):
                pulses = 2 if boss.hp <= boss.max_hp // 3 else 1
                for _ in range(pulses):
                    events.append(f"boss_prism_followup:{endpoint[0]}:{endpoint[1]}:"
                                  f"{self.player.position[0]}:{self.player.position[1]}")
                    reward += self._damage_entity(self.player, 7, events, "boss_prism_followup")
            if boss.reflections == 3:
                boss.exposed_rounds = 5
                events.append("boss_shield_break")
                reward += 15
        elif endpoint in self.breakable_walls:
            self.breakable_walls.remove(endpoint)
            self.walls.remove(endpoint)
            events.append(f"boss_cover_break:{endpoint[0]}:{endpoint[1]}")
        elif self.player.position in self.prism_attack_cells():
            reward += self._damage_entity(self.player, boss.beam_damage, events, "boss_prism")
            beam_hit = True
        if side_hit and not beam_hit:
            reward += self._damage_entity(self.player, boss.beam_damage, events, "boss_prism")
        boss.shots_fired += 1
        if boss.hp <= boss.max_hp // 3:
            boss.spin_step = (boss.spin_step + 2) % 24
        if boss.exposed_rounds == 0:
            if boss.reflections >= 2 and not boss.lunge_used and self.rng.random() < .45:
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
            y = previous[1]
            boss.position = (x, y)
            boss.target = None
            if previous != boss.position:
                events.append(f"boss_move:{previous[0]}:{previous[1]}:{x}:{y}")
            if boss.hp <= boss.max_hp // 3 and self.rng.random() < .55:
                boss.target = self.player.position
                events.append("boss_aim")
        else:
            boss.target = None
        return reward

    def _resolve_furnace(self, boss: FurnaceHydra, events: list[str]) -> float:
        heat_reward = 0.0
        for valve_x, rounds in list(boss.valve_heat.items()):
            if rounds > 1:
                boss.valve_heat[valve_x] = rounds - 1
            elif self.player.position == (valve_x, 12):
                heat_reward += self._damage_entity(self.player, 10, events, "furnace_valve_heat")
        for cell, rounds in list(self.furnace_burns.items()):
            if rounds == 1:
                del self.furnace_burns[cell]
                self.fires.remove(cell)
                events.append(f"furnace_burn_end:{cell[0]}:{cell[1]}")
            else:
                self.furnace_burns[cell] = rounds - 1
        if boss.exposed_rounds:
            boss.exposed_rounds -= 1
            if boss.exposed_rounds == 2:
                previous = boss.position
                goal_x = 11 if self.player.position[0] >= boss.position[0] else 9
                boss.position = (goal_x, 7)
                if boss.position != previous:
                    events.append(f"boss_move:{previous[0]}:{previous[1]}:{boss.position[0]}:{boss.position[1]}")
            if not boss.exposed_rounds:
                boss.valves_opened.clear()
                boss.valve_heat.clear()
                events.append("boss_shield_restored")
            return heat_reward
        if boss.target is None:
            boss.attack_kind = ("triple" if boss.attacks >= 2 and boss.attacks % 4 == 2 else
                                "wave" if boss.attacks == 0 or boss.non_wave_streak >= 2 else
                                self.rng.choice(("wave", "wave", "fireball")))
            if boss.attack_kind in ("wave", "triple"):
                remaining = [x for x in (10, 7, 13) if x not in boss.valves_opened]
                boss.head_x = (min(remaining, key=lambda x: abs(x - self.player.position[0]))
                               if boss.attack_kind == "triple" and remaining else
                               self.rng.choice(remaining or [10, 7, 13]))
                boss.wave_columns = ((7, 10, 13) if boss.attack_kind == "triple" else
                                     (boss.head_x - 1, boss.head_x, boss.head_x + 1)
                                     if boss.hp <= boss.max_hp // 2 and self.rng.random() < .5 else
                                     (boss.head_x,))
                boss.wave_rapid = False
                boss.combo_queued = (boss.attack_kind == "wave" and boss.hp <= boss.max_hp // 2
                                     and self.rng.random() < .5)
                boss.target = (boss.head_x, 16)
            else:
                boss.target = self.player.position
            self._step_exposed_boss(boss,
                                    (max(8, min(12, boss.target[0])), 7), events)
            events.append(f"furnace_aim:{boss.attack_kind}:{boss.target[0]}:{boss.target[1]}")
            return heat_reward
        reward = heat_reward
        if boss.attack_kind in ("wave", "triple"):
            columns = boss.wave_columns or ((boss.head_x - 1, boss.head_x, boss.head_x + 1)
                                            if boss.hp <= boss.max_hp // 2 else (boss.head_x,))
            events.append(f"furnace_wave:{boss.head_x}:{','.join(map(str, columns))}:{int(boss.wave_rapid)}")
            if (self.player.position in self.coolant_valves and self.player.position[0] in columns and
                    self.player.position[0] not in boss.valves_opened and
                    (boss.attack_kind == "triple" or self.player.position[0] == boss.head_x) and
                    not boss.wave_rapid):
                valve_x = self.player.position[0]
                boss.valves_opened.add(valve_x)
                boss.valve_heat[valve_x] = 2
                events.append(f"furnace_valve:{valve_x}")
                reward += 15
            elif 8 <= self.player.position[1] <= 16 and self.player.position[0] in columns:
                damage = boss.wave_damage if boss.attack_kind == "triple" or self.player.position[0] == boss.head_x else 12
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
        boss.non_wave_streak = 0 if boss.attack_kind in ("wave", "triple") else boss.non_wave_streak + 1
        boss.target = None
        if len(boss.valves_opened) == 3:
            boss.exposed_rounds = 3
            boss.position = (10, 7)
            boss.combo_queued = False
            if self.player.loadout.bow and self.player.loadout.arrows < 3:
                self.player.loadout.arrows += 3
                events.append("furnace_restock_arrows")
            events.append("boss_shield_break")
        elif boss.attack_kind == "wave" and boss.combo_queued:
            boss.combo_queued = False
            boss.wave_rapid = True
            remaining = [x for x in (10, 7, 13) if x not in boss.valves_opened]
            boss.head_x = self.rng.choice(remaining or [10, 7, 13])
            boss.wave_columns = (boss.head_x - 1, boss.head_x, boss.head_x + 1)
            boss.target = (boss.head_x, 16)
            events.append(f"furnace_aim:wave:{boss.head_x}:16")
            events.append("furnace_rapid_combo")
        else:
            previous = boss.position
            boss.position = (max(8, min(12, boss.position[0] +
                                  (self.player.position[0] > boss.position[0]) -
                                  (self.player.position[0] < boss.position[0]))),
                             max(6, min(9, boss.position[1] +
                                 (self.player.position[1] > boss.position[1]) -
                                 (self.player.position[1] < boss.position[1]))))
            if previous != boss.position:
                events.append(f"boss_move:{previous[0]}:{previous[1]}:{boss.position[0]}:{boss.position[1]}")
        return reward

    def _resolve_storm(self, boss: StormChoir, events: list[str]) -> float:
        reward = 0.0
        if boss.net_target:
            target = boss.net_target
            events.append(f"storm_net_fire:{target[0]}:{target[1]}")
            if self._distance(self.player.position, target) <= 1:
                reward += self._damage_entity(self.player, boss.net_damage, events, "storm_net")
            boss.net_target = None
        if boss.exposed_rounds:
            boss.exposed_rounds -= 1
            if boss.exposed_rounds:
                self._step_exposed_boss(boss, (min(12, max(8, self.player.position[0])), 6), events)
            if not boss.exposed_rounds:
                if boss.hp <= boss.max_hp * 2 // 3 and self.relay_pads != {(7, 8), (13, 8)}:
                    self.relay_pads = {(7, 8), (13, 8)}
                    events.append("storm_relay_shift")
                events.append("boss_shield_restored")
            return reward
        if boss.overdrive_rounds:
            previous = boss.position
            side = self.rng.choice((-2, -1, 1, 2))
            flank = (max(3, min(16, self.player.position[0] + side)), self.player.position[1])
            self._step_exposed_boss(boss, flank, [])
            self._step_exposed_boss(boss, self.player.position, [])
            if boss.position != previous:
                events.append(f"boss_move:{previous[0]}:{previous[1]}:"
                              f"{boss.position[0]}:{boss.position[1]}")
                events.append(f"storm_overdrive_move:{previous[0]}:{previous[1]}:"
                              f"{boss.position[0]}:{boss.position[1]}")
            if self._distance(boss.position, self.player.position) <= 1:
                reward += self._damage_entity(self.player, boss.overdrive_damage,
                                              events, "storm_overdrive")
            boss.overdrive_rounds -= 1
            if not boss.overdrive_rounds:
                boss.fatigue_rounds = 2
                events.append("storm_overdrive_exhausted")
            return reward
        if boss.fatigue_rounds:
            boss.fatigue_rounds -= 1
            events.append("storm_fatigued")
            return reward
        if boss.attacks >= boss.next_overdrive_attack and boss.target is None:
            boss.next_overdrive_attack = boss.attacks + self.rng.randint(3, 5)
            boss.attacks += 1
            boss.overdrive_rounds = 3
            events.append("storm_overdrive_charge")
            return reward
        if boss.target is None:
            boss.attack_kind = ("chain" if boss.attacks == 0 or boss.attacks % 3 == 2 else
                                self.rng.choice(("chain", "chain", "surge")))
            previous = boss.position
            player_x = self.player.position[0]
            goal_x = min(13, max(7, player_x + self.rng.choice((-3, -2, -1, 1, 2, 3))))
            speed = self.rng.choice((2, 3))
            x = boss.position[0] + max(-speed, min(speed, goal_x - boss.position[0]))
            destination = (x, self.rng.choice((5, 6, 7)))
            if destination != self.player.position:
                boss.position = destination
            if boss.position != previous:
                events.append(f"boss_move:{previous[0]}:{previous[1]}:{boss.position[0]}:{boss.position[1]}")
            boss.target = self.player.position
            events.append(f"storm_aim:{boss.attack_kind}:{boss.target[0]}:{boss.target[1]}")
            return reward
        if boss.attack_kind == "chain":
            chain = self.storm_chain()
            events.append("storm_chain:" + ":".join(f"{x}:{y}" for x, y in chain))
            grounded = (boss.target in self.relay_pads and boss.target != boss.last_ground_pad and
                        self.player.position == boss.target and
                        len(chain) == 6)
            if grounded:
                boss.last_ground_pad = boss.target
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
        if boss.attacks == 1 or self.rng.random() < (
                .7 if boss.hp <= boss.max_hp * 2 // 3 else .45):
            boss.net_target = self.player.position
            events.append(f"storm_net_place:{boss.net_target[0]}:{boss.net_target[1]}")
        if not boss.exposed_rounds:
            side = self.rng.choice((-3, -2, 2, 3))
            self._step_exposed_boss(boss,
                                    (max(7, min(13, self.player.position[0] + side)),
                                     self.rng.choice((5, 6, 7))), events)
        return reward

    def _resolve_chrono(self, boss: ChronoMantis, events: list[str]) -> float:
        for anchor, rounds in list(boss.anchor_cooldowns.items()):
            if rounds <= 1:
                del boss.anchor_cooldowns[anchor]
                events.append(f"chrono_anchor_ready:{anchor[0]}:{anchor[1]}")
            else:
                boss.anchor_cooldowns[anchor] = rounds - 1
        if boss.exposed_rounds:
            boss.exposed_rounds -= 1
            if boss.exposed_rounds == 3:
                self._step_exposed_boss(boss,
                                        (max(6, min(14, self.player.position[0] + 2)),
                                         max(7, self.player.position[1] - 2)), events)
            elif boss.exposed_rounds == 2:
                previous = boss.position
                anchors = ((7, 7), (13, 7))
                boss.position = max((cell for cell in anchors if cell != self.player.position and cell not in self.walls),
                                    key=lambda cell: (self._distance(cell, self.player.position), -cell[0]),
                                    default=previous)
                if boss.position != previous:
                    events.extend((f"boss_move:{previous[0]}:{previous[1]}:{boss.position[0]}:{boss.position[1]}",
                                   "chrono_retreat"))
                boss.retreat_target = self.player.position
                events.append(f"chrono_retreat_aim:{boss.retreat_target[0]}:{boss.retreat_target[1]}")
            elif boss.exposed_rounds == 1 and boss.retreat_target:
                target = boss.retreat_target
                events.append(f"chrono_retreat_burst:{target[0]}:{target[1]}")
                if self.player.position in ((target[0] - 1, target[1]), target, (target[0] + 1, target[1])):
                    reward = self._damage_entity(self.player, boss.echo_damage, events, "chrono_retreat")
                else:
                    reward = 0.0
                boss.retreat_target = None
                return reward
            if not boss.exposed_rounds:
                if boss.hp <= boss.max_hp * 2 // 3 and self.time_anchors != {(9, 13), (14, 13)}:
                    self.time_anchors = {(9, 13), (14, 13)}
                    events.append("chrono_anchor_shift")
                boss.phase = "flank"
                boss.retreat_target = None
                events.append("boss_shield_restored")
            return 0.0
        if boss.fatigue_rounds:
            boss.fatigue_rounds -= 1
            self._step_exposed_boss(boss, (boss.position[0], 7), events)
            events.append("chrono_fatigued")
            return 0.0
        slowed = boss.slow_rounds > 0
        if slowed:
            boss.slow_rounds -= 1
        if boss.phase == "flank":
            previous = boss.position
            launch_x = 12 if self.chrono_landing_x() == 9 else 11
            speed = 1 if slowed else 5
            delta = max(-speed, min(speed, launch_x - boss.position[0]))
            x = boss.position[0] + delta
            lane = self.rng.choice((-1, 0, 1)) if boss.moves > 1 else 0
            goal_y = min(10, max(7, self.player.position[1] - 3 + lane))
            y = boss.position[1] + max(-2, min(2, goal_y - boss.position[1]))
            if (x, y) == self.player.position:
                y = boss.position[1]
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
            flank_x = target[0] + (1 if boss.position[0] <= target[0] else -1)
            self._step_exposed_boss(boss, (max(5, min(15, flank_x)),
                                            max(7, target[1] - 1)), events)
            events.append(f"chrono_slash:{target[0]}:{target[1]}")
            if self._distance(self.player.position, target) <= 1:
                if self.player.position in self.time_anchors and self.player.position not in boss.anchor_cooldowns:
                    events.append("chrono_anchor_guard")
                damage = (boss.slash_damage // 2 if self.player.position in self.time_anchors and
                          self.player.position not in boss.anchor_cooldowns else boss.slash_damage)
                reward += self._damage_entity(self.player, damage, events, "chrono_slash")
            boss.leap_target = self.player.position
            boss.leap_countdown = 1
            boss.phase = "leap"
            events.append(f"chrono_leap_aim:{boss.leap_target[0]}:{boss.leap_target[1]}")
            return reward
        if boss.leap_countdown > 1:
            boss.leap_countdown -= 1
            previous = boss.position
            boss.position = (boss.position[0], 7 if boss.position[1] >= 8 else 8)
            if previous != boss.position:
                events.append(f"boss_move:{previous[0]}:{previous[1]}:{boss.position[0]}:{boss.position[1]}")
            events.append(f"chrono_leap_charge:{boss.leap_target[0]}:{boss.leap_target[1]}")
            return 0.0
        previous = boss.position
        target = boss.leap_target
        occupied = self.player.position == target
        if occupied:
            neighbors = (self.add(target, direction) for direction in DIRECTIONS)
            safe = (cell for cell in neighbors if self.in_bounds(cell) and
                    cell not in self.walls | self.pits | self.fires | self.spikes and
                    cell != self.player.position)
            boss.position = min(safe, key=lambda cell: (self._distance(cell, previous), cell), default=previous)
        else:
            boss.position = target
        boss.moves += 1
        events.append(f"boss_move:{previous[0]}:{previous[1]}:{boss.position[0]}:{boss.position[1]}")
        events.append(f"chrono_leap:{target[0]}:{target[1]}")
        reward = 0.0
        if (target in self.time_anchors and target not in boss.anchor_cooldowns and
                (occupied or boss.primed_anchor == target)):
            boss.exposed_rounds = 4
            boss.fatigue_rounds = 1
            boss.anchor_cooldowns[target] = 6
            events.extend(("chrono_anchor", "chrono_echo_replay", "boss_shield_break"))
            reward += 15
        elif occupied:
            reward += self._damage_entity(self.player, boss.leap_damage, events, "chrono_leap")
        elif self.player.position == boss.slash_target:
            reward += self._damage_entity(self.player, boss.echo_damage, events, "chrono_echo")
            events.append(f"chrono_echo:{boss.slash_target[0]}:{boss.slash_target[1]}")
        boss.phase = "flank"
        boss.slash_target = None
        boss.leap_target = None
        boss.primed_anchor = None
        boss.leap_countdown = 0
        if not boss.exposed_rounds and boss.moves % 4 == 0:
            boss.fatigue_rounds = 1
        elif not boss.exposed_rounds and self.rng.random() < .75:
            boss.phase = "slash"
            boss.slash_target = self.player.position
            events.append(f"chrono_slash_aim:{boss.slash_target[0]}:{boss.slash_target[1]}")
        return reward
