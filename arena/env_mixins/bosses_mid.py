"""Void Angler (50), Iron Gardener (60) and Mirror Seraph (70).

Mixin for :class:`arena.env.ArenaEnv`; methods were moved verbatim from ``arena/env.py``.
"""
from __future__ import annotations

from ..boss import IronGardener, MirrorSeraph, VoidAngler, ray_cells
from ..entities import DIRECTIONS, Enemy


class MidBossMixin:
    def iron_root_ready(self, x: int) -> bool:
        seed = (x, 10)
        return seed in self.vine_walls or self.vine_seeds.get(seed) == 1

    def _void_approach(self, boss: VoidAngler, target: tuple[int, int], events: list[str]) -> None:
        previous = boss.position
        visited = {previous}
        for _ in range(2):
            x, y = boss.position
            options = ((x + dx, y + dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)
                       if dx or dy)
            safe = [cell for cell in options if self.in_bounds(cell) and cell not in visited and
                    cell not in self.walls | self.pits | self.fires | self.spikes | self.gravity_nodes and
                    cell != self.player.position]
            closer = [cell for cell in safe if self._distance(cell, target) < self._distance(boss.position, target)]
            destination = min(closer or (safe if target == self.player.position and
                                          self._distance(boss.position, target) <= 2 else []),
                              key=lambda cell: (self._distance(cell, target), cell), default=boss.position)
            if destination == boss.position:
                break
            boss.position = destination
            visited.add(destination)
        if boss.position != previous:
            events.append(f"boss_move:{previous[0]}:{previous[1]}:{boss.position[0]}:{boss.position[1]}")

    def _void_retreat(self, boss: VoidAngler, events: list[str]) -> None:
        anchors = ((7, 6), (13, 6))
        safe = (cell for cell in anchors if cell not in self.walls and cell != self.player.position)
        destination = max(safe, key=lambda cell: (self._distance(cell, self.player.position),
                                                  -cell[0]), default=boss.position)
        if destination != boss.position:
            previous = boss.position
            boss.position = destination
            events.extend((f"boss_move:{previous[0]}:{previous[1]}:{destination[0]}:{destination[1]}",
                           "void_warp"))

    def _resolve_void(self, boss: VoidAngler, events: list[str]) -> float:
        aftershock = 0.0
        for cell, rounds in list(boss.node_aftershock.items()):
            if self.player.position == cell:
                aftershock += self._damage_entity(self.player, 8, events, "void_aftershock")
            if rounds <= 1:
                del boss.node_aftershock[cell]
            else:
                boss.node_aftershock[cell] = rounds - 1
        if boss.exposed_rounds:
            boss.exposed_rounds -= 1
            if boss.exposed_rounds:
                self._void_approach(boss, self.player.position, events)
            if not boss.exposed_rounds:
                boss.drained_nodes.clear()
                events.append("boss_shield_restored")
            return aftershock
        if boss.target is None:
            remaining = self.gravity_nodes - boss.drained_nodes
            boss.attack_kind = ("mine" if remaining and boss.attacks == 0 else
                                "beam" if boss.attacks == 1 else
                                "hook" if boss.attacks == 2 else
                                "pulse" if boss.attacks >= 4 and boss.attacks % 6 == 4 else
                                "mine" if remaining and boss.attacks % 4 == 3 else
                                self.rng.choice(("mine", "mine", "beam", "hook")) if remaining else
                                self.rng.choice(("beam", "hook")))
            boss.target = (min(remaining, key=lambda cell: (self._distance(self.player.position, cell), cell))
                           if boss.attack_kind == "mine" and remaining else self.player.position)
            if boss.attack_kind == "mine":
                self._void_approach(boss, boss.target, events)
            elif boss.attack_kind == "beam":
                self._void_retreat(boss, events)
            elif boss.attack_kind == "hook":
                self._void_approach(boss, self.player.position, events)
            events.append(f"void_aim:{boss.attack_kind}:{boss.target[0]}:{boss.target[1]}")
            return aftershock
        target = boss.target
        reward = aftershock
        if boss.attack_kind == "mine":
            events.append(f"void_mine:{target[0]}:{target[1]}")
            if target in self.gravity_nodes - boss.drained_nodes and self.player.position == target:
                boss.drained_nodes.add(target)
                boss.node_aftershock[target] = 2
                events.append(f"void_drain:{len(boss.drained_nodes)}")
                reward += 15
                if len(boss.drained_nodes) == 3:
                    boss.exposed_rounds = 3
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
        elif boss.attack_kind == "pulse":
            events.append(f"void_pulse:{target[0]}:{target[1]}")
            if self._distance(self.player.position, target) <= 1 and not self.player.invulnerable:
                reward += self._damage_entity(self.player, boss.pulse_damage, events, "void_pulse")
                x, y = self.player.position
                options = (self.add((x, y), direction) for direction in DIRECTIONS)
                landing = min((cell for cell in options if self.in_bounds(cell) and
                               cell not in self.walls | self.pits | self.fires | self.spikes and
                               not self.boss_at(cell) and not self.enemy_at(cell)),
                              key=lambda cell: (self._distance(cell, boss.position), cell), default=None)
                if landing and self._distance(landing, boss.position) < self._distance((x, y), boss.position):
                    self.player.position = landing
                    events.append(f"void_pull:{x}:{y}:{landing[0]}:{landing[1]}")
            boss.attacks += 1
            boss.attack_kind = "beam"
            boss.target = self.player.position
            self._void_retreat(boss, events)
            events.append(f"void_aim:beam:{boss.target[0]}:{boss.target[1]}")
            return reward
        elif boss.attack_kind == "beam":
            events.append(f"void_beam:{boss.position[0]}:{boss.position[1]}:{target[0]}:{target[1]}")
            if self._distance(self.player.position, target) <= 1:
                reward += self._damage_entity(self.player, boss.beam_damage, events, "void_beam")
        else:
            events.append(f"void_hook_fire:{boss.position[0]}:{boss.position[1]}:{target[0]}:{target[1]}")
            if self._distance(self.player.position, target) <= 1:
                options = (self.add(boss.position, direction) for direction in DIRECTIONS)
                safe = (cell for cell in options if self.in_bounds(cell) and
                        cell not in self.walls | self.pits | self.fires | self.spikes | self.gravity_nodes and
                        not self.enemy_at(cell) and cell != self.player.position)
                landing = min(safe, key=lambda cell: (self._distance(cell, self.player.position), cell), default=None)
                if landing:
                    previous = self.player.position
                    self.player.position = landing
                    self.hooked_actions = 1
                    events.append(f"void_hook:{previous[0]}:{previous[1]}:{landing[0]}:{landing[1]}")
                    reward += self._damage_entity(self.player, boss.hook_damage, events, "void_hook")
        boss.attacks += 1
        boss.target = None
        if not boss.exposed_rounds and boss.attack_kind == "mine":
            self._void_approach(boss, target, events)
        elif not boss.exposed_rounds and boss.attack_kind == "hook":
            self._void_approach(boss, self.player.position, events)
        elif not boss.exposed_rounds and boss.attack_kind == "beam":
            self._void_retreat(boss, events)
        return reward

    def _resolve_iron(self, boss: IronGardener, events: list[str]) -> float:
        spores_reward = 0.0
        for cell, rounds in list(self.iron_spores.items()):
            if self.player.position == cell:
                spores_reward += self._damage_entity(self.player, 4, events, "iron_spores")
            if rounds <= 1:
                del self.iron_spores[cell]
                events.append(f"iron_spores_fade:{cell[0]}:{cell[1]}")
            else:
                self.iron_spores[cell] = rounds - 1
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
            if boss.exposed_rounds:
                self._step_exposed_boss(boss, (8 if self.player.position[0] >= boss.position[0] else 12, 6), events)
            if not boss.exposed_rounds:
                boss.refluxed_roots.clear()
                for vine in self.vine_walls:
                    self.walls.remove(vine)
                self.vine_walls.clear()
                self.vine_seeds = {(x, 10): 2 for x in (6, 9, 12, 15)}
                events.append("boss_shield_restored")
            return spores_reward
        if boss.target is None:
            boss.attack_kind = ("bloom" if boss.attacks >= 3 and boss.attacks % 4 == 3 else
                                "flame" if boss.attacks == 0 or boss.attacks % 3 == 2 else
                                self.rng.choice(("flame", "flame", "flame", "thorn")))
            remaining = self.root_plates - {(x, 12) for x in boss.refluxed_roots}
            boss.target = (min(remaining, key=lambda cell: (self._distance(self.player.position, cell), cell))
                           if boss.attack_kind == "flame" and remaining else self.player.position)
            self._step_exposed_boss(boss,
                                    (max(8, min(12, boss.target[0])),
                                     max(5, min(8, self.player.position[1] - 5))), events)
            events.append(f"iron_aim:{boss.attack_kind}:{boss.target[0]}:{boss.target[1]}")
            return spores_reward
        target = boss.target
        reward = spores_reward
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
                self.vine_seeds[vine] = 1
                events.append(f"iron_vine_seed:{x}:10")
        elif boss.attack_kind == "bloom":
            events.append(f"iron_bloom:{target[0]}:{target[1]}")
            if self._distance(self.player.position, target) <= 2:
                reward += self._damage_entity(self.player, boss.bloom_damage, events, "iron_bloom")
            x, y = target
            candidates = [(x + dx, y + dy) for dx in range(-2, 3) for dy in range(-2, 3)
                          if abs(dx) + abs(dy) == 2 and self.in_bounds((x + dx, y + dy)) and
                          (x + dx, y + dy) not in self.walls | self.pits | self.root_plates]
            for cell in self.rng.sample(candidates, min(4, len(candidates))):
                self.iron_spores[cell] = 2
                events.append(f"iron_spores:{cell[0]}:{cell[1]}")
        else:
            events.append(f"iron_thorn:{target[0]}:{target[1]}")
            if self._distance(self.player.position, target) <= 1:
                reward += self._damage_entity(self.player, boss.thorn_damage, events, "iron_thorn")
        boss.attacks += 1
        boss.target = None
        if not boss.exposed_rounds:
            self._step_exposed_boss(boss,
                                    (max(8, min(12, self.player.position[0])),
                                     max(5, min(8, self.player.position[1] - 5))), events)
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

    def mirror_echo_cells(self, target: tuple[int, int] | None = None) -> set[tuple[int, int]]:
        target = target if target is not None else (self.boss.echo_target if isinstance(self.boss, MirrorSeraph)
                                                   else None)
        if target is None:
            return set()
        x, y = target
        other = (x + 4 if x <= self.config.width // 2 else x - 4, y)
        return {(cx + dx, cy + dy) for cx, cy in (target, other)
                for dx, dy in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1))
                if self.in_bounds((cx + dx, cy + dy))}

    def mirror_rush_cells(self) -> set[tuple[int, int]]:
        if not isinstance(self.boss, MirrorSeraph) or self.boss.rush_target is None:
            return set()
        x, y = self.boss.rush_target
        return {(x + dx, y + dy) for dx, dy in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1))
                if self.in_bounds((x + dx, y + dy))}

    def _mirror_clone_cell(self, index: int, player_position: tuple[int, int] | None = None) -> tuple[int, int]:
        x, y = player_position or self.player.position
        return ((20 - x, y), (x, 20 - y), (20 - x, 20 - y), (y, 20 - x))[index]

    def _mirror_clone_destination(self, enemy: Enemy, player_position: tuple[int, int]) -> tuple[int, int]:
        destination = self._mirror_clone_cell(int(enemy.summoned_by[-1]), player_position)
        if (self.in_bounds(destination) and destination != player_position and
                destination not in self.walls | self.pits | self.fires | self.spikes |
                self.mirror_locks and not self.boss_at(destination) and
                all(other is enemy or other.position != destination for other in self.enemies)):
            return destination
        return enemy.position

    def _move_mirror_clones(self, events: list[str]) -> None:
        for enemy in self.enemies:
            if not enemy.summoned_by or not enemy.summoned_by.startswith("mirror_"):
                continue
            enemy.position = self._mirror_clone_destination(enemy, self.player.position)

    def _spawn_mirror_clones(self, boss: MirrorSeraph, events: list[str]) -> None:
        if boss.clones_spawned or boss.hp > boss.max_hp * 2 // 3:
            return
        boss.clones_spawned = True
        for index in range(4):
            destination = self._mirror_clone_cell(index)
            if (self.in_bounds(destination) and destination != self.player.position and
                    destination not in self.walls | self.pits | self.mirror_locks and
                    not self.boss_at(destination) and not self.enemy_at(destination)):
                self.enemies.append(Enemy(destination, hp=24, summoned_by=f"mirror_{index}"))
                events.append(f"mirror_clone_spawn:{index}:{destination[0]}:{destination[1]}")
                break

    def _mirror_clone_strike(self, boss: MirrorSeraph, events: list[str]) -> float:
        clones = [enemy for enemy in self.enemies if enemy.summoned_by and
                  enemy.summoned_by.startswith("mirror_") and
                  self._distance(enemy.position, self.player.position) <= 2]
        if not clones:
            return 0.0
        damage = min(12, 4 * len(clones))
        events.append(f"mirror_clone_strike:{len(clones)}")
        return self._damage_entity(self.player, damage, events, "mirror_clone")

    def _resolve_mirror(self, boss: MirrorSeraph, events: list[str]) -> float:
        reward = self._mirror_clone_strike(boss, events)
        if boss.silence_rounds:
            boss.silence_rounds -= 1
        if boss.exposed_rounds:
            boss.exposed_rounds -= 1
            if boss.exposed_rounds:
                self._step_exposed_boss(boss, (self.player.position[0],
                                              max(7, self.player.position[1] - 2)), events)
            if boss.echo_target:
                target = boss.echo_target
                events.append(f"mirror_echo:{target[0]}:{target[1]}")
                if self.player.position in self.mirror_echo_cells(target):
                    reward += self._damage_entity(self.player, boss.echo_damage, events, "mirror_echo")
                boss.echo_target = None
            elif boss.evaded and boss.exposed_rounds:
                boss.echo_target = self.player.position
                events.append(f"mirror_echo_aim:{boss.echo_target[0]}:{boss.echo_target[1]}")
                boss.evaded = False
            if not boss.exposed_rounds:
                boss.broken_locks.clear()
                boss.echo_target = None
                boss.evade_ready = boss.evaded = boss.emp_jammed = False
                self._spawn_mirror_clones(boss, events)
                events.append("boss_shield_restored")
            return reward
        if boss.rush_target is not None:
            target = boss.rush_target
            danger = self.mirror_rush_cells()
            previous = boss.position
            for cell in ray_cells(previous, target):
                if cell in self.walls | self.pits | self.mirror_locks or not self.in_bounds(cell):
                    break
                if cell != self.player.position:
                    boss.position = cell
            events.append(f"mirror_rush:{target[0]}:{target[1]}")
            if boss.position != previous:
                events.append(f"boss_move:{previous[0]}:{previous[1]}:{boss.position[0]}:{boss.position[1]}")
            if self.player.position in danger and self._distance(boss.position, self.player.position) <= 1:
                reward += self._damage_entity(self.player, boss.rush_damage, events, "mirror_rush")
            boss.rush_target = None
            return reward
        if boss.copied_action is None:
            if boss.copies >= 3 * (boss.rushes + 1):
                boss.rush_target = self.player.position
                boss.rushes += 1
                events.append(f"mirror_rush_aim:{boss.rush_target[0]}:{boss.rush_target[1]}")
                return reward
            action = self.last_non_wait_action or "move_n"
            boss.copied_action = action
            direction = action[-1] if action[-1] in "nsew" else None
            boss.mirrored_direction = {"e": "w", "w": "e"}.get(direction, direction)
            # Sidestep to line up the mirrored shot; never shift a locked warning.
            desired_x = 11 if boss.mirrored_direction == "w" else 9 if boss.mirrored_direction == "e" else 10
            destination = (desired_x, 7 if boss.mirrored_direction in ("n", "s") else 6)
            if destination != boss.position and destination != self.player.position:
                previous = boss.position
                boss.position = destination
                events.append(f"boss_move:{previous[0]}:{previous[1]}:{destination[0]}:{destination[1]}")
            path = self.mirror_ray()
            boss.target = path[-1] if path else None
            boss.echo_target = self.player.position if direction else None
            events.append(f"mirror_aim:{action}:{boss.mirrored_direction or '-'}")
            if boss.echo_target:
                events.append(f"mirror_echo_aim:{boss.echo_target[0]}:{boss.echo_target[1]}")
            return reward
        action = boss.copied_action
        path = self.mirror_ray()
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
                    boss.exposed_rounds = 6
                    boss.evade_ready = True
                    events.append("boss_shield_break")
            elif self.player.position in path or any(self._distance(self.player.position, cell) == 1
                                                     for cell in path):
                damage = boss.dash_damage if action.startswith("dash_") else boss.shard_damage
                reward += self._damage_entity(self.player, damage, events, "mirror_shard")
        if boss.echo_target:
            events.append(f"mirror_echo:{boss.echo_target[0]}:{boss.echo_target[1]}")
            if self.player.position in self.mirror_echo_cells():
                reward += self._damage_entity(self.player, boss.echo_damage, events, "mirror_echo")
        boss.copied_action = boss.mirrored_direction = boss.target = None
        boss.echo_target = None
        boss.copies += 1
        return reward
