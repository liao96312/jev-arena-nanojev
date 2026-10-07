"""Regular enemy AI: intent planning, pathing and intent resolution (shots, explosions, charges).

Mixin for :class:`arena.env.ArenaEnv`; methods were moved verbatim from ``arena/env.py``.
"""
from __future__ import annotations

from ..entities import DIRECTIONS, ENEMY_MELEE_BONUS, ENEMY_MOVE_DELAY, ENEMY_SPEED_LEVEL, Enemy, EnemyType, Intent, IntentType
from collections import deque


class EnemyAIMixin:
    def _plan_enemy_intents(self, enemies: list[Enemy] | None = None) -> None:
        occupied = {enemy.position for enemy in self.enemies}
        for enemy in enemies or self.enemies:
            if enemy.summoned_by and enemy.summoned_by.startswith("mirror_"):
                enemy.intent = None
                continue
            if enemy.stunned:
                enemy.intent = Intent(IntentType.WAIT, countdown=1)
                continue
            distance = self._distance(enemy.position, self.player.position)
            if enemy.enemy_type == EnemyType.BOMBER and distance <= self.config.bomber_radius + 1:
                enemy.intent = Intent(IntentType.EXPLODE, countdown=2, power=self.config.bomber_damage)
                continue
            if distance == 1:
                if enemy.enemy_type == EnemyType.RAZOR_HOUND and self.round < self.config.spawn_protection_rounds:
                    enemy.intent = Intent(IntentType.WAIT)
                    continue
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
            path = self._enemy_path(
                enemy.position, occupied - {enemy.position},
                seek_shot=enemy.enemy_type == EnemyType.ARCHER and self.round > self.config.spawn_protection_rounds)
            if not path:
                enemy.intent = Intent(IntentType.WAIT)
                continue
            direction = next(name for name in DIRECTIONS if self.add(enemy.position, name) == path[0])
            if enemy.enemy_type == EnemyType.RAZOR_HOUND:
                enemy.intent = Intent(IntentType.SPRINT, direction, 1, path=path[:2])
            else:
                enemy.intent = Intent(IntentType.MOVE, direction, self._enemy_move_interval(enemy))

    def _enemy_path(self, start: tuple[int, int], occupied: set[tuple[int, int]],
                    seek_shot: bool = False) -> tuple[tuple[int, int], ...]:
        queue = deque((start,))
        paths = {start: ()}
        melee_route = ()
        while queue:
            position = queue.popleft()
            path = paths[position]
            if position != start and self._distance(position, self.player.position) == 1:
                if not seek_shot:
                    return path
                melee_route = melee_route or path
            if seek_shot and position != start and self._distance(position, self.player.position) >= 3:
                dx = self.player.position[0] - position[0]
                dy = self.player.position[1] - position[1]
                if dx == 0 or dy == 0:
                    direction = "e" if dx > 0 else "w" if dx < 0 else "s" if dy > 0 else "n"
                    if self._clear_shot_to_player(position, direction):
                        return path
            for direction in DIRECTIONS:
                target = self.add(position, direction)
                if (target in paths or not self.in_bounds(target) or target in self.walls or target in self.fires or
                        target in self.spikes or target in self.pits or target in self.barrels or
                        target in occupied or target == self.player.position):
                    continue
                paths[target] = path + (target,)
                queue.append(target)
        return melee_route

    def _enemy_move_interval(self, enemy: Enemy) -> int:
        if enemy.summoned_by == "furnace":
            return 1
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
            if enemy.summoned_by and enemy.summoned_by.startswith("mirror_"):
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
            elif intent.kind == IntentType.SPRINT:
                origin = enemy.position
                traversed = []
                for target in intent.path:
                    if (not self.in_bounds(target) or target in self.walls or target in self.fires or
                            target in self.spikes or target in self.pits or target in self.barrels or
                            target in occupied or target == self.player.position):
                        break
                    occupied.remove(enemy.position)
                    enemy.position = target
                    occupied.add(target)
                    traversed.append(target)
                if enemy.position != origin:
                    events.append("enemy_move")
                    events.append(f"hound_sprint:{origin[0]}:{origin[1]}:" +
                                  ":".join(f"{x}:{y}" for x, y in traversed))
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
