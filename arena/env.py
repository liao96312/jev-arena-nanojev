from __future__ import annotations

import copy
import random
from collections import deque
from dataclasses import dataclass

from .entities import (DIRECTIONS, ENEMY_MELEE_BONUS, ENEMY_MOVE_DELAY, ENEMY_SPEED_LEVEL, Action,
                       Enemy, EnemyType, Intent, IntentType, Player, PlayerLoadout)


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
    barrel_damage: int = 20
    barrel_radius: int = 1
    action_points: int = 1
    dash_ap_cost: int = 1
    emp_ap_cost: int = 1
    finish_on_all_gems: bool = False


def campaign_config(level: int) -> ArenaConfig:
    if level < 1:
        raise ValueError("level must be positive")
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
        barrel_damage=20 + (level - 1) // 3,
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
                    not self.enemy_at(target)):
                actions.append(Action(f"move_{direction}"))
            if self.enemy_at(target) or target in self.barrels:
                actions.append(Action(f"attack_{direction}"))
            if self.enemy_at(target):
                if self.in_bounds(self.add(target, direction)):
                    actions.append(Action(f"shove_{direction}"))
            destination = self.add(target, direction)
            if (not self.player.cooldowns.get("dash", 0) and self.in_bounds(target) and
                    self.in_bounds(destination) and target not in self.walls and destination not in self.walls and
                    target not in self.pits and destination not in self.pits and
                    target not in self.barrels and destination not in self.barrels and
                    not self.enemy_at(target) and not self.enemy_at(destination)):
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
                enemy = self.enemy_at(target)
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
        elif self.config.finish_on_all_gems and not self.gems:
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

    def _damage_entity(self, entity: Player | Enemy, amount: int, events: list[str], source: str) -> float:
        if entity is self.player and self.player.invulnerable:
            events.append(f"invulnerable:{source}")
            return 0.0
        actual = min(amount, entity.hp)
        entity.hp -= actual
        if entity is self.player:
            self.damage_taken += actual
            events.append(f"damage:{source}:{actual}")
            return -0.1 * actual
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

    def _ray_target(self, origin: tuple[int, int], direction: str, range_: int) -> tuple[Enemy, int] | None:
        target = origin
        for distance in range(1, range_ + 1):
            target = self.add(target, direction)
            if not self.in_bounds(target) or target in self.walls:
                return None
            enemy = self.enemy_at(target)
            if enemy:
                return enemy, distance
        return None

    def _barrel_target(self, origin: tuple[int, int], direction: str, range_: int) -> tuple[tuple[int, int], int] | None:
        target = origin
        for distance in range(1, range_ + 1):
            target = self.add(target, direction)
            if not self.in_bounds(target) or target in self.walls or self.enemy_at(target):
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
