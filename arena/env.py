from __future__ import annotations

import copy
import random
from dataclasses import dataclass

from .entities import DIRECTIONS, Action, Enemy, EnemyType, Intent, IntentType, Player


@dataclass(frozen=True)
class ArenaConfig:
    width: int = 20
    height: int = 20
    max_ticks: int = 500
    walls: int = 35
    enemies: int = 3
    gems: int = 6
    fires: int = 10
    medkits: int = 2
    enemy_damage: int = 5
    fire_damage: int = 10
    attack_damage: int = 20
    heal_amount: int = 35
    enemy_move_interval: int = 1
    charger_ratio: float = 0.25
    bomber_ratio: float = 0.15
    charger_range: int = 4
    charger_damage: int = 10
    collision_damage: int = 15
    bomber_damage: int = 20
    bomber_radius: int = 1
    finish_on_all_gems: bool = False


def campaign_config(level: int) -> ArenaConfig:
    if level < 1:
        raise ValueError("level must be positive")
    return ArenaConfig(
        max_ticks=min(300, 160 + level * 20),
        walls=min(55, 24 + level * 3),
        enemies=min(8, 1 + (level + 1) // 2),
        gems=min(8, 2 + (level + 1) // 2),
        fires=min(24, 4 + level * 2),
        medkits=max(1, 3 - level // 3),
        enemy_damage=min(12, 4 + (level - 1) // 2),
        fire_damage=min(18, 8 + level),
        enemy_move_interval=2,
        charger_ratio=min(0.45, 0.2 + level * 0.03),
        bomber_ratio=min(0.25, 0.1 + level * 0.02),
        finish_on_all_gems=True,
    )


@dataclass(frozen=True)
class StepResult:
    reward: float
    done: bool
    events: tuple[str, ...]


class ArenaEnv:
    def __init__(self, config: ArenaConfig | None = None):
        self.config = config or ArenaConfig()
        self.reset(0)

    def reset(self, seed: int = 0) -> dict:
        if self.config.enemy_move_interval < 1 or self.config.charger_range < 1:
            raise ValueError("enemy intervals and ranges must be positive")
        if (not 0 <= self.config.charger_ratio <= 1 or not 0 <= self.config.bomber_ratio <= 1 or
                self.config.charger_ratio + self.config.bomber_ratio > 1):
            raise ValueError("enemy type ratios must be between zero and one and sum to at most one")
        self.seed = seed
        self.rng = random.Random(seed)
        self.tick = self.score = self.gems_collected = self.kills = self.environment_kills = 0
        self.damage_taken = 0
        self.previous_player_position: tuple[int, int] | None = None
        self.last_action: str | None = None
        self.done = False
        cells = [(x, y) for y in range(self.config.height) for x in range(self.config.width)]
        self.rng.shuffle(cells)
        needed = 1 + self.config.walls + self.config.enemies + self.config.gems + self.config.fires + self.config.medkits
        if needed > len(cells):
            raise ValueError("map contains more entities than cells")

        take = iter(cells)
        self.player = Player(next(take))
        self.walls = {next(take) for _ in range(self.config.walls)}
        self.enemies = []
        for _ in range(self.config.enemies):
            roll = self.rng.random()
            enemy_type = (EnemyType.BOMBER if roll < self.config.bomber_ratio else
                          EnemyType.CHARGER if roll < self.config.bomber_ratio + self.config.charger_ratio else
                          EnemyType.CHASER)
            self.enemies.append(Enemy(next(take), enemy_type=enemy_type))
        self.gems = {next(take) for _ in range(self.config.gems)}
        self.fires = {next(take) for _ in range(self.config.fires)}
        self.medkits = {next(take) for _ in range(self.config.medkits)}
        self._plan_enemy_intents()
        return self.observation()

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

    def legal_actions(self) -> list[Action]:
        actions: list[Action] = []
        for direction in DIRECTIONS:
            target = self.add(self.player.position, direction)
            if self.in_bounds(target) and target not in self.walls and not self.enemy_at(target):
                actions.append(Action(f"move_{direction}"))
            if self.enemy_at(target):
                actions.append(Action(f"attack_{direction}"))
                if self.in_bounds(self.add(target, direction)):
                    actions.append(Action(f"shove_{direction}"))
        if self.player.medkits and self.player.hp < 100:
            actions.append(Action.HEAL)
        actions.append(Action.WAIT)
        return actions

    def step(self, action: Action | str) -> StepResult:
        if self.done:
            raise RuntimeError("episode is already finished")
        action = Action(action)
        if action not in self.legal_actions():
            raise ValueError(f"illegal action: {action}")

        origin = self.player.position
        reward = 0.05
        events: list[str] = []
        if action.value.startswith("move_"):
            self.player.position = self.add(self.player.position, action.value[-1])
            reward += self._collect(events)
            if self.player.position in self.fires:
                reward += self._damage_entity(self.player, self.config.fire_damage, events, "fire") - 3
        elif action.value.startswith("attack_"):
            enemy = self.enemy_at(self.add(self.player.position, action.value[-1]))
            assert enemy is not None
            events.append("attack")
            reward += self._damage_entity(enemy, self.config.attack_damage, events, "attack")
        elif action.value.startswith("shove_"):
            direction = action.value[-1]
            enemy = self.enemy_at(self.add(self.player.position, direction))
            assert enemy is not None
            reward += self._push_entity(enemy, direction, events)
        elif action == Action.HEAL:
            self.player.medkits -= 1
            self.player.hp = min(100, self.player.hp + self.config.heal_amount)
            events.append("heal")

        if self.player.hp > 0:
            reward += self._resolve_enemy_intents(events)
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
        return StepResult(round(reward, 4), self.done, tuple(events))

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
        return reward

    def _damage_entity(self, entity: Player | Enemy, amount: int, events: list[str], source: str) -> float:
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
        if source != "attack":
            self.environment_kills += 1
            events.append("environment_kill")
        events.append("kill")
        return 20.0 if source == "attack" else 10.0

    def _push_entity(self, enemy: Enemy, direction: str, events: list[str]) -> float:
        destination = self.add(enemy.position, direction)
        events.append(f"shove:{direction}")
        if destination in self.walls:
            return self._damage_entity(enemy, self.config.collision_damage, events, "wall")
        blocker = self.enemy_at(destination)
        if blocker:
            reward = self._damage_entity(blocker, self.config.collision_damage, events, "collision")
            if enemy in self.enemies:
                reward += self._damage_entity(enemy, self.config.collision_damage, events, "collision")
            return reward
        enemy.position = destination
        if destination in self.fires:
            return self._damage_entity(enemy, self.config.fire_damage, events, "fire")
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
                enemy.intent = Intent(IntentType.MELEE, direction, self.config.enemy_move_interval,
                                      self.config.enemy_damage)
                continue
            dx = self.player.position[0] - enemy.position[0]
            dy = self.player.position[1] - enemy.position[1]
            if enemy.enemy_type == EnemyType.CHARGER and (dx == 0 or dy == 0):
                direction = "e" if dx > 0 else "w" if dx < 0 else "s" if dy > 0 else "n"
                enemy.intent = Intent(IntentType.CHARGE, direction, self.config.enemy_move_interval,
                                      self.config.charger_damage)
                continue
            directions = []
            for direction in DIRECTIONS:
                target = self.add(enemy.position, direction)
                if (self.in_bounds(target) and target not in self.walls and target not in occupied and
                        target != self.player.position and
                        self._distance(target, self.player.position) < distance):
                    directions.append(direction)
            enemy.intent = Intent(IntentType.MOVE, self.rng.choice(directions),
                                  self.config.enemy_move_interval) if directions else Intent(IntentType.WAIT)

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
            elif intent.kind == IntentType.MOVE and intent.direction:
                target = self.add(enemy.position, intent.direction)
                if (self.in_bounds(target) and target not in self.walls and target not in occupied and
                        target != self.player.position):
                    occupied.remove(enemy.position)
                    enemy.position = target
                    occupied.add(target)
                    events.append("enemy_move")
                    if target in self.fires:
                        reward += self._damage_entity(enemy, self.config.fire_damage, events, "fire")
                        if enemy not in self.enemies:
                            occupied.discard(target)
            elif intent.kind == IntentType.CHARGE and intent.direction:
                reward += self._resolve_charge(enemy, intent.direction, occupied, events)
            elif intent.kind == IntentType.EXPLODE:
                reward += self._resolve_explosion(enemy, intent.power, events)
                occupied.discard(enemy.position)
            if enemy in self.enemies:
                resolved.append(enemy)
        if resolved:
            self._plan_enemy_intents(resolved)
        return reward

    def _resolve_explosion(self, bomber: Enemy, damage: int, events: list[str]) -> float:
        center = bomber.position
        self.enemies.remove(bomber)
        events.append("bomber_explode")
        reward = 0.0
        if self._distance(center, self.player.position) <= self.config.bomber_radius:
            reward += self._damage_entity(self.player, damage, events, "explosion")
        for enemy in list(self.enemies):
            if self._distance(center, enemy.position) <= self.config.bomber_radius:
                reward += self._damage_entity(enemy, damage, events, "explosion")
        return reward

    def _resolve_charge(self, enemy: Enemy, direction: str, occupied: set[tuple[int, int]],
                        events: list[str]) -> float:
        reward = 0.0
        for _ in range(self.config.charger_range):
            target = self.add(enemy.position, direction)
            if not self.in_bounds(target) or target in self.walls:
                events.append("charge_blocked")
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
                break
            occupied.remove(enemy.position)
            enemy.position = target
            occupied.add(target)
            events.append("charger_move")
            if target in self.fires:
                reward += self._damage_entity(enemy, self.config.fire_damage, events, "fire")
                if enemy not in self.enemies:
                    occupied.discard(target)
                    break
        return reward

    @staticmethod
    def _distance(a: tuple[int, int], b: tuple[int, int]) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def observation(self) -> dict:
        return {
            "seed": self.seed,
            "tick": self.tick,
            "hp": self.player.hp,
            "score": self.score,
            "position": self.player.position,
            "previous_position": self.previous_player_position,
            "last_action": self.last_action,
            "medkits_carried": self.player.medkits,
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
            "medkits": tuple(sorted(self.medkits)),
            "gems_collected": self.gems_collected,
            "kills": self.kills,
            "environment_kills": self.environment_kills,
            "damage_taken": self.damage_taken,
            "done": self.done,
        }
