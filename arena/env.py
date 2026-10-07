from __future__ import annotations

import copy
import math
import random
from collections import deque
from dataclasses import dataclass

# Some of these names are no longer used here directly (the logic lives in arena/env_mixins/),
# but they stay importable from arena.env for backward compatibility.
from .entities import (DIRECTIONS, ENEMY_MELEE_BONUS, ENEMY_MOVE_DELAY, ENEMY_SPEED_LEVEL, Action,
                       Enemy, EnemyType, Intent, IntentType, Player, PlayerLoadout)
from .boss import ApexArbiter, ChronoMantis, FurnaceHydra, IronGardener, MirrorSeraph, NullWeaver, PrismWarden, SiegeLeviathan, StormChoir, VoidAngler, ray_cells
from .env_mixins import (CombatMixin, EarlyBossMixin, EnemyAIMixin, LateBossMixin, MidBossMixin, ObservationMixin,
                         TerrainMixin)


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
        return ArenaConfig(difficulty_level=level, max_ticks=400 if level in (10, 100) else 300, walls=0, enemies=0, gems=0,
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


class ArenaEnv(TerrainMixin, CombatMixin, EnemyAIMixin, EarlyBossMixin, MidBossMixin, LateBossMixin,
               ObservationMixin):
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
        self.hooked_actions = 0
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
            if (self.enemy_at(target) or self.boss_at(target) or target in self.barrels or
                    target == getattr(self.boss, "gate", None) and target in self.apex_cage or
                    isinstance(self.boss, FurnaceHydra) and self.boss.target and
                    self.boss.attack_kind in ("wave", "triple") and
                    target == (self.boss.head_x, 12) and
                    self.boss.head_x not in self.boss.valves_opened or
                    isinstance(self.boss, VoidAngler) and self.boss.target == target and
                    self.boss.attack_kind == "mine" and target not in self.boss.drained_nodes or
                    isinstance(self.boss, IronGardener) and self.boss.target and
                    self.boss.attack_kind == "flame" and target in self.vine_walls and
                    target[0] == self.boss.target[0] or
                    isinstance(self.boss, ChronoMantis) and self.boss.phase == "leap" and
                    target == self.boss.leap_target and target in self.time_anchors or
                    isinstance(self.boss, NullWeaver) and target in self.null_nodes and
                    not self.boss.exposed_rounds):
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
            if isinstance(self.boss, PrismWarden) and target in self.reflectors - self.boss.used_reflectors:
                destination = self.add(target, direction)
                if (self.in_bounds(destination) and destination not in self.walls | self.pits |
                        self.fires | self.spikes | self.barrels | self.reflectors and
                        destination not in self.medkits | self.energy_cells | self.pistol_pickups and
                        not self.enemy_at(destination) and not self.boss_at(destination)):
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
                 self._distance(self.player.position, self.boss.position) <=
                 (3 if self.boss.exposed_rounds else self.config.emp_radius) or
                 isinstance(self.boss, StormChoir) and self.boss.target and
                 self.boss.attack_kind == "chain" and
                 any(self._distance(self.player.position, pad) <= 2
                     for pad in self.relay_pads))):
            actions.append(Action.EMP)
        ranged.sort(key=lambda item: item[0])
        actions.extend(action for _, action in ranged[:max(0, 11 - len(actions))])
        actions.append(Action.WAIT)
        locked = self.boss.blocked_kind if isinstance(self.boss, NullWeaver) else None
        return [action for action in actions if self._action_cost(action) <= self.ap_remaining and
                not (self.hooked_actions and action.value.startswith(("move_", "dash_"))) and
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
            self.apex_fast_volley_resolved = False
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
            elif (isinstance(self.boss, FurnaceHydra) and self.boss.target and
                  self.boss.attack_kind in ("wave", "triple") and
                  target == (self.boss.head_x, 12) and
                  self.boss.head_x not in self.boss.valves_opened):
                self.boss.valves_opened.add(self.boss.head_x)
                self.boss.valve_heat[self.boss.head_x] = 2
                events.append(f"furnace_valve_strike:{self.boss.head_x}")
                reward += 15
            elif (isinstance(self.boss, VoidAngler) and self.boss.attack_kind == "mine" and
                  self.boss.target == target and target not in self.boss.drained_nodes):
                self.boss.drained_nodes.add(target)
                self.boss.node_aftershock[target] = 2
                events.append(f"void_node_cut:{target[0]}:{target[1]}")
                reward += 15
                if len(self.boss.drained_nodes) == 3:
                    self.boss.exposed_rounds = 3
                    self.boss.target = None
                    events.append("boss_shield_break")
            elif (isinstance(self.boss, IronGardener) and self.boss.target and
                  self.boss.attack_kind == "flame" and target in self.vine_walls and
                  target[0] == self.boss.target[0]):
                self.vine_walls.remove(target)
                self.walls.remove(target)
                self.boss.refluxed_roots.add(target[0])
                events.append(f"iron_vine_cut:{target[0]}:{target[1]}")
                reward += 15
                if len(self.boss.refluxed_roots) == 4:
                    self.boss.exposed_rounds = 4
                    self.boss.target = None
                    events.append("boss_shield_break")
            elif (isinstance(self.boss, ChronoMantis) and self.boss.phase == "leap" and
                  target == self.boss.leap_target and target in self.time_anchors and
                  target not in self.boss.anchor_cooldowns):
                self.boss.primed_anchor = target
                events.append(f"chrono_anchor_prime:{target[0]}:{target[1]}")
            elif isinstance(self.boss, NullWeaver) and target in self.null_nodes and not self.boss.exposed_rounds:
                reward += self._activate_null_node(target, events)
                events.append(f"null_node_strike:{target[0]}:{target[1]}")
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
                if position in self.reflectors:
                    destination = self.add(position, direction)
                    self.reflectors.remove(position)
                    self.reflectors.add(destination)
                    events.append(f"prism_reflector_shove:{position[0]}:{position[1]}:{destination[0]}:{destination[1]}")
                else:
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
            if (isinstance(self.boss, MirrorSeraph) and self.boss.exposed_rounds and
                    self._distance(self.player.position, self.boss.position) <= 3):
                self.boss.emp_jammed = True
                events.append("mirror_emp_jam")
            if (isinstance(self.boss, StormChoir) and self.boss.target and
                    self.boss.attack_kind == "chain" and
                    any(self._distance(self.player.position, pad) <= 2
                        for pad in self.relay_pads)):
                self.boss.exposed_rounds = 3
                self.boss.target = None
                events.extend(("storm_emp_ground", "boss_shield_break"))
                reward += 15
            self.player.cooldowns["emp"] = self.config.emp_cooldown
            events.append(f"emp:{len(affected)}")

        if self.player.hp > 0 and self.player.position in self.fires:
            reward += self._damage_entity(self.player, self.config.fire_damage, events, "fire")
            if action.value.startswith("move_"):
                reward -= 3

        if action != Action.WAIT:
            self.last_non_wait_action = action.value
        if isinstance(self.boss, MirrorSeraph):
            self._move_mirror_clones(events)
        if self.hooked_actions:
            self.hooked_actions -= 1
            if not self.hooked_actions:
                events.append("void_hook_release")
        self.ap_remaining -= self._action_cost(action)
        if (self.player.hp > 0 and self.ap_remaining == 1 and
                isinstance(self.boss, ApexArbiter) and self.boss.barrage_volley == 2 and
                self.boss.countdown == 1):
            reward += self._resolve_apex(self.boss, events)
            self.apex_fast_volley_resolved = True
            events.append("apex_fast_barrage")
        if self.player.hp > 0 and self.ap_remaining == 0:
            reward += self._resolve_enemy_intents(events)
            if not self.apex_fast_volley_resolved:
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

    @staticmethod
    def _distance(a: tuple[int, int], b: tuple[int, int]) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
