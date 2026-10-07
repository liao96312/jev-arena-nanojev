"""Threat preview and the observation dict exposed to agents.

Mixin for :class:`arena.env.ArenaEnv`; methods were moved verbatim from ``arena/env.py``.
"""
from __future__ import annotations

from ..boss import ApexArbiter, ChronoMantis, FurnaceHydra, IronGardener, MirrorSeraph, NullWeaver, PrismWarden, SiegeLeviathan, StormChoir, VoidAngler
from ..entities import IntentType


class ObservationMixin:
    def imminent_threats(self, position: tuple[int, int] | None = None) -> tuple[tuple[str, int], ...]:
        position = position or self.player.position
        threats: list[tuple[str, int]] = []
        if isinstance(self.boss, PrismWarden) and self.boss.target:
            rays = self.prism_rays()
            if any(position in ray for ray in rays[1:]) or (
                    rays[0] and rays[0][-1] not in self.walls | self.reflectors and
                    position in self.prism_attack_cells()):
                threats.append(("prism_warden/beam", self.boss.beam_damage))
        if isinstance(self.boss, PrismWarden) and self.boss.lunge_target and self._distance(position, self.boss.lunge_target) <= 1:
            threats.append(("prism_warden/lunge", self.boss.lunge_damage))
        if isinstance(self.boss, FurnaceHydra) and self.boss.target:
            if self.boss.attack_kind in ("wave", "triple") and 8 <= position[1] <= 16:
                columns = self.boss.wave_columns or ((self.boss.head_x - 1, self.boss.head_x,
                                                      self.boss.head_x + 1)
                                                     if self.boss.hp <= self.boss.max_hp // 2 else
                                                     (self.boss.head_x,))
                safe_valve = (position in self.coolant_valves and
                              position[0] not in self.boss.valves_opened and
                              not self.boss.wave_rapid and
                              (self.boss.attack_kind == "triple" or position[0] == self.boss.head_x))
                if position[0] in columns and not safe_valve:
                    damage = (self.boss.wave_damage if self.boss.attack_kind == "triple" or
                              position[0] == self.boss.head_x else 12)
                    threats.append(("furnace_hydra/wave", damage))
            elif self.boss.attack_kind == "fireball" and self._distance(position, self.boss.target) <= 1:
                threats.append(("furnace_hydra/fireball", self.boss.fireball_damage))
        if (isinstance(self.boss, FurnaceHydra) and position in self.coolant_valves and
                position[0] in self.boss.valve_heat and self.boss.valve_heat[position[0]] <= 1):
            threats.append(("furnace_hydra/valve_heat", 10))
        if isinstance(self.boss, StormChoir) and self.boss.target:
            if self.boss.attack_kind == "chain":
                chain = self.storm_chain()
                if ((position == self.boss.target or position in chain[2:]) and
                        not (position == self.boss.target and position in self.relay_pads and
                             position != self.boss.last_ground_pad and len(chain) == 6)):
                    threats.append(("storm_choir/arc", self.boss.arc_damage))
            elif self._distance(position, self.boss.target) <= 1:
                damage = self.boss.surge_damage + (4 if self.boss.hp <= self.boss.max_hp * 2 // 3 else 0)
                threats.append(("storm_choir/surge", damage))
        if isinstance(self.boss, StormChoir) and self.boss.net_target:
            if self._distance(position, self.boss.net_target) <= 1:
                threats.append(("storm_choir/net", self.boss.net_damage))
        if isinstance(self.boss, StormChoir) and self.boss.overdrive_rounds:
            if self._distance(position, self.boss.position) <= 5:
                threats.append(("storm_choir/overdrive", self.boss.overdrive_damage))
        if isinstance(self.boss, ChronoMantis):
            if (self.boss.phase == "slash" and
                    self._distance(position, self.boss.slash_target) <= 1):
                damage = (self.boss.slash_damage // 2 if position in self.time_anchors and
                          position not in self.boss.anchor_cooldowns else
                          self.boss.slash_damage)
                threats.append(("chrono_mantis/slash", damage))
            if self.boss.phase == "leap":
                if (position == self.boss.leap_target and
                        (position not in self.time_anchors or position in self.boss.anchor_cooldowns)):
                    threats.append(("chrono_mantis/leap", self.boss.leap_damage))
                elif position == self.boss.slash_target and position != self.boss.leap_target:
                    threats.append(("chrono_mantis/echo", self.boss.echo_damage))
            if (self.boss.retreat_target and self.boss.exposed_rounds == 2 and
                    position[1] == self.boss.retreat_target[1] and
                    abs(position[0] - self.boss.retreat_target[0]) <= 1):
                threats.append(("chrono_mantis/retreat", self.boss.echo_damage))
        if isinstance(self.boss, VoidAngler) and self.boss.attack_kind == "beam" and self.boss.target:
            if self._distance(position, self.boss.target) <= 1:
                threats.append(("void_angler/beam", self.boss.beam_damage))
        if isinstance(self.boss, VoidAngler) and self.boss.attack_kind == "hook" and self.boss.target:
            if self._distance(position, self.boss.target) <= 1:
                threats.append(("void_angler/hook", self.boss.hook_damage))
        if isinstance(self.boss, VoidAngler) and self.boss.attack_kind == "pulse" and self.boss.target:
            if self._distance(position, self.boss.target) <= 1:
                threats.append(("void_angler/pulse", self.boss.pulse_damage))
        if isinstance(self.boss, VoidAngler) and position in self.boss.node_aftershock:
            threats.append(("void_angler/aftershock", 8))
        if isinstance(self.boss, IronGardener) and self.boss.target:
            if self.boss.attack_kind == "flame" and position[0] == self.boss.target[0] and 8 <= position[1] <= 16:
                if not (position == self.boss.target and self.iron_root_ready(position[0])):
                    threats.append(("iron_gardener/flame", self.boss.flame_damage))
            elif self.boss.attack_kind == "bloom" and self._distance(position, self.boss.target) <= 2:
                threats.append(("iron_gardener/bloom", self.boss.bloom_damage))
            elif self.boss.attack_kind == "thorn" and self._distance(position, self.boss.target) <= 1:
                threats.append(("iron_gardener/thorn", self.boss.thorn_damage))
        if isinstance(self.boss, IronGardener) and position in self.iron_spores:
            threats.append(("iron_gardener/spores", 4))
        if (isinstance(self.boss, MirrorSeraph) and self.boss.copied_action and
                any(self._distance(position, cell) <= 1 for cell in self.mirror_ray())):
            path = self.mirror_ray()
            if path[-1] not in self.mirror_locks or path[-1] in self.boss.broken_locks:
                damage = (self.boss.dash_damage if self.boss.copied_action.startswith("dash_")
                          else self.boss.shard_damage)
                threats.append(("mirror_seraph/shard", damage))
        if (isinstance(self.boss, MirrorSeraph) and
                position in self.mirror_echo_cells()):
            threats.append(("mirror_seraph/echo", self.boss.echo_damage))
        if isinstance(self.boss, MirrorSeraph) and position in self.mirror_rush_cells():
            threats.append(("mirror_seraph/rush", self.boss.rush_damage))
        if isinstance(self.boss, MirrorSeraph):
            clones = sum(1 for enemy in self.enemies if enemy.summoned_by and
                         enemy.summoned_by.startswith("mirror_") and
                         self._distance(position, self._mirror_clone_destination(enemy, position)) <= 2)
            if clones:
                threats.append(("mirror_seraph/clone", min(12, 4 * clones)))
        if isinstance(self.boss, SiegeLeviathan) and self.boss.rail_target is not None and self.boss.charge <= 1:
            if self.rail_threatens(position):
                threats.append(("siege_leviathan/railgun", self.boss.rail_damage))
        if isinstance(self.boss, SiegeLeviathan) and position in self.siege_blast_cells():
            threats.append(("siege_leviathan/blast", {"cross": 22, "pit": 28, "shrapnel": 18}[self.boss.blast_kind]))
        if isinstance(self.boss, SiegeLeviathan) and position in self.siege_charge_path():
            threats.append(("siege_leviathan/charge", 28))
        if isinstance(self.boss, NullWeaver) and self.boss.erase_countdown == 1:
            if position in self.boss.erase_targets:
                threats.append(("null_weaver/fracture", self.boss.fracture_damage))
        if isinstance(self.boss, ApexArbiter) and self.boss.countdown == 1:
            if position in self.boss.danger:
                power = {"cage": 26, "barrage": 16, "charge": 32, "gravity": 24,
                         "cage_barrage": 36, "charge_gravity": 36, "verdict": 36}[self.boss.kind]
                if not (self.boss.kind == "cage" and self.boss.gate_broken or
                        self.boss.kind == "cage_barrage" and self.boss.gate_broken and
                        position not in self.boss.barrage_cells or
                        self.boss.kind == "charge" and self.apex_seals[2] in self.boss.danger or
                        self.boss.kind == "gravity" and position == self.apex_seals[3]):
                    threats.append((f"apex_arbiter/{self.boss.kind}", power))
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
            "last_non_wait_action": self.last_non_wait_action,
            "medkits_carried": self.player.medkits,
            "loadout": (self.player.loadout.bow, self.player.loadout.pistol,
                        self.player.loadout.arrows, self.player.loadout.energy),
            "cooldowns": tuple(sorted(self.player.cooldowns.items())),
            "hooked_actions": self.hooked_actions,
            "walls": tuple(sorted(self.walls)),
            "enemies": tuple((enemy.position, enemy.hp) for enemy in self.enemies),
            "boss": (("prism", self.boss.position, self.boss.hp, self.boss.reflections,
                      tuple(sorted(self.boss.used_reflectors)), self.boss.shots_fired,
                      self.boss.exposed_rounds, self.boss.target, self.boss.lunge_target,
                      self.boss.returning, tuple(sorted(self.boss.reflector_cooldowns.items())),
                      self.boss.reflector_lockout,
                      self.boss.spin_step) if isinstance(self.boss, PrismWarden) else
                     ("furnace", self.boss.position, self.boss.hp, tuple(sorted(self.boss.valves_opened)),
                      tuple(sorted(self.boss.valve_heat.items())),
                      self.boss.exposed_rounds, self.boss.attack_kind, self.boss.target,
                      self.boss.attacks, self.boss.summons, self.boss.wave_columns,
                      self.boss.wave_rapid, self.boss.combo_queued,
                      self.boss.non_wave_streak) if isinstance(self.boss, FurnaceHydra) else
                     ("storm", self.boss.position, self.boss.hp, self.boss.exposed_rounds,
                      self.boss.attack_kind, self.boss.target, self.boss.attacks,
                      self.boss.net_target, self.boss.last_ground_pad,
                      self.boss.overdrive_rounds, self.boss.fatigue_rounds,
                      self.boss.next_overdrive_attack) if isinstance(self.boss, StormChoir) else
                     ("chrono", self.boss.position, self.boss.hp, self.boss.exposed_rounds,
                      self.boss.phase, self.boss.slash_target, self.boss.leap_target,
                      self.boss.retreat_target, self.boss.primed_anchor,
                      self.boss.leap_countdown, self.boss.moves,
                      self.boss.slow_rounds, self.boss.fatigue_rounds,
                      tuple(sorted(self.boss.anchor_cooldowns.items()))) if isinstance(self.boss, ChronoMantis) else
                     ("void", self.boss.position, self.boss.hp, self.boss.exposed_rounds,
                      tuple(sorted(self.boss.drained_nodes)), self.boss.attack_kind,
                      self.boss.target, self.boss.attacks,
                      tuple(sorted(self.boss.node_aftershock.items()))) if isinstance(self.boss, VoidAngler) else
                     ("iron", self.boss.position, self.boss.hp, self.boss.exposed_rounds,
                      tuple(sorted(self.boss.refluxed_roots)), self.boss.attack_kind,
                      self.boss.target, self.boss.attacks, self.boss.summons) if isinstance(self.boss, IronGardener) else
                     ("mirror", self.boss.position, self.boss.hp, self.boss.exposed_rounds,
                      tuple(sorted(self.boss.broken_locks)), self.boss.copied_action,
                      self.boss.mirrored_direction, self.boss.target,
                      self.boss.echo_target, self.boss.evade_ready, self.boss.evaded,
                      self.boss.emp_jammed, self.boss.clones_spawned,
                      self.boss.copies, self.boss.rushes, self.boss.rush_target) if isinstance(self.boss, MirrorSeraph) else
                     ("siege", self.boss.position, self.boss.hp, self.boss.exposed_rounds,
                      tuple(sorted(self.boss.broken_locks)), self.boss.rail_axis,
                      self.boss.rail_target, self.boss.charge, self.boss.shots,
                      self.boss.blasts,
                      self.boss.blast_target, self.boss.blast_kind,
                      tuple(sorted(self.boss.blast_cells)),
                      self.boss.charge_target) if isinstance(self.boss, SiegeLeviathan) else
                     ("null", self.boss.position, self.boss.hp, self.boss.exposed_rounds,
                      self.boss.node_index, self.boss.blocked_kind,
                      tuple(sorted(self.boss.erase_targets)), self.boss.erase_countdown,
                      self.boss.cycles, self.boss.warp_target,
                      self.boss.fracture_axis,
                      tuple(sorted(self.boss.fracture_cells))) if isinstance(self.boss, NullWeaver) else
                     ("apex", self.boss.position, self.boss.hp, self.boss.exposed_rounds,
                      self.boss.seals, self.boss.kind, self.boss.target, self.boss.countdown,
                      tuple(sorted(self.boss.danger)), tuple(sorted(self.boss.barrage_cells)),
                      self.boss.barrage_phase, self.boss.barrage_volley,
                      tuple(sorted(self.boss.fired_barrage_cells)), self.boss.fired_barrage_volley,
                      self.boss.gate, self.boss.gate_broken,
                       self.boss.finale_cycles, self.boss.appeal,
                       self.boss.appeal_ready) if self.boss else None),
            "reflectors": tuple(sorted(self.reflectors)),
            "coolant_valves": tuple(sorted(self.coolant_valves)),
            "grounding_pylons": tuple(sorted(self.grounding_pylons)),
            "relay_pads": tuple(sorted(self.relay_pads)),
            "time_anchors": tuple(sorted(self.time_anchors)),
            "gravity_nodes": tuple(sorted(self.gravity_nodes)),
            "root_plates": tuple(sorted(self.root_plates)),
            "mirror_locks": tuple(sorted(self.mirror_locks)),
            "rail_locks": tuple(sorted(self.rail_locks)),
            "rail_covers": tuple(sorted(self.rail_covers.items())),
            "rail_rebuilds": tuple(sorted(self.rail_rebuilds.items())),
            "siege_pits": tuple(sorted(self.siege_pits.items())),
            "null_nodes": self.null_nodes,
            "null_void": tuple(sorted(self.null_void)),
            "iron_spores": tuple(sorted(self.iron_spores.items())),
            "apex_fast_volley_resolved": self.apex_fast_volley_resolved,
            "apex_seals": self.apex_seals,
            "apex_cage": tuple(sorted(self.apex_cage)),
            "vine_seeds": tuple(sorted(self.vine_seeds.items())),
            "vine_walls": tuple(sorted(self.vine_walls)),
            "enemy_intents": tuple((enemy.enemy_type.value, enemy.position, enemy.hp,
                                    enemy.intent.kind.value if enemy.intent else None,
                                    enemy.intent.direction if enemy.intent else None,
                                    enemy.intent.countdown if enemy.intent else 0,
                                    enemy.intent.power if enemy.intent else 0,
                                    enemy.stunned, enemy.summoned_by,
                                    enemy.intent.path if enemy.intent else ()) for enemy in self.enemies),
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
