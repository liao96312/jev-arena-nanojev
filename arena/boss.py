from dataclasses import dataclass, field


@dataclass
class PrismWarden:
    position: tuple[int, int] = (10, 8)
    hp: int = 168
    max_hp: int = 168
    reflections: int = 0
    used_reflectors: set[tuple[int, int]] = field(default_factory=set)
    reflector_cooldowns: dict[tuple[int, int], int] = field(default_factory=dict)
    reflector_lockout: int = 0
    exposed_rounds: int = 0
    target: tuple[int, int] | None = None
    lunge_target: tuple[int, int] | None = None
    returning: bool = False
    lunge_used: bool = False
    shots_fired: int = 0
    sweep: bool = False
    beam_damage: int = 14
    lunge_damage: int = 24
    spin_step: int = 0


@dataclass
class FurnaceHydra:
    position: tuple[int, int] = (10, 7)
    hp: int = 120
    max_hp: int = 120
    exposed_rounds: int = 0
    valves_opened: set[int] = field(default_factory=set)
    valve_heat: dict[int, int] = field(default_factory=dict)
    target: tuple[int, int] | None = None
    attack_kind: str = "wave"
    head_x: int = 10
    attacks: int = 0
    summons: int = 0
    wave_damage: int = 22
    fireball_damage: int = 16
    wave_columns: tuple[int, ...] = ()
    wave_rapid: bool = False
    combo_queued: bool = False
    non_wave_streak: int = 0


@dataclass
class StormChoir:
    position: tuple[int, int] = (10, 5)
    hp: int = 156
    max_hp: int = 156
    exposed_rounds: int = 0
    target: tuple[int, int] | None = None
    attack_kind: str = "chain"
    attacks: int = 0
    next_overdrive_attack: int = 2
    arc_damage: int = 14
    surge_damage: int = 26
    net_target: tuple[int, int] | None = None
    net_damage: int = 12
    last_ground_pad: tuple[int, int] | None = None
    overdrive_rounds: int = 0
    fatigue_rounds: int = 0
    overdrive_damage: int = 8


@dataclass
class ChronoMantis:
    position: tuple[int, int] = (10, 7)
    hp: int = 168
    max_hp: int = 168
    exposed_rounds: int = 0
    phase: str = "flank"
    slash_target: tuple[int, int] | None = None
    leap_target: tuple[int, int] | None = None
    primed_anchor: tuple[int, int] | None = None
    retreat_target: tuple[int, int] | None = None
    leap_countdown: int = 0
    moves: int = 0
    slash_damage: int = 24
    echo_damage: int = 18
    leap_damage: int = 30
    slow_rounds: int = 0
    fatigue_rounds: int = 0
    anchor_cooldowns: dict[tuple[int, int], int] = field(default_factory=dict)


@dataclass
class VoidAngler:
    position: tuple[int, int] = (10, 6)
    hp: int = 132
    max_hp: int = 132
    exposed_rounds: int = 0
    drained_nodes: set[tuple[int, int]] = field(default_factory=set)
    target: tuple[int, int] | None = None
    attack_kind: str = "mine"
    attacks: int = 0
    beam_damage: int = 22
    hook_damage: int = 18
    pulse_damage: int = 8
    node_aftershock: dict[tuple[int, int], int] = field(default_factory=dict)


@dataclass
class IronGardener:
    position: tuple[int, int] = (10, 6)
    hp: int = 120
    max_hp: int = 120
    exposed_rounds: int = 0
    refluxed_roots: set[int] = field(default_factory=set)
    target: tuple[int, int] | None = None
    attack_kind: str = "flame"
    attacks: int = 0
    summons: int = 0
    flame_damage: int = 22
    thorn_damage: int = 18
    bloom_damage: int = 16


@dataclass
class MirrorSeraph:
    position: tuple[int, int] = (10, 6)
    hp: int = 192
    max_hp: int = 192
    exposed_rounds: int = 0
    broken_locks: set[tuple[int, int]] = field(default_factory=set)
    copied_action: str | None = None
    mirrored_direction: str | None = None
    target: tuple[int, int] | None = None
    echo_target: tuple[int, int] | None = None
    shard_damage: int = 18
    dash_damage: int = 26
    echo_damage: int = 14
    silence_rounds: int = 0
    evade_ready: bool = False
    evaded: bool = False
    emp_jammed: bool = False
    clones_spawned: bool = False
    copies: int = 0
    rushes: int = 0
    rush_target: tuple[int, int] | None = None
    rush_damage: int = 18


@dataclass
class SiegeLeviathan:
    position: tuple[int, int] = (10, 5)
    hp: int = 168
    max_hp: int = 168
    exposed_rounds: int = 0
    broken_locks: set[tuple[int, int]] = field(default_factory=set)
    rail_axis: str = "v"
    rail_target: int | None = None
    charge: int = 0
    shots: int = 0
    blasts: int = 0
    rail_damage: int = 36
    blast_target: tuple[int, int] | None = None
    blast_kind: str = "cross"
    blast_cells: set[tuple[int, int]] = field(default_factory=set)
    charge_target: tuple[int, int] | None = None


@dataclass
class NullWeaver:
    position: tuple[int, int] = (10, 5)
    hp: int = 184
    max_hp: int = 184
    exposed_rounds: int = 0
    node_index: int = 0
    blocked_kind: str | None = None
    erase_targets: set[tuple[int, int]] = field(default_factory=set)
    erase_countdown: int = 0
    cycles: int = 0
    fracture_damage: int = 20
    warp_target: tuple[int, int] | None = None
    fracture_axis: str = "h"
    fracture_cells: set[tuple[int, int]] = field(default_factory=set)


@dataclass
class ApexArbiter:
    position: tuple[int, int] = (10, 5)
    hp: int = 216
    max_hp: int = 216
    exposed_rounds: int = 0
    seals: int = 0
    kind: str = "cage"
    countdown: int = 0
    target: tuple[int, int] | None = None
    danger: set[tuple[int, int]] = field(default_factory=set)
    barrage_cells: set[tuple[int, int]] = field(default_factory=set)
    barrage_phase: int = 0
    barrage_volley: int = 0
    fired_barrage_cells: set[tuple[int, int]] = field(default_factory=set)
    fired_barrage_volley: int = 0
    gate: tuple[int, int] | None = None
    gate_broken: bool = False
    finale_cycles: int = 0
    appeal: tuple[int, int] | None = None
    appeal_ready: bool = False

def ray_cells(start: tuple[int, int], end: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    """The same grid trace is used for the warning and the actual shot."""
    x, y = start
    dx, dy = abs(end[0] - x), abs(end[1] - y)
    sx, sy = (1 if x < end[0] else -1), (1 if y < end[1] else -1)
    error = dx - dy
    cells = []
    while (x, y) != end:
        twice = 2 * error
        if twice > -dy:
            error -= dy
            x += sx
        if twice < dx:
            error += dx
            y += sy
        cells.append((x, y))
    return tuple(cells)
