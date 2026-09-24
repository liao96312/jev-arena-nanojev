from dataclasses import dataclass, field


@dataclass
class PrismWarden:
    position: tuple[int, int] = (10, 8)
    hp: int = 168
    max_hp: int = 168
    reflections: int = 0
    used_reflectors: set[tuple[int, int]] = field(default_factory=set)
    exposed_rounds: int = 0
    target: tuple[int, int] | None = None
    lunge_target: tuple[int, int] | None = None
    returning: bool = False
    lunge_used: bool = False
    shots_fired: int = 0
    sweep: bool = False
    beam_damage: int = 14
    lunge_damage: int = 24


@dataclass
class FurnaceHydra:
    position: tuple[int, int] = (10, 7)
    hp: int = 105
    max_hp: int = 105
    exposed_rounds: int = 0
    valves_opened: set[int] = field(default_factory=set)
    target: tuple[int, int] | None = None
    attack_kind: str = "wave"
    head_x: int = 10
    attacks: int = 0
    wave_damage: int = 22
    fireball_damage: int = 16


@dataclass
class StormChoir:
    position: tuple[int, int] = (10, 5)
    hp: int = 132
    max_hp: int = 132
    exposed_rounds: int = 0
    target: tuple[int, int] | None = None
    attack_kind: str = "chain"
    attacks: int = 0
    arc_damage: int = 14
    surge_damage: int = 26


@dataclass
class ChronoMantis:
    position: tuple[int, int] = (10, 7)
    hp: int = 168
    max_hp: int = 168
    exposed_rounds: int = 0
    phase: str = "flank"
    slash_target: tuple[int, int] | None = None
    leap_target: tuple[int, int] | None = None
    leap_countdown: int = 0
    moves: int = 0
    slash_damage: int = 24
    echo_damage: int = 18


@dataclass
class VoidAngler:
    position: tuple[int, int] = (10, 6)
    hp: int = 108
    max_hp: int = 108
    exposed_rounds: int = 0
    drained_nodes: set[tuple[int, int]] = field(default_factory=set)
    target: tuple[int, int] | None = None
    attack_kind: str = "mine"
    attacks: int = 0
    beam_damage: int = 22

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
