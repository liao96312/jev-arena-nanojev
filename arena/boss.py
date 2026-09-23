from dataclasses import dataclass, field


@dataclass
class PrismWarden:
    position: tuple[int, int] = (10, 8)
    hp: int = 60
    max_hp: int = 60
    reflections: int = 0
    exposed_rounds: int = 0
    target: tuple[int, int] | None = None
    lunge_target: tuple[int, int] | None = None
    returning: bool = False
    lunge_used: bool = False
    beam_damage: int = 14
    lunge_damage: int = 24


@dataclass
class FurnaceHydra:
    position: tuple[int, int] = (10, 7)
    hp: int = 72
    max_hp: int = 72
    exposed_rounds: int = 0
    valves_opened: set[int] = field(default_factory=set)
    target: tuple[int, int] | None = None
    attack_kind: str = "wave"
    head_x: int = 10
    attacks: int = 0
    wave_damage: int = 22
    fireball_damage: int = 16

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
