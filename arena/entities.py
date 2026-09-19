from dataclasses import dataclass, field
from enum import StrEnum


class Action(StrEnum):
    MOVE_N = "move_n"
    MOVE_S = "move_s"
    MOVE_W = "move_w"
    MOVE_E = "move_e"
    ATTACK_N = "attack_n"
    ATTACK_S = "attack_s"
    ATTACK_W = "attack_w"
    ATTACK_E = "attack_e"
    SHOVE_N = "shove_n"
    SHOVE_S = "shove_s"
    SHOVE_W = "shove_w"
    SHOVE_E = "shove_e"
    DASH_N = "dash_n"
    DASH_S = "dash_s"
    DASH_W = "dash_w"
    DASH_E = "dash_e"
    HEAL = "heal"
    WAIT = "wait"


class EnemyType(StrEnum):
    CHASER = "chaser"
    CHARGER = "charger"
    ARCHER = "archer"
    BOMBER = "bomber"


class IntentType(StrEnum):
    MOVE = "move"
    MELEE = "melee"
    CHARGE = "charge"
    SHOOT = "shoot"
    EXPLODE = "explode"
    WAIT = "wait"


DIRECTIONS = {
    "n": (0, -1),
    "s": (0, 1),
    "w": (-1, 0),
    "e": (1, 0),
}


@dataclass
class Player:
    position: tuple[int, int]
    hp: int = 100
    medkits: int = 0
    cooldowns: dict[str, int] = field(default_factory=lambda: {"dash": 0})


@dataclass
class Intent:
    kind: IntentType
    direction: str | None = None
    countdown: int = 1
    power: int = 0


@dataclass
class Enemy:
    position: tuple[int, int]
    hp: int = 30
    enemy_type: EnemyType = EnemyType.CHASER
    intent: Intent | None = None
    stunned: int = 0
