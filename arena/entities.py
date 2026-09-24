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
    SHOOT_BOW_N = "shoot_bow_n"
    SHOOT_BOW_S = "shoot_bow_s"
    SHOOT_BOW_W = "shoot_bow_w"
    SHOOT_BOW_E = "shoot_bow_e"
    SHOOT_PISTOL_N = "shoot_pistol_n"
    SHOOT_PISTOL_S = "shoot_pistol_s"
    SHOOT_PISTOL_W = "shoot_pistol_w"
    SHOOT_PISTOL_E = "shoot_pistol_e"
    EMP = "emp"
    HEAL = "heal"
    WAIT = "wait"


class EnemyType(StrEnum):
    CHASER = "chaser"
    CHARGER = "charger"
    ARCHER = "archer"
    BOMBER = "bomber"
    RAZOR_HOUND = "razor_hound"


ENEMY_HP = {
    EnemyType.CHASER: 30,
    EnemyType.CHARGER: 45,
    EnemyType.ARCHER: 20,
    EnemyType.BOMBER: 24,
    EnemyType.RAZOR_HOUND: 18,
}
ENEMY_MELEE_BONUS = {
    EnemyType.CHASER: 0,
    EnemyType.CHARGER: 3,
    EnemyType.ARCHER: -1,
    EnemyType.BOMBER: 1,
    EnemyType.RAZOR_HOUND: 0,
}
ENEMY_MOVE_DELAY = {
    EnemyType.CHASER: 0,
    EnemyType.CHARGER: 1,
    EnemyType.ARCHER: 2,
    EnemyType.BOMBER: 2,
    EnemyType.RAZOR_HOUND: 0,
}
ENEMY_SPEED_LEVEL = {
    EnemyType.CHASER: 6,
    EnemyType.CHARGER: 12,
    EnemyType.ARCHER: 18,
    EnemyType.BOMBER: 24,
    EnemyType.RAZOR_HOUND: 5,
}


class IntentType(StrEnum):
    MOVE = "move"
    SPRINT = "sprint"
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
class PlayerLoadout:
    bow: bool = False
    pistol: bool = False
    arrows: int = 0
    energy: int = 0


@dataclass
class Player:
    position: tuple[int, int]
    hp: int = 100
    medkits: int = 0
    invulnerable: bool = False
    cooldowns: dict[str, int] = field(default_factory=lambda: {"dash": 0, "emp": 0})
    loadout: PlayerLoadout = field(default_factory=PlayerLoadout)


@dataclass
class Intent:
    kind: IntentType
    direction: str | None = None
    countdown: int = 1
    power: int = 0
    path: tuple[tuple[int, int], ...] = ()


@dataclass
class Enemy:
    position: tuple[int, int]
    hp: int | None = None
    enemy_type: EnemyType = EnemyType.CHASER
    intent: Intent | None = None
    stunned: int = 0
    summoned_by: str | None = None
    max_hp: int = field(init=False)

    def __post_init__(self) -> None:
        base_hp = ENEMY_HP[self.enemy_type]
        self.hp = base_hp if self.hp is None else self.hp
        self.max_hp = max(base_hp, self.hp)
