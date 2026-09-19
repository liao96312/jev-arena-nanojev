from dataclasses import dataclass
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
    HEAL = "heal"
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


@dataclass
class Enemy:
    position: tuple[int, int]
    hp: int = 30
