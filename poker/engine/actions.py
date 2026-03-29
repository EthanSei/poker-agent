"""Player actions for the poker engine."""

from dataclasses import dataclass
from enum import Enum


class Action(Enum):
    FOLD = "fold"
    CHECK = "check"
    CALL = "call"
    BET = "bet"
    RAISE = "raise"
    ALL_IN = "all_in"


@dataclass(frozen=True)
class PlayerAction:
    seat: int
    action: Action
    amount: int = 0
