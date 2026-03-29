from dataclasses import dataclass
from enum import Enum, IntEnum

import secrets


class Rank(IntEnum):
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    NINE = 9
    TEN = 10
    JACK = 11
    QUEEN = 12
    KING = 13
    ACE = 14


class Suit(Enum):
    HEARTS = "H"
    DIAMONDS = "D"
    CLUBS = "C"
    SPADES = "S"


_RANK_TO_STR: dict[int, str] = {10: "T", 11: "J", 12: "Q", 13: "K", 14: "A"}
_STR_TO_RANK: dict[str, int] = {"T": 10, "J": 11, "Q": 12, "K": 13, "A": 14}


@dataclass(frozen=True)
class Card:
    rank: Rank
    suit: Suit

    def __str__(self) -> str:
        rank_str = _RANK_TO_STR.get(self.rank, str(self.rank.value))
        return f"{rank_str}{self.suit.value}"

    def __repr__(self) -> str:
        return f"Card({self!s})"

    @classmethod
    def from_str(cls, s: str) -> "Card":
        """Parse 'As', 'Td', '2c' etc."""
        if len(s) != 2:
            raise ValueError(f"Invalid card string: {s!r}")
        rank_char = s[0]
        suit_char = s[1]
        try:
            rank_val = _STR_TO_RANK.get(rank_char) or int(rank_char)
            return cls(rank=Rank(rank_val), suit=Suit(suit_char.upper()))
        except (ValueError, KeyError) as e:
            raise ValueError(f"Invalid card string: {s!r}") from e


class Deck:
    """Standard 52-card deck with cryptographic shuffling."""

    def __init__(self) -> None:
        self._cards: list[Card] = [Card(rank, suit) for suit in Suit for rank in Rank]
        self._index: int = 0
        self.shuffle()

    def shuffle(self) -> None:
        """Shuffle using CSPRNG (secrets module)."""
        rng = secrets.SystemRandom()
        rng.shuffle(self._cards)
        self._index = 0

    def deal(self, n: int = 1) -> list[Card]:
        """Deal n cards from top of deck. Raises if not enough cards."""
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")
        if self._index + n > len(self._cards):
            raise ValueError(f"Not enough cards: {self.remaining} remaining, {n} requested")
        cards = self._cards[self._index : self._index + n]
        self._index += n
        return cards

    @property
    def remaining(self) -> int:
        return len(self._cards) - self._index

    def reset(self) -> None:
        """Reset and reshuffle the deck."""
        self._index = 0
        self.shuffle()
