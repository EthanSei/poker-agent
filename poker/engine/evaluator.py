from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from enum import IntEnum
from itertools import combinations

from poker.engine.cards import Card, Rank


class HandCategory(IntEnum):
    HIGH_CARD = 0
    PAIR = 1
    TWO_PAIR = 2
    THREE_OF_A_KIND = 3
    STRAIGHT = 4
    FLUSH = 5
    FULL_HOUSE = 6
    FOUR_OF_A_KIND = 7
    STRAIGHT_FLUSH = 8


@dataclass(frozen=True, order=True)
class HandResult:
    category: HandCategory
    tiebreakers: tuple[int, ...]


def evaluate_hand(cards: Sequence[Card]) -> HandResult:
    """Evaluate best 5-card hand from 5-7 cards."""
    if len(cards) < 5 or len(cards) > 7:
        raise ValueError(f"Need 5-7 cards, got {len(cards)}")
    if len(cards) == 5:
        return _evaluate_five(cards)
    best: HandResult | None = None
    for combo in combinations(cards, 5):
        result = _evaluate_five(combo)
        if best is None or result > best:
            best = result
    assert best is not None
    return best


def _is_flush(cards: Sequence[Card]) -> bool:
    return len({c.suit for c in cards}) == 1


def _is_straight(ranks_desc: list[int]) -> tuple[bool, int]:
    """Check if sorted-descending ranks form a straight. Returns (is_straight, high_card)."""
    # Normal straight check
    if ranks_desc[0] - ranks_desc[4] == 4 and len(set(ranks_desc)) == 5:
        return True, ranks_desc[0]
    # Wheel: A-2-3-4-5
    if ranks_desc == [Rank.ACE, Rank.FIVE, Rank.FOUR, Rank.THREE, Rank.TWO]:
        return True, Rank.FIVE
    return False, 0


def _evaluate_five(cards: Sequence[Card]) -> HandResult:
    ranks = sorted((c.rank.value for c in cards), reverse=True)
    flush = _is_flush(cards)
    straight, straight_high = _is_straight(ranks)

    if straight and flush:
        return HandResult(HandCategory.STRAIGHT_FLUSH, (straight_high,))

    freq = Counter(ranks)
    counts = sorted(freq.values(), reverse=True)

    if counts == [4, 1]:
        quads_rank = _rank_with_count(freq, 4)
        kicker = _rank_with_count(freq, 1)
        return HandResult(HandCategory.FOUR_OF_A_KIND, (quads_rank, kicker))

    if counts == [3, 2]:
        trips_rank = _rank_with_count(freq, 3)
        pair_rank = _rank_with_count(freq, 2)
        return HandResult(HandCategory.FULL_HOUSE, (trips_rank, pair_rank))

    if flush:
        return HandResult(HandCategory.FLUSH, tuple(ranks))

    if straight:
        return HandResult(HandCategory.STRAIGHT, (straight_high,))

    if counts == [3, 1, 1]:
        trips_rank = _rank_with_count(freq, 3)
        kickers = sorted((r for r, c in freq.items() if c == 1), reverse=True)
        return HandResult(HandCategory.THREE_OF_A_KIND, (trips_rank, *kickers))

    if counts == [2, 2, 1]:
        pairs = sorted((r for r, c in freq.items() if c == 2), reverse=True)
        kicker = _rank_with_count(freq, 1)
        return HandResult(HandCategory.TWO_PAIR, (pairs[0], pairs[1], kicker))

    if counts == [2, 1, 1, 1]:
        pair_rank = _rank_with_count(freq, 2)
        kickers = sorted((r for r, c in freq.items() if c == 1), reverse=True)
        return HandResult(HandCategory.PAIR, (pair_rank, *kickers))

    return HandResult(HandCategory.HIGH_CARD, tuple(ranks))


def _rank_with_count(freq: Counter[int], count: int) -> int:
    """Return the rank that appears exactly `count` times."""
    for rank, c in freq.items():
        if c == count:
            return rank
    raise ValueError(f"No rank with count {count}")  # pragma: no cover
