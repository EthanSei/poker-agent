"""Tests for pot and side-pot calculation."""

import pytest

from poker.engine.pot import Pot, award_pots, calculate_pots


class TestCalculatePots:
    def test_single_pot_no_allin(self) -> None:
        bets = {0: 100, 1: 100, 2: 100}
        pots = calculate_pots(bets, set())
        assert pots == [Pot(amount=300, eligible_seats=frozenset({0, 1, 2}))]

    def test_one_allin_short(self) -> None:
        # A=50 allin, B=100, C=100
        bets = {0: 50, 1: 100, 2: 100}
        pots = calculate_pots(bets, set())
        assert pots == [
            Pot(amount=150, eligible_seats=frozenset({0, 1, 2})),
            Pot(amount=100, eligible_seats=frozenset({1, 2})),
        ]

    def test_three_way_allin(self) -> None:
        # A=100, B=300, C=500
        bets = {0: 100, 1: 300, 2: 500}
        pots = calculate_pots(bets, set())
        assert pots == [
            Pot(amount=300, eligible_seats=frozenset({0, 1, 2})),  # 100 * 3
            Pot(amount=400, eligible_seats=frozenset({1, 2})),  # 200 * 2
            Pot(amount=200, eligible_seats=frozenset({2})),  # 200 * 1
        ]

    def test_allin_with_fold(self) -> None:
        # A=100, B=200(folds), C=200
        bets = {0: 100, 1: 200, 2: 200}
        folded = {1}
        pots = calculate_pots(bets, folded)
        # Main pot: 100*3=300, eligible {0, 2} (B folded)
        # Side pot: 100*2=200, eligible {2} (B folded)
        assert pots == [
            Pot(amount=300, eligible_seats=frozenset({0, 2})),
            Pot(amount=200, eligible_seats=frozenset({2})),
        ]

    def test_everyone_equal(self) -> None:
        bets = {0: 50, 1: 50, 2: 50, 3: 50}
        pots = calculate_pots(bets, set())
        assert pots == [Pot(amount=200, eligible_seats=frozenset({0, 1, 2, 3}))]

    def test_single_player_remaining(self) -> None:
        # A=100, B=100(folds), C=100(folds)
        bets = {0: 100, 1: 100, 2: 100}
        folded = {1, 2}
        pots = calculate_pots(bets, folded)
        assert pots == [Pot(amount=300, eligible_seats=frozenset({0}))]

    def test_empty_bets(self) -> None:
        assert calculate_pots({}, set()) == []

    def test_all_fold_except_short_stack(self) -> None:
        # A=50, B=100(folds), C=100(folds)
        # B and C's extra 50 each should roll into A's pot
        bets = {0: 50, 1: 100, 2: 100}
        folded = {1, 2}
        pots = calculate_pots(bets, folded)
        assert len(pots) == 1
        assert pots[0].eligible_seats == frozenset({0})
        assert pots[0].amount == 250  # all chips go to sole eligible player

    def test_chip_conservation(self) -> None:
        """Total pot amounts must equal total bets for various scenarios."""
        scenarios = [
            ({0: 100, 1: 100, 2: 100}, set()),
            ({0: 50, 1: 100, 2: 100}, set()),
            ({0: 100, 1: 300, 2: 500}, set()),
            ({0: 100, 1: 200, 2: 200}, {1}),
            ({0: 50, 1: 100, 2: 100}, {1, 2}),
            ({0: 100, 1: 100, 2: 100}, {1, 2}),
        ]
        for bets, folded in scenarios:
            pots = calculate_pots(bets, folded)
            total_pots = sum(p.amount for p in pots)
            total_bets = sum(bets.values())
            assert total_pots == total_bets, (
                f"Chip leak: bets={bets}, folded={folded}, "
                f"total_pots={total_pots}, total_bets={total_bets}"
            )


class TestAwardPots:
    def test_single_winner(self) -> None:
        pots = [Pot(amount=300, eligible_seats=frozenset({0, 1, 2}))]
        rankings = {0: (5,), 1: (3,), 2: (1,)}
        assert award_pots(pots, rankings) == {0: 300}

    def test_split_pot(self) -> None:
        pots = [Pot(amount=200, eligible_seats=frozenset({0, 1}))]
        rankings = {0: (5,), 1: (5,)}
        assert award_pots(pots, rankings) == {0: 100, 1: 100}

    def test_split_odd_chip(self) -> None:
        # 3 players tie for pot of 100 → 34 to lowest seat, 33 each to others
        pots = [Pot(amount=100, eligible_seats=frozenset({0, 1, 2}))]
        rankings = {0: (5,), 1: (5,), 2: (5,)}
        result = award_pots(pots, rankings)
        assert result == {0: 34, 1: 33, 2: 33}

    def test_multi_pot_different_winners(self) -> None:
        # Short stack wins main pot, big stack wins side pot
        pots = [
            Pot(amount=150, eligible_seats=frozenset({0, 1, 2})),
            Pot(amount=100, eligible_seats=frozenset({1, 2})),
        ]
        rankings = {0: (10,), 1: (5,), 2: (3,)}
        result = award_pots(pots, rankings)
        # seat 0 wins main (150), seat 1 wins side (100)
        assert result == {0: 150, 1: 100}

    def test_winner_not_in_side_pot(self) -> None:
        # Best hand is short-stack all-in, they win main but not side
        pots = [
            Pot(amount=300, eligible_seats=frozenset({0, 1, 2})),
            Pot(amount=400, eligible_seats=frozenset({1, 2})),
        ]
        rankings = {0: (10,), 1: (7,), 2: (5,)}
        result = award_pots(pots, rankings)
        # seat 0 wins main (300), seat 1 wins side (400)
        assert result == {0: 300, 1: 400}

    def test_award_conservation(self) -> None:
        """Total winnings must equal total pot amounts."""
        pots = [
            Pot(amount=150, eligible_seats=frozenset({0, 1, 2})),
            Pot(amount=100, eligible_seats=frozenset({1, 2})),
        ]
        rankings = {0: (10,), 1: (5,), 2: (3,)}
        result = award_pots(pots, rankings)
        assert sum(result.values()) == sum(p.amount for p in pots)

    def test_award_missing_rankings_raises(self) -> None:
        pots = [Pot(amount=100, eligible_seats=frozenset({0, 1}))]
        with pytest.raises(ValueError, match="none in hand_rankings"):
            award_pots(pots, {2: (5,)})
