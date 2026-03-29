import json
from pathlib import Path

import pytest

from poker.engine.cards import Card
from poker.engine.evaluator import HandCategory, HandResult, evaluate_hand

FIXTURES = json.loads((Path(__file__).parent / "fixtures/hand_eval_cases.json").read_text())


class TestEvaluateHand:
    def test_high_card(self) -> None:
        cards = [Card.from_str(s) for s in ["AS", "KH", "QD", "JC", "9S"]]
        result = evaluate_hand(cards)
        assert result.category == HandCategory.HIGH_CARD
        assert result.tiebreakers == (14, 13, 12, 11, 9)

    def test_pair(self) -> None:
        cards = [Card.from_str(s) for s in ["AS", "AH", "KD", "QC", "JS"]]
        result = evaluate_hand(cards)
        assert result.category == HandCategory.PAIR
        assert result.tiebreakers == (14, 13, 12, 11)

    def test_two_pair(self) -> None:
        cards = [Card.from_str(s) for s in ["AS", "AH", "KD", "KC", "QS"]]
        result = evaluate_hand(cards)
        assert result.category == HandCategory.TWO_PAIR
        assert result.tiebreakers == (14, 13, 12)

    def test_three_of_a_kind(self) -> None:
        cards = [Card.from_str(s) for s in ["AS", "AH", "AD", "KC", "QS"]]
        result = evaluate_hand(cards)
        assert result.category == HandCategory.THREE_OF_A_KIND
        assert result.tiebreakers == (14, 13, 12)

    def test_straight(self) -> None:
        cards = [Card.from_str(s) for s in ["5S", "6H", "7D", "8C", "9S"]]
        result = evaluate_hand(cards)
        assert result.category == HandCategory.STRAIGHT
        assert result.tiebreakers == (9,)

    def test_flush(self) -> None:
        cards = [Card.from_str(s) for s in ["AS", "KS", "QS", "JS", "9S"]]
        result = evaluate_hand(cards)
        assert result.category == HandCategory.FLUSH
        assert result.tiebreakers == (14, 13, 12, 11, 9)

    def test_full_house(self) -> None:
        cards = [Card.from_str(s) for s in ["AS", "AH", "AD", "KC", "KS"]]
        result = evaluate_hand(cards)
        assert result.category == HandCategory.FULL_HOUSE
        assert result.tiebreakers == (14, 13)

    def test_four_of_a_kind(self) -> None:
        cards = [Card.from_str(s) for s in ["AS", "AH", "AD", "AC", "KS"]]
        result = evaluate_hand(cards)
        assert result.category == HandCategory.FOUR_OF_A_KIND
        assert result.tiebreakers == (14, 13)

    def test_straight_flush(self) -> None:
        cards = [Card.from_str(s) for s in ["5S", "6S", "7S", "8S", "9S"]]
        result = evaluate_hand(cards)
        assert result.category == HandCategory.STRAIGHT_FLUSH
        assert result.tiebreakers == (9,)

    def test_wheel_straight(self) -> None:
        cards = [Card.from_str(s) for s in ["AS", "2H", "3D", "4C", "5S"]]
        result = evaluate_hand(cards)
        assert result.category == HandCategory.STRAIGHT
        assert result.tiebreakers == (5,)

    def test_wheel_straight_flush(self) -> None:
        cards = [Card.from_str(s) for s in ["AH", "2H", "3H", "4H", "5H"]]
        result = evaluate_hand(cards)
        assert result.category == HandCategory.STRAIGHT_FLUSH
        assert result.tiebreakers == (5,)

    def test_broadway_straight(self) -> None:
        cards = [Card.from_str(s) for s in ["TS", "JH", "QD", "KC", "AS"]]
        result = evaluate_hand(cards)
        assert result.category == HandCategory.STRAIGHT
        assert result.tiebreakers == (14,)

    def test_seven_card_best_five(self) -> None:
        # 7 cards containing a flush in spades
        cards = [Card.from_str(s) for s in ["AS", "KS", "QS", "JS", "9S", "2H", "3D"]]
        result = evaluate_hand(cards)
        assert result.category == HandCategory.FLUSH
        assert result.tiebreakers == (14, 13, 12, 11, 9)

    def test_seven_card_finds_straight(self) -> None:
        cards = [Card.from_str(s) for s in ["4S", "5H", "6D", "7C", "8S", "KH", "2D"]]
        result = evaluate_hand(cards)
        assert result.category == HandCategory.STRAIGHT
        assert result.tiebreakers == (8,)

    def test_six_cards(self) -> None:
        cards = [Card.from_str(s) for s in ["AS", "AH", "KD", "QC", "JS", "3H"]]
        result = evaluate_hand(cards)
        assert result.category == HandCategory.PAIR
        assert result.tiebreakers == (14, 13, 12, 11)

    def test_invalid_too_few(self) -> None:
        cards = [Card.from_str(s) for s in ["AS", "AH", "KD", "QC"]]
        with pytest.raises(ValueError, match="Need 5-7 cards"):
            evaluate_hand(cards)

    def test_invalid_too_many(self) -> None:
        cards = [Card.from_str(s) for s in ["AS", "AH", "KD", "QC", "JS", "TS", "9S", "8S"]]
        with pytest.raises(ValueError, match="Need 5-7 cards"):
            evaluate_hand(cards)

    def test_hand_result_ordering(self) -> None:
        pair = HandResult(HandCategory.PAIR, (14, 13, 12, 11))
        two_pair = HandResult(HandCategory.TWO_PAIR, (3, 2, 4))
        assert two_pair > pair


class TestHandComparison:
    @pytest.mark.parametrize("case", FIXTURES, ids=[c["desc"] for c in FIXTURES])
    def test_comparison(self, case: dict[str, object]) -> None:
        assert isinstance(case["hand_a"], list)
        assert isinstance(case["hand_b"], list)
        assert isinstance(case["expected"], str)
        hand_a = [Card.from_str(s) for s in case["hand_a"]]
        hand_b = [Card.from_str(s) for s in case["hand_b"]]
        result_a = evaluate_hand(hand_a)
        result_b = evaluate_hand(hand_b)
        if case["expected"] == "a_wins":
            assert result_a > result_b, f"{result_a} should beat {result_b}"
        elif case["expected"] == "b_wins":
            assert result_a < result_b, f"{result_a} should lose to {result_b}"
        else:
            assert result_a == result_b, f"{result_a} should tie {result_b}"
