import pytest

from poker.engine.cards import Card, Deck, Rank, Suit


class TestRank:
    def test_ordering(self) -> None:
        assert Rank.TWO < Rank.ACE
        assert Rank.JACK < Rank.QUEEN < Rank.KING < Rank.ACE
        assert Rank.NINE < Rank.TEN

    def test_values(self) -> None:
        assert Rank.TWO == 2
        assert Rank.ACE == 14


class TestCard:
    def test_creation_and_equality(self) -> None:
        c1 = Card(Rank.ACE, Suit.SPADES)
        c2 = Card(Rank.ACE, Suit.SPADES)
        assert c1 == c2

    def test_frozen(self) -> None:
        card = Card(Rank.ACE, Suit.SPADES)
        with pytest.raises(AttributeError):
            card.rank = Rank.KING  # type: ignore[misc]

    def test_str_face_cards(self) -> None:
        assert str(Card(Rank.ACE, Suit.SPADES)) == "AS"
        assert str(Card(Rank.KING, Suit.HEARTS)) == "KH"
        assert str(Card(Rank.QUEEN, Suit.DIAMONDS)) == "QD"
        assert str(Card(Rank.JACK, Suit.CLUBS)) == "JC"
        assert str(Card(Rank.TEN, Suit.HEARTS)) == "TH"

    def test_str_number_cards(self) -> None:
        assert str(Card(Rank.TWO, Suit.CLUBS)) == "2C"
        assert str(Card(Rank.NINE, Suit.SPADES)) == "9S"
        assert str(Card(Rank.FIVE, Suit.DIAMONDS)) == "5D"

    def test_repr(self) -> None:
        card = Card(Rank.ACE, Suit.SPADES)
        assert repr(card) == "Card(AS)"

    def test_from_str_round_trip(self) -> None:
        cases = ["AS", "KH", "QD", "JC", "TH", "9S", "2C", "5D", "7H"]
        for s in cases:
            assert str(Card.from_str(s)) == s

    def test_from_str_case_insensitive_suit(self) -> None:
        assert Card.from_str("As") == Card(Rank.ACE, Suit.SPADES)
        assert Card.from_str("2c") == Card(Rank.TWO, Suit.CLUBS)

    def test_from_str_invalid(self) -> None:
        with pytest.raises(ValueError):
            Card.from_str("Xx")
        with pytest.raises(ValueError):
            Card.from_str("1s")
        with pytest.raises(ValueError):
            Card.from_str("")
        with pytest.raises(ValueError):
            Card.from_str("Ace")

    def test_hashable(self) -> None:
        card_set = {Card(Rank.ACE, Suit.SPADES), Card(Rank.ACE, Suit.SPADES)}
        assert len(card_set) == 1


class TestDeck:
    def test_initial_size(self) -> None:
        deck = Deck()
        assert deck.remaining == 52

    def test_deal_reduces_remaining(self) -> None:
        deck = Deck()
        deck.deal(5)
        assert deck.remaining == 47

    def test_deal_too_many_raises(self) -> None:
        deck = Deck()
        deck.deal(50)
        with pytest.raises(ValueError, match="Not enough cards"):
            deck.deal(5)

    def test_all_cards_unique(self) -> None:
        deck = Deck()
        cards = deck.deal(52)
        assert len(set(cards)) == 52

    def test_shuffle_changes_order(self) -> None:
        """Shuffle should produce a different order. Run multiple trials to avoid flakiness."""
        deck = Deck()
        orders: list[list[Card]] = []
        for _ in range(5):
            deck.reset()
            orders.append(deck.deal(52))
        # At least two of the five orderings should differ
        unique_orders = {tuple(o) for o in orders}
        assert len(unique_orders) > 1

    def test_reset_restores_full_deck(self) -> None:
        deck = Deck()
        deck.deal(30)
        assert deck.remaining == 22
        deck.reset()
        assert deck.remaining == 52
