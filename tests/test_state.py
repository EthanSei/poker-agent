"""Tests for the game state machine."""

from poker.engine.actions import Action, PlayerAction
from poker.engine.errors import IllegalActionError
from poker.engine.state import (
    GameState,
    Street,
    apply_action,
    legal_actions,
    new_hand,
    validate_state,
)


def _make_hand(
    stacks: dict[int, int] | None = None,
    dealer: int = 0,
    sb: int = 50,
    bb: int = 100,
) -> GameState:
    """Helper to create a standard hand."""
    if stacks is None:
        stacks = {0: 1000, 1: 1000, 2: 1000}
    return new_hand(stacks, dealer_seat=dealer, small_blind=sb, big_blind=bb)


# 1. new_hand creates valid state with blinds posted
class TestNewHand:
    def test_creates_valid_state(self) -> None:
        state = _make_hand()
        assert state.street == Street.PREFLOP
        assert not state.hand_over
        assert len(state.board) == 0
        validate_state(state)

    def test_blinds_posted(self) -> None:
        # 3 players: dealer=0, SB=seat1, BB=seat2
        state = _make_hand(stacks={0: 1000, 1: 1000, 2: 1000}, dealer=0)
        sb_player = next(p for p in state.players if p.seat == 1)
        bb_player = next(p for p in state.players if p.seat == 2)
        assert sb_player.stack == 950
        assert sb_player.current_bet == 50
        assert bb_player.stack == 900
        assert bb_player.current_bet == 100
        assert state.pot == 150

    def test_hole_cards_dealt(self) -> None:
        state = _make_hand()
        for p in state.players:
            assert len(p.hole_cards) == 2

    # 2. Heads-up: dealer posts SB
    def test_heads_up_dealer_posts_sb(self) -> None:
        state = _make_hand(stacks={0: 1000, 1: 1000}, dealer=0)
        dealer_player = next(p for p in state.players if p.seat == 0)
        other_player = next(p for p in state.players if p.seat == 1)
        # Dealer posts SB
        assert dealer_player.stack == 950
        assert dealer_player.current_bet == 50
        # Other posts BB
        assert other_player.stack == 900
        assert other_player.current_bet == 100
        # Preflop: action on dealer (left of BB wraps to dealer in heads-up)
        assert state.current_actor.seat == 0

    def test_at_least_two_players(self) -> None:
        try:
            _make_hand(stacks={0: 1000})
            assert False, "Should have raised"
        except Exception:
            pass


# 3. Full hand: preflop -> flop -> turn -> river -> showdown
class TestFullHand:
    def test_full_hand_all_check(self) -> None:
        state = _make_hand(stacks={0: 1000, 1: 1000, 2: 1000}, dealer=0)
        validate_state(state)

        # Preflop: seat 0 is UTG (left of BB=seat2), calls
        assert state.current_actor.seat == 0
        state = apply_action(state, PlayerAction(seat=0, action=Action.CALL, amount=100))

        # seat 1 (SB) calls
        assert state.current_actor.seat == 1
        state = apply_action(state, PlayerAction(seat=1, action=Action.CALL, amount=50))

        # seat 2 (BB) checks (option)
        assert state.current_actor.seat == 2
        state = apply_action(state, PlayerAction(seat=2, action=Action.CHECK))

        # Now on flop
        assert state.street == Street.FLOP
        assert len(state.board) == 3
        validate_state(state)

        # Flop: all check (postflop starts left of dealer = seat 1)
        assert state.current_actor.seat == 1
        state = apply_action(state, PlayerAction(seat=1, action=Action.CHECK))
        state = apply_action(state, PlayerAction(seat=2, action=Action.CHECK))
        state = apply_action(state, PlayerAction(seat=0, action=Action.CHECK))

        # Turn
        assert state.street == Street.TURN
        assert len(state.board) == 4
        validate_state(state)

        # Turn: all check
        state = apply_action(state, PlayerAction(seat=1, action=Action.CHECK))
        state = apply_action(state, PlayerAction(seat=2, action=Action.CHECK))
        state = apply_action(state, PlayerAction(seat=0, action=Action.CHECK))

        # River
        assert state.street == Street.RIVER
        assert len(state.board) == 5
        validate_state(state)

        # River: all check
        state = apply_action(state, PlayerAction(seat=1, action=Action.CHECK))
        state = apply_action(state, PlayerAction(seat=2, action=Action.CHECK))
        state = apply_action(state, PlayerAction(seat=0, action=Action.CHECK))

        # Showdown complete
        assert state.hand_over
        assert state.street == Street.COMPLETE
        assert len(state.winners) > 0
        validate_state(state)

        # Chips conserved
        total = sum(p.stack for p in state.players) + state.pot
        assert total == 3000


# 4. Fold to win
class TestFoldToWin:
    def test_everyone_folds(self) -> None:
        state = _make_hand(stacks={0: 1000, 1: 1000, 2: 1000}, dealer=0)

        # UTG folds
        state = apply_action(state, PlayerAction(seat=0, action=Action.FOLD))
        # SB folds
        state = apply_action(state, PlayerAction(seat=1, action=Action.FOLD))

        assert state.hand_over
        assert state.street == Street.COMPLETE
        # BB wins the pot (SB 50 + BB 100 = 150) — wait, actually
        # SB folded so BB wins
        assert 2 in state.winners
        assert state.winners[2] == 150

        bb_player = next(p for p in state.players if p.seat == 2)
        assert bb_player.stack == 1050
        validate_state(state)


# 5. All-in and call
class TestAllIn:
    def test_all_in_and_call(self) -> None:
        state = _make_hand(stacks={0: 500, 1: 500}, dealer=0)

        # Dealer (seat 0, SB) goes all-in
        state = apply_action(state, PlayerAction(seat=0, action=Action.ALL_IN, amount=450))

        # Seat 1 (BB) calls
        assert state.current_actor.seat == 1
        state = apply_action(state, PlayerAction(seat=1, action=Action.CALL, amount=400))

        # Hand should complete (deal out remaining board and showdown)
        assert state.hand_over
        assert state.street == Street.COMPLETE
        assert len(state.board) == 5
        total = sum(p.stack for p in state.players)
        assert total == 1000
        validate_state(state)


# 6. Legal actions
class TestLegalActions:
    def test_cant_check_facing_bet(self) -> None:
        state = _make_hand(stacks={0: 1000, 1: 1000, 2: 1000}, dealer=0)
        # UTG faces the BB, can't check
        actions = legal_actions(state)
        action_types = {a.action for a in actions}
        assert Action.CHECK not in action_types
        assert Action.FOLD in action_types
        assert Action.CALL in action_types

    def test_can_check_when_no_bet(self) -> None:
        state = _make_hand(stacks={0: 1000, 1: 1000, 2: 1000}, dealer=0)
        # Get to flop
        state = apply_action(state, PlayerAction(seat=0, action=Action.CALL, amount=100))
        state = apply_action(state, PlayerAction(seat=1, action=Action.CALL, amount=50))
        state = apply_action(state, PlayerAction(seat=2, action=Action.CHECK))
        # Flop: first actor can check
        actions = legal_actions(state)
        action_types = {a.action for a in actions}
        assert Action.CHECK in action_types
        assert Action.FOLD not in action_types

    def test_must_raise_not_bet_facing_bet(self) -> None:
        state = _make_hand(stacks={0: 1000, 1: 1000, 2: 1000}, dealer=0)
        # UTG raises
        state = apply_action(state, PlayerAction(seat=0, action=Action.RAISE, amount=200))
        # SB faces a raise
        actions = legal_actions(state)
        action_types = {a.action for a in actions}
        assert Action.BET not in action_types
        assert Action.RAISE in action_types or Action.ALL_IN in action_types


# 7. Illegal actions
class TestIllegalActions:
    def test_wrong_seat(self) -> None:
        state = _make_hand(stacks={0: 1000, 1: 1000, 2: 1000}, dealer=0)
        try:
            apply_action(state, PlayerAction(seat=1, action=Action.FOLD))
            assert False, "Should have raised"
        except IllegalActionError:
            pass

    def test_invalid_raise_amount(self) -> None:
        state = _make_hand(stacks={0: 1000, 1: 1000, 2: 1000}, dealer=0)
        # Try to raise less than min raise
        try:
            apply_action(state, PlayerAction(seat=0, action=Action.RAISE, amount=110))
            assert False, "Should have raised"
        except IllegalActionError:
            pass


# 8. Min-raise enforcement
class TestMinRaise:
    def test_min_raise_size(self) -> None:
        state = _make_hand(stacks={0: 5000, 1: 5000, 2: 5000}, dealer=0)
        # UTG raises to 200 (raise of 100 over BB of 100)
        state = apply_action(state, PlayerAction(seat=0, action=Action.RAISE, amount=200))
        # SB must raise at least 100 more (to 300 total, so 250 more from SB)
        actions = legal_actions(state)
        raise_actions = [a for a in actions if a.action == Action.RAISE]
        assert len(raise_actions) >= 1
        # Min raise should be to 300 (current_bet 200 + min_raise 100 = 300)
        # SB current_bet is 50, so amount = 300 - 50 = 250
        assert raise_actions[0].amount == 250

    def test_reraise_increases_min(self) -> None:
        state = _make_hand(stacks={0: 5000, 1: 5000, 2: 5000}, dealer=0)
        # UTG raises to 200
        state = apply_action(state, PlayerAction(seat=0, action=Action.RAISE, amount=200))
        # SB raises to 500 (raise of 300 over 200)
        state = apply_action(state, PlayerAction(seat=1, action=Action.RAISE, amount=450))
        # BB must raise at least 300 more (to 800)
        actions = legal_actions(state)
        raise_actions = [a for a in actions if a.action == Action.RAISE]
        assert len(raise_actions) >= 1
        # Min raise to 800, BB current_bet is 100, so amount = 700
        assert raise_actions[0].amount == 700


# 9. Side pot scenario
class TestSidePot:
    def test_short_stack_all_in(self) -> None:
        state = _make_hand(stacks={0: 300, 1: 1000, 2: 1000}, dealer=0)

        # UTG (seat 0, short stack) goes all-in for 300 (full stack, no blind posted)
        state = apply_action(state, PlayerAction(seat=0, action=Action.ALL_IN, amount=300))

        # SB (seat 1, posted 50) calls 300 total = 250 more
        state = apply_action(state, PlayerAction(seat=1, action=Action.CALL, amount=250))

        # BB (seat 2, posted 100) calls 300 total = 200 more
        state = apply_action(state, PlayerAction(seat=2, action=Action.CALL, amount=200))

        # Flop - remaining active non-all-in players act
        assert state.street == Street.FLOP

        # Both check through remaining streets
        for _street in range(3):  # flop, turn, river
            if state.hand_over:
                break
            for _action in range(2):
                if state.hand_over:
                    break
                seat = state.current_actor.seat
                state = apply_action(state, PlayerAction(seat=seat, action=Action.CHECK))

        assert state.hand_over
        total = sum(p.stack for p in state.players)
        assert total == 2300  # total chips conserved
        validate_state(state)


# 10. Chip conservation across hands
class TestChipConservation:
    def test_across_two_hands(self) -> None:
        stacks = {0: 1000, 1: 1000}

        state = _make_hand(stacks=stacks, dealer=0)
        # Quick fold
        state = apply_action(state, PlayerAction(seat=0, action=Action.FOLD))
        assert state.hand_over

        # New hand with updated stacks
        new_stacks = {p.seat: p.stack for p in state.players}
        assert sum(new_stacks.values()) == 2000

        state2 = _make_hand(stacks=new_stacks, dealer=1)
        validate_state(state2)
        state2 = apply_action(state2, PlayerAction(seat=1, action=Action.FOLD))
        assert state2.hand_over

        final_total = sum(p.stack for p in state2.players)
        assert final_total == 2000


# 11. Street advancement: correct board cards
class TestStreetAdvancement:
    def test_board_card_counts(self) -> None:
        state = _make_hand(stacks={0: 1000, 1: 1000}, dealer=0)

        # Preflop
        assert len(state.board) == 0

        # Call to see flop
        state = apply_action(state, PlayerAction(seat=0, action=Action.CALL, amount=50))
        state = apply_action(state, PlayerAction(seat=1, action=Action.CHECK))
        assert state.street == Street.FLOP
        assert len(state.board) == 3

        # Check through flop
        state = apply_action(state, PlayerAction(seat=1, action=Action.CHECK))
        state = apply_action(state, PlayerAction(seat=0, action=Action.CHECK))
        assert state.street == Street.TURN
        assert len(state.board) == 4

        # Check through turn
        state = apply_action(state, PlayerAction(seat=1, action=Action.CHECK))
        state = apply_action(state, PlayerAction(seat=0, action=Action.CHECK))
        assert state.street == Street.RIVER
        assert len(state.board) == 5

        # Check through river
        state = apply_action(state, PlayerAction(seat=1, action=Action.CHECK))
        state = apply_action(state, PlayerAction(seat=0, action=Action.CHECK))
        assert state.hand_over


# 12. BB option
class TestBBOption:
    def test_bb_can_raise_after_limps(self) -> None:
        state = _make_hand(stacks={0: 1000, 1: 1000, 2: 1000}, dealer=0)

        # UTG calls
        state = apply_action(state, PlayerAction(seat=0, action=Action.CALL, amount=100))
        # SB calls
        state = apply_action(state, PlayerAction(seat=1, action=Action.CALL, amount=50))

        # BB should get to act (option)
        assert state.current_actor.seat == 2
        assert state.street == Street.PREFLOP

        # BB can raise
        actions = legal_actions(state)
        action_types = {a.action for a in actions}
        assert Action.CHECK in action_types
        # BB can raise (it shows as BET since current_bet matches their bet)
        # Actually after limps, BB's current_bet == state.current_bet, so they can check
        # They should also be able to raise — which shows as BET since no one raised
        assert Action.BET in action_types or Action.RAISE in action_types

    def test_bb_check_advances_to_flop(self) -> None:
        state = _make_hand(stacks={0: 1000, 1: 1000, 2: 1000}, dealer=0)

        # UTG calls, SB calls, BB checks
        state = apply_action(state, PlayerAction(seat=0, action=Action.CALL, amount=100))
        state = apply_action(state, PlayerAction(seat=1, action=Action.CALL, amount=50))
        state = apply_action(state, PlayerAction(seat=2, action=Action.CHECK))

        assert state.street == Street.FLOP


# Test to_player_view
class TestPlayerView:
    def test_hides_other_hole_cards(self) -> None:
        state = _make_hand(stacks={0: 1000, 1: 1000, 2: 1000}, dealer=0)
        view = state.to_player_view(0)
        players = view["players"]
        assert isinstance(players, list)
        for pv in players:
            assert isinstance(pv, dict)
            if pv["seat"] == 0:
                assert len(pv["hole_cards"]) == 2  # type: ignore[arg-type]
            else:
                assert len(pv["hole_cards"]) == 0  # type: ignore[arg-type]

    def test_shows_all_cards_at_showdown(self) -> None:
        state = _make_hand(stacks={0: 500, 1: 500}, dealer=0)
        # All-in and call
        state = apply_action(state, PlayerAction(seat=0, action=Action.ALL_IN, amount=450))
        state = apply_action(state, PlayerAction(seat=1, action=Action.CALL, amount=400))
        assert state.hand_over
        view = state.to_player_view(0)
        players = view["players"]
        assert isinstance(players, list)
        for pv in players:
            assert isinstance(pv, dict)
            if state.players[0].is_active and state.players[1].is_active:
                assert len(pv["hole_cards"]) == 2  # type: ignore[arg-type]
