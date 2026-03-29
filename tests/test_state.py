"""Tests for the game state machine."""

import pytest

from poker.engine.actions import Action, PlayerAction
from poker.engine.errors import IllegalActionError
from poker.engine.state import (
    PokerEnv,
    StepResult,
    Street,
    apply_action,
    legal_actions,
    new_hand,
    reset,
    validate_state,
)


def _make_hand(
    stacks: dict[int, int] | None = None,
    dealer: int = 0,
    sb: int = 50,
    bb: int = 100,
) -> PokerEnv:
    """Helper to create a standard hand."""
    if stacks is None:
        stacks = {0: 1000, 1: 1000, 2: 1000}
    return new_hand(stacks, dealer_seat=dealer, small_blind=sb, big_blind=bb)


# -- Gymnasium-shaped API tests -----------------------------------------------


class TestGymnasiumAPI:
    def test_reset_returns_obs_and_env(self) -> None:
        obs, env = reset({0: 1000, 1: 1000, 2: 1000}, dealer_seat=0)
        assert isinstance(obs, dict)
        assert isinstance(env, PokerEnv)
        assert obs["street"] == "preflop"
        assert obs["current_actor_seat"] == 0
        assert len(obs["legal_actions"]) > 0

    def test_step_returns_step_result(self) -> None:
        obs, env = reset({0: 1000, 1: 1000, 2: 1000}, dealer_seat=0)
        result = env.step(PlayerAction(seat=0, action=Action.CALL, amount=100))
        assert isinstance(result, StepResult)
        assert isinstance(result.observation, dict)
        assert isinstance(result.reward, dict)
        assert result.terminated is False
        assert result.truncated is False
        assert result.info["street"] == "preflop"

    def test_step_rewards_zero_until_hand_over(self) -> None:
        _, env = reset({0: 1000, 1: 1000}, dealer_seat=0)
        result = env.step(PlayerAction(seat=0, action=Action.CALL, amount=50))
        for reward in result.reward.values():
            assert reward == 0.0

    def test_step_rewards_nonzero_at_hand_end(self) -> None:
        _, env = reset({0: 1000, 1: 1000, 2: 1000}, dealer_seat=0)
        # UTG folds, SB folds -> BB wins
        env.step(PlayerAction(seat=0, action=Action.FOLD))
        result = env.step(PlayerAction(seat=1, action=Action.FOLD))
        assert result.terminated is True
        # BB gained 50 from SB
        assert result.reward[2] == 50.0
        # UTG lost nothing (no blind), SB lost 50
        assert result.reward[0] == 0.0
        assert result.reward[1] == -50.0

    def test_observe_shows_legal_actions_for_current_actor(self) -> None:
        _, env = reset({0: 1000, 1: 1000, 2: 1000}, dealer_seat=0)
        obs = env.observe(seat=0)
        assert len(obs["legal_actions"]) > 0
        # Observing from a non-current-actor seat shows no legal actions
        obs_other = env.observe(seat=1)
        assert len(obs_other["legal_actions"]) == 0

    def test_current_actor_seat_none_when_over(self) -> None:
        _, env = reset({0: 1000, 1: 1000}, dealer_seat=0)
        env.step(PlayerAction(seat=0, action=Action.FOLD))
        assert env.current_actor_seat is None

    def test_full_hand_via_step(self) -> None:
        """Play a complete hand using only the step() API."""
        _, env = reset({0: 1000, 1: 1000}, dealer_seat=0)

        # Preflop: SB calls, BB checks
        result = env.step(PlayerAction(seat=0, action=Action.CALL, amount=50))
        assert not result.terminated
        result = env.step(PlayerAction(seat=1, action=Action.CHECK))
        assert not result.terminated

        # Flop through river: all check
        while not result.terminated:
            seat = env.current_actor_seat
            assert seat is not None
            result = env.step(PlayerAction(seat=seat, action=Action.CHECK))

        assert result.terminated
        assert sum(result.reward.values()) == 0.0  # zero-sum


# -- Original engine tests (updated to use PokerEnv) -------------------------


class TestNewHand:
    def test_creates_valid_state(self) -> None:
        state = _make_hand()
        assert state.street == Street.PREFLOP
        assert not state.hand_over
        assert len(state.board) == 0
        validate_state(state)

    def test_blinds_posted(self) -> None:
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

    def test_heads_up_dealer_posts_sb(self) -> None:
        state = _make_hand(stacks={0: 1000, 1: 1000}, dealer=0)
        dealer_player = next(p for p in state.players if p.seat == 0)
        other_player = next(p for p in state.players if p.seat == 1)
        assert dealer_player.stack == 950
        assert dealer_player.current_bet == 50
        assert other_player.stack == 900
        assert other_player.current_bet == 100
        assert state.current_actor.seat == 0

    def test_at_least_two_players(self) -> None:
        with pytest.raises(Exception):
            _make_hand(stacks={0: 1000})


class TestFullHand:
    def test_full_hand_all_check(self) -> None:
        state = _make_hand(stacks={0: 1000, 1: 1000, 2: 1000}, dealer=0)
        validate_state(state)

        assert state.current_actor.seat == 0
        state = apply_action(state, PlayerAction(seat=0, action=Action.CALL, amount=100))
        assert state.current_actor.seat == 1
        state = apply_action(state, PlayerAction(seat=1, action=Action.CALL, amount=50))
        assert state.current_actor.seat == 2
        state = apply_action(state, PlayerAction(seat=2, action=Action.CHECK))

        assert state.street == Street.FLOP
        assert len(state.board) == 3
        validate_state(state)

        assert state.current_actor.seat == 1
        state = apply_action(state, PlayerAction(seat=1, action=Action.CHECK))
        state = apply_action(state, PlayerAction(seat=2, action=Action.CHECK))
        state = apply_action(state, PlayerAction(seat=0, action=Action.CHECK))

        assert state.street == Street.TURN
        assert len(state.board) == 4

        state = apply_action(state, PlayerAction(seat=1, action=Action.CHECK))
        state = apply_action(state, PlayerAction(seat=2, action=Action.CHECK))
        state = apply_action(state, PlayerAction(seat=0, action=Action.CHECK))

        assert state.street == Street.RIVER
        assert len(state.board) == 5

        state = apply_action(state, PlayerAction(seat=1, action=Action.CHECK))
        state = apply_action(state, PlayerAction(seat=2, action=Action.CHECK))
        state = apply_action(state, PlayerAction(seat=0, action=Action.CHECK))

        assert state.hand_over
        assert state.street == Street.COMPLETE
        assert len(state.winners) > 0
        validate_state(state)

        total = sum(p.stack for p in state.players) + state.pot
        assert total == 3000


class TestFoldToWin:
    def test_everyone_folds(self) -> None:
        state = _make_hand(stacks={0: 1000, 1: 1000, 2: 1000}, dealer=0)
        state = apply_action(state, PlayerAction(seat=0, action=Action.FOLD))
        state = apply_action(state, PlayerAction(seat=1, action=Action.FOLD))

        assert state.hand_over
        assert state.street == Street.COMPLETE
        assert 2 in state.winners
        assert state.winners[2] == 150

        bb_player = next(p for p in state.players if p.seat == 2)
        assert bb_player.stack == 1050
        validate_state(state)


class TestAllIn:
    def test_all_in_and_call(self) -> None:
        state = _make_hand(stacks={0: 500, 1: 500}, dealer=0)
        state = apply_action(state, PlayerAction(seat=0, action=Action.ALL_IN, amount=450))
        assert state.current_actor.seat == 1
        state = apply_action(state, PlayerAction(seat=1, action=Action.CALL, amount=400))

        assert state.hand_over
        assert state.street == Street.COMPLETE
        assert len(state.board) == 5
        total = sum(p.stack for p in state.players)
        assert total == 1000
        validate_state(state)


class TestLegalActions:
    def test_cant_check_facing_bet(self) -> None:
        state = _make_hand(stacks={0: 1000, 1: 1000, 2: 1000}, dealer=0)
        actions = legal_actions(state)
        action_types = {a.action for a in actions}
        assert Action.CHECK not in action_types
        assert Action.FOLD in action_types
        assert Action.CALL in action_types

    def test_can_check_when_no_bet(self) -> None:
        state = _make_hand(stacks={0: 1000, 1: 1000, 2: 1000}, dealer=0)
        state = apply_action(state, PlayerAction(seat=0, action=Action.CALL, amount=100))
        state = apply_action(state, PlayerAction(seat=1, action=Action.CALL, amount=50))
        state = apply_action(state, PlayerAction(seat=2, action=Action.CHECK))
        actions = legal_actions(state)
        action_types = {a.action for a in actions}
        assert Action.CHECK in action_types
        assert Action.FOLD not in action_types

    def test_must_raise_not_bet_facing_bet(self) -> None:
        state = _make_hand(stacks={0: 1000, 1: 1000, 2: 1000}, dealer=0)
        state = apply_action(state, PlayerAction(seat=0, action=Action.RAISE, amount=200))
        actions = legal_actions(state)
        action_types = {a.action for a in actions}
        assert Action.BET not in action_types
        assert Action.RAISE in action_types or Action.ALL_IN in action_types


class TestIllegalActions:
    def test_wrong_seat(self) -> None:
        state = _make_hand(stacks={0: 1000, 1: 1000, 2: 1000}, dealer=0)
        with pytest.raises(IllegalActionError):
            apply_action(state, PlayerAction(seat=1, action=Action.FOLD))

    def test_invalid_raise_amount(self) -> None:
        state = _make_hand(stacks={0: 1000, 1: 1000, 2: 1000}, dealer=0)
        with pytest.raises(IllegalActionError):
            apply_action(state, PlayerAction(seat=0, action=Action.RAISE, amount=110))


class TestMinRaise:
    def test_min_raise_size(self) -> None:
        state = _make_hand(stacks={0: 5000, 1: 5000, 2: 5000}, dealer=0)
        state = apply_action(state, PlayerAction(seat=0, action=Action.RAISE, amount=200))
        actions = legal_actions(state)
        raise_actions = [a for a in actions if a.action == Action.RAISE]
        assert len(raise_actions) >= 1
        assert raise_actions[0].amount == 250

    def test_reraise_increases_min(self) -> None:
        state = _make_hand(stacks={0: 5000, 1: 5000, 2: 5000}, dealer=0)
        state = apply_action(state, PlayerAction(seat=0, action=Action.RAISE, amount=200))
        state = apply_action(state, PlayerAction(seat=1, action=Action.RAISE, amount=450))
        actions = legal_actions(state)
        raise_actions = [a for a in actions if a.action == Action.RAISE]
        assert len(raise_actions) >= 1
        assert raise_actions[0].amount == 700


class TestSidePot:
    def test_short_stack_all_in(self) -> None:
        state = _make_hand(stacks={0: 300, 1: 1000, 2: 1000}, dealer=0)
        state = apply_action(state, PlayerAction(seat=0, action=Action.ALL_IN, amount=300))
        state = apply_action(state, PlayerAction(seat=1, action=Action.CALL, amount=250))
        state = apply_action(state, PlayerAction(seat=2, action=Action.CALL, amount=200))

        assert state.street == Street.FLOP

        for _street in range(3):
            if state.hand_over:
                break
            for _action in range(2):
                if state.hand_over:
                    break
                seat = state.current_actor.seat
                state = apply_action(state, PlayerAction(seat=seat, action=Action.CHECK))

        assert state.hand_over
        total = sum(p.stack for p in state.players)
        assert total == 2300
        validate_state(state)


class TestChipConservation:
    def test_across_two_hands(self) -> None:
        stacks = {0: 1000, 1: 1000}

        state = _make_hand(stacks=stacks, dealer=0)
        state = apply_action(state, PlayerAction(seat=0, action=Action.FOLD))
        assert state.hand_over

        new_stacks = {p.seat: p.stack for p in state.players}
        assert sum(new_stacks.values()) == 2000

        state2 = _make_hand(stacks=new_stacks, dealer=1)
        validate_state(state2)
        state2 = apply_action(state2, PlayerAction(seat=1, action=Action.FOLD))
        assert state2.hand_over

        final_total = sum(p.stack for p in state2.players)
        assert final_total == 2000


class TestStreetAdvancement:
    def test_board_card_counts(self) -> None:
        state = _make_hand(stacks={0: 1000, 1: 1000}, dealer=0)
        assert len(state.board) == 0

        state = apply_action(state, PlayerAction(seat=0, action=Action.CALL, amount=50))
        state = apply_action(state, PlayerAction(seat=1, action=Action.CHECK))
        assert state.street == Street.FLOP
        assert len(state.board) == 3

        state = apply_action(state, PlayerAction(seat=1, action=Action.CHECK))
        state = apply_action(state, PlayerAction(seat=0, action=Action.CHECK))
        assert state.street == Street.TURN
        assert len(state.board) == 4

        state = apply_action(state, PlayerAction(seat=1, action=Action.CHECK))
        state = apply_action(state, PlayerAction(seat=0, action=Action.CHECK))
        assert state.street == Street.RIVER
        assert len(state.board) == 5

        state = apply_action(state, PlayerAction(seat=1, action=Action.CHECK))
        state = apply_action(state, PlayerAction(seat=0, action=Action.CHECK))
        assert state.hand_over


class TestBBOption:
    def test_bb_can_raise_after_limps(self) -> None:
        state = _make_hand(stacks={0: 1000, 1: 1000, 2: 1000}, dealer=0)
        state = apply_action(state, PlayerAction(seat=0, action=Action.CALL, amount=100))
        state = apply_action(state, PlayerAction(seat=1, action=Action.CALL, amount=50))

        assert state.current_actor.seat == 2
        assert state.street == Street.PREFLOP

        actions = legal_actions(state)
        action_types = {a.action for a in actions}
        assert Action.CHECK in action_types
        assert Action.BET in action_types or Action.RAISE in action_types

    def test_bb_check_advances_to_flop(self) -> None:
        state = _make_hand(stacks={0: 1000, 1: 1000, 2: 1000}, dealer=0)
        state = apply_action(state, PlayerAction(seat=0, action=Action.CALL, amount=100))
        state = apply_action(state, PlayerAction(seat=1, action=Action.CALL, amount=50))
        state = apply_action(state, PlayerAction(seat=2, action=Action.CHECK))
        assert state.street == Street.FLOP


class TestPlayerView:
    def test_hides_other_hole_cards(self) -> None:
        state = _make_hand(stacks={0: 1000, 1: 1000, 2: 1000}, dealer=0)
        view = state.observe(0)
        players = view["players"]
        assert isinstance(players, list)
        for pv in players:
            assert isinstance(pv, dict)
            if pv["seat"] == 0:
                assert len(pv["hole_cards"]) == 2
            else:
                assert len(pv["hole_cards"]) == 0

    def test_shows_active_cards_at_showdown(self) -> None:
        state = _make_hand(stacks={0: 500, 1: 500}, dealer=0)
        state = apply_action(state, PlayerAction(seat=0, action=Action.ALL_IN, amount=450))
        state = apply_action(state, PlayerAction(seat=1, action=Action.CALL, amount=400))
        assert state.hand_over
        view = state.observe(0)
        players = view["players"]
        assert isinstance(players, list)
        for pv in players:
            assert isinstance(pv, dict)
            if state.players[0].is_active and state.players[1].is_active:
                assert len(pv["hole_cards"]) == 2
