"""Game state machine for No-Limit Texas Hold'em.

Follows a Gymnasium-shaped API (reset/step/observation) for compatibility
with RL training, but has no framework dependency. Multi-agent turn-based:
each step() is one player's action, and the observation is from the
current actor's perspective.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from poker.engine.actions import Action, PlayerAction
from poker.engine.cards import Card, Deck
from poker.engine.errors import IllegalActionError, InvalidStateError
from poker.engine.evaluator import evaluate_hand
from poker.engine.pot import award_pots, calculate_pots


class Street(Enum):
    PREFLOP = "preflop"
    FLOP = "flop"
    TURN = "turn"
    RIVER = "river"
    SHOWDOWN = "showdown"
    COMPLETE = "complete"


@dataclass
class PlayerState:
    seat: int
    stack: int
    hole_cards: list[Card] = field(default_factory=list)
    is_active: bool = True  # still in hand (not folded)
    is_all_in: bool = False
    current_bet: int = 0  # bet in current betting round
    total_bet: int = 0  # total bet across all rounds this hand


# ---------------------------------------------------------------------------
# Observation / StepResult types
# ---------------------------------------------------------------------------

Observation = dict[str, Any]


@dataclass(frozen=True)
class StepResult:
    """Return type of PokerEnv.step(), shaped like Gymnasium."""

    observation: Observation
    reward: dict[int, float]  # seat -> reward (chip delta this step, 0 until hand ends)
    terminated: bool  # hand is over
    truncated: bool  # always False (no time limit)
    info: dict[str, Any]


# ---------------------------------------------------------------------------
# PokerEnv — the Gymnasium-shaped environment
# ---------------------------------------------------------------------------


@dataclass
class PokerEnv:
    """No-Limit Texas Hold'em environment with Gymnasium-shaped API.

    Multi-agent, turn-based. Each step() is one player's action.
    Use `current_actor_seat` to know whose turn it is, and `legal_actions()`
    to get valid moves.
    """

    players: list[PlayerState]
    deck: Deck
    board: list[Card]
    street: Street
    pot: int
    current_bet: int
    min_raise: int
    dealer_seat: int
    small_blind: int
    big_blind: int
    current_actor_idx: int
    last_raiser_idx: int | None
    hand_over: bool = False
    winners: dict[int, int] = field(default_factory=dict)
    _total_chips: int = 0
    _acted_this_round: set[int] = field(default_factory=set)
    _initial_stacks: dict[int, int] = field(default_factory=dict)

    # -- Gymnasium-shaped API -------------------------------------------------

    @property
    def current_actor_seat(self) -> int | None:
        """Seat of the player who must act next, or None if hand is over."""
        if self.hand_over:
            return None
        return self.players[self.current_actor_idx].seat

    @property
    def current_actor(self) -> PlayerState:
        return self.players[self.current_actor_idx]

    def observe(self, seat: int | None = None) -> Observation:
        """Get observation for a player. Defaults to current actor.

        Hides other players' hole cards (and folded cards at showdown).
        Includes legal actions for the observed player.
        """
        if seat is None:
            seat = self.current_actor.seat if not self.hand_over else self.players[0].seat

        players_view: list[dict[str, Any]] = []
        for p in self.players:
            pv: dict[str, Any] = {
                "seat": p.seat,
                "stack": p.stack,
                "is_active": p.is_active,
                "is_all_in": p.is_all_in,
                "current_bet": p.current_bet,
                "total_bet": p.total_bet,
            }
            show = p.seat == seat or (
                (self.street == Street.SHOWDOWN or self.hand_over) and p.is_active
            )
            pv["hole_cards"] = [str(c) for c in p.hole_cards] if show else []
            players_view.append(pv)

        return {
            "players": players_view,
            "board": [str(c) for c in self.board],
            "street": self.street.value,
            "pot": self.pot,
            "current_bet": self.current_bet,
            "min_raise": self.min_raise,
            "dealer_seat": self.dealer_seat,
            "small_blind": self.small_blind,
            "big_blind": self.big_blind,
            "current_actor_seat": self.current_actor_seat,
            "hand_over": self.hand_over,
            "winners": self.winners,
            "legal_actions": [
                {"action": a.action.value, "amount": a.amount} for a in legal_actions(self)
            ]
            if not self.hand_over and self.current_actor.seat == seat
            else [],
        }

    def step(self, action: PlayerAction) -> StepResult:
        """Apply a player action. Returns (observation, reward, terminated, truncated, info).

        Observation is from the next actor's perspective (or current if hand ended).
        Reward is chip delta per seat (non-zero only when hand completes).
        """
        apply_action(self, action)

        # Compute rewards: chip delta from initial stacks (only meaningful at hand end)
        reward: dict[int, float] = {}
        if self.hand_over:
            for p in self.players:
                reward[p.seat] = float(p.stack - self._initial_stacks[p.seat])
        else:
            for p in self.players:
                reward[p.seat] = 0.0

        obs = self.observe()
        info: dict[str, Any] = {
            "street": self.street.value,
            "winners": self.winners if self.hand_over else {},
        }

        return StepResult(
            observation=obs,
            reward=reward,
            terminated=self.hand_over,
            truncated=False,
            info=info,
        )

    def validate(self) -> None:
        """Check game state invariants. Raises InvalidStateError if violated."""
        validate_state(self)

    # Legacy aliases
    def to_player_view(self, seat: int) -> Observation:
        """Legacy alias for observe(seat)."""
        return self.observe(seat)


# ---------------------------------------------------------------------------
# Module-level functions (the engine core)
# ---------------------------------------------------------------------------


def _find_player_idx(state: PokerEnv, seat: int) -> int:
    for i, p in enumerate(state.players):
        if p.seat == seat:
            return i
    raise InvalidStateError(f"Seat {seat} not found")


def _next_active_idx(state: PokerEnv, from_idx: int) -> int:
    """Find next player index that is active and not all-in."""
    n = len(state.players)
    for offset in range(1, n + 1):
        idx = (from_idx + offset) % n
        p = state.players[idx]
        if p.is_active and not p.is_all_in:
            return idx
    return from_idx


def _count_active(state: PokerEnv) -> int:
    return sum(1 for p in state.players if p.is_active)


def _count_can_act(state: PokerEnv) -> int:
    """Count players who are active and not all-in."""
    return sum(1 for p in state.players if p.is_active and not p.is_all_in)


def _post_blind(player: PlayerState, amount: int) -> int:
    """Post a blind, possibly going all-in. Returns actual amount posted."""
    actual = min(amount, player.stack)
    player.stack -= actual
    player.current_bet = actual
    player.total_bet = actual
    if player.stack == 0:
        player.is_all_in = True
    return actual


def reset(
    seats_and_stacks: dict[int, int],
    dealer_seat: int,
    small_blind: int = 50,
    big_blind: int = 100,
) -> tuple[Observation, PokerEnv]:
    """Create a new hand. Returns (observation, env).

    Gymnasium-shaped: like env.reset() returning (obs, info),
    but returns the env itself since it's not a persistent object.
    """
    env = new_hand(seats_and_stacks, dealer_seat, small_blind, big_blind)
    obs = env.observe()
    return obs, env


def new_hand(
    seats_and_stacks: dict[int, int],
    dealer_seat: int,
    small_blind: int = 50,
    big_blind: int = 100,
) -> PokerEnv:
    """Create a new hand. Posts blinds, deals hole cards, sets up preflop."""
    if len(seats_and_stacks) < 2:
        raise InvalidStateError("Need at least 2 players")

    sorted_seats = sorted(seats_and_stacks.keys())
    players = [PlayerState(seat=s, stack=seats_and_stacks[s]) for s in sorted_seats]

    deck = Deck()
    state = PokerEnv(
        players=players,
        deck=deck,
        board=[],
        street=Street.PREFLOP,
        pot=0,
        current_bet=0,
        min_raise=big_blind,
        dealer_seat=dealer_seat,
        small_blind=small_blind,
        big_blind=big_blind,
        current_actor_idx=0,
        last_raiser_idx=None,
        _initial_stacks=dict(seats_and_stacks),
    )

    n = len(players)
    dealer_idx = _find_player_idx(state, dealer_seat)

    if n == 2:
        sb_idx = dealer_idx
        bb_idx = (dealer_idx + 1) % n
    else:
        sb_idx = (dealer_idx + 1) % n
        bb_idx = (dealer_idx + 2) % n

    sb_posted = _post_blind(players[sb_idx], small_blind)
    bb_posted = _post_blind(players[bb_idx], big_blind)
    state.current_bet = max(sb_posted, bb_posted)
    state.pot = sb_posted + bb_posted

    state._total_chips = sum(p.stack for p in players) + state.pot

    for p in players:
        p.hole_cards = deck.deal(2)

    first_actor_idx = (bb_idx + 1) % n
    if players[first_actor_idx].is_all_in:
        first_actor_idx = _next_active_idx(state, first_actor_idx)
    state.current_actor_idx = first_actor_idx
    state.last_raiser_idx = None

    if _count_can_act(state) <= 1 and _count_active(state) >= 2:
        if _count_can_act(state) == 1:
            can_act_idx = next(
                i for i, p in enumerate(state.players) if p.is_active and not p.is_all_in
            )
            state.current_actor_idx = can_act_idx
        elif _count_can_act(state) == 0:
            _advance_street(state)

    return state


def legal_actions(state: PokerEnv) -> list[PlayerAction]:
    """Return all legal actions for the current actor."""
    if state.hand_over:
        return []

    player = state.current_actor
    seat = player.seat
    actions: list[PlayerAction] = []
    to_call = state.current_bet - player.current_bet

    if to_call > 0:
        actions.append(PlayerAction(seat=seat, action=Action.FOLD))

        call_amount = min(to_call, player.stack)
        if call_amount > 0:
            actions.append(PlayerAction(seat=seat, action=Action.CALL, amount=call_amount))

        min_raise_to = state.current_bet + state.min_raise
        if player.stack + player.current_bet > state.current_bet:
            if player.stack + player.current_bet >= min_raise_to:
                min_raise_amount = min_raise_to - player.current_bet
                actions.append(
                    PlayerAction(seat=seat, action=Action.RAISE, amount=min_raise_amount)
                )
                if player.stack > min_raise_amount:
                    actions.append(
                        PlayerAction(seat=seat, action=Action.ALL_IN, amount=player.stack)
                    )
            else:
                actions.append(PlayerAction(seat=seat, action=Action.ALL_IN, amount=player.stack))
    else:
        actions.append(PlayerAction(seat=seat, action=Action.CHECK))

        if player.stack > 0:
            min_bet = state.big_blind
            if player.stack >= min_bet:
                actions.append(PlayerAction(seat=seat, action=Action.BET, amount=min_bet))
                if player.stack > min_bet:
                    actions.append(
                        PlayerAction(seat=seat, action=Action.ALL_IN, amount=player.stack)
                    )
            else:
                actions.append(PlayerAction(seat=seat, action=Action.ALL_IN, amount=player.stack))

    return actions


def apply_action(state: PokerEnv, action: PlayerAction) -> PokerEnv:
    """Apply a player action and return the (mutated) state."""
    if state.hand_over:
        raise IllegalActionError("Hand is over")

    player = state.current_actor
    if action.seat != player.seat:
        raise IllegalActionError(f"Not seat {action.seat}'s turn, expected seat {player.seat}")

    valid = legal_actions(state)
    _validate_action(action, valid)

    if action.action == Action.FOLD:
        player.is_active = False

    elif action.action == Action.CHECK:
        pass

    elif action.action == Action.CALL:
        amount = min(state.current_bet - player.current_bet, player.stack)
        player.stack -= amount
        player.current_bet += amount
        player.total_bet += amount
        state.pot += amount
        if player.stack == 0:
            player.is_all_in = True

    elif action.action in (Action.BET, Action.RAISE):
        amount = action.amount
        player.stack -= amount
        player.current_bet += amount
        player.total_bet += amount
        state.pot += amount
        raise_increment = player.current_bet - state.current_bet
        if raise_increment > state.min_raise:
            state.min_raise = raise_increment
        state.current_bet = player.current_bet
        state.last_raiser_idx = state.current_actor_idx
        if player.stack == 0:
            player.is_all_in = True

    elif action.action == Action.ALL_IN:
        amount = action.amount
        player.stack -= amount
        player.current_bet += amount
        player.total_bet += amount
        state.pot += amount
        player.is_all_in = True
        if player.current_bet > state.current_bet:
            raise_increment = player.current_bet - state.current_bet
            if raise_increment >= state.min_raise:
                state.min_raise = raise_increment
                state.last_raiser_idx = state.current_actor_idx
            state.current_bet = player.current_bet

    state._acted_this_round.add(state.current_actor_idx)

    if _count_active(state) == 1:
        _resolve_fold_win(state)
        return state

    _advance_action(state)

    return state


def _validate_action(action: PlayerAction, valid: list[PlayerAction]) -> None:
    """Check that action matches one of the legal actions."""
    for v in valid:
        if v.action == action.action:
            if action.action in (Action.FOLD, Action.CHECK):
                return
            if action.action == Action.CALL:
                if action.amount == v.amount:
                    return
            if action.action in (Action.BET, Action.RAISE):
                all_in_actions = [a for a in valid if a.action == Action.ALL_IN]
                max_amount = all_in_actions[0].amount if all_in_actions else v.amount
                if v.amount <= action.amount <= max_amount:
                    return
                if action.amount == v.amount:
                    return
            if action.action == Action.ALL_IN:
                if action.amount == v.amount:
                    return
    raise IllegalActionError(f"Illegal action: {action}")


def _advance_action(state: PokerEnv) -> None:
    """Move to next actor or advance street if betting round is over."""
    next_idx = _next_active_idx(state, state.current_actor_idx)

    round_over = False

    if _count_can_act(state) == 0:
        round_over = True
    elif state.last_raiser_idx is not None:
        if next_idx == state.last_raiser_idx:
            round_over = True
        elif state.players[state.last_raiser_idx].is_all_in:
            all_acted = all(
                i in state._acted_this_round
                for i, p in enumerate(state.players)
                if p.is_active and not p.is_all_in
            )
            if all_acted:
                round_over = True
    else:
        all_acted = all(
            i in state._acted_this_round
            for i, p in enumerate(state.players)
            if p.is_active and not p.is_all_in
        )
        if all_acted:
            round_over = True

    if round_over:
        _advance_street(state)
    else:
        state.current_actor_idx = next_idx


def _advance_street(state: PokerEnv) -> None:
    """Deal community cards and set up next betting round."""
    state._acted_this_round = set()
    for p in state.players:
        p.current_bet = 0
    state.current_bet = 0
    state.min_raise = state.big_blind
    state.last_raiser_idx = None

    if state.street == Street.PREFLOP:
        state.street = Street.FLOP
        state.board.extend(state.deck.deal(3))
    elif state.street == Street.FLOP:
        state.street = Street.TURN
        state.board.extend(state.deck.deal(1))
    elif state.street == Street.TURN:
        state.street = Street.RIVER
        state.board.extend(state.deck.deal(1))
    elif state.street == Street.RIVER:
        _resolve_showdown(state)
        return
    else:
        return

    can_act = _count_can_act(state)
    if can_act <= 1 and _count_active(state) >= 2:
        _advance_street(state)
        return

    dealer_idx = _find_player_idx(state, state.dealer_seat)
    n = len(state.players)
    for offset in range(1, n + 1):
        idx = (dealer_idx + offset) % n
        if state.players[idx].is_active and not state.players[idx].is_all_in:
            state.current_actor_idx = idx
            break


def _resolve_fold_win(state: PokerEnv) -> None:
    """Everyone folded except one player — they win the pot."""
    winner = next(p for p in state.players if p.is_active)
    winner.stack += state.pot
    state.winners = {winner.seat: state.pot}
    state.pot = 0
    state.hand_over = True
    state.street = Street.COMPLETE


def _resolve_showdown(state: PokerEnv) -> None:
    """Evaluate hands and distribute pots."""
    state.street = Street.SHOWDOWN

    bets: dict[int, int] = {}
    folded: set[int] = set()
    for p in state.players:
        bets[p.seat] = p.total_bet
        if not p.is_active:
            folded.add(p.seat)

    hand_rankings: dict[int, tuple[int, ...]] = {}
    for p in state.players:
        if p.is_active:
            result = evaluate_hand(p.hole_cards + state.board)
            hand_rankings[p.seat] = (result.category, *result.tiebreakers)

    pots = calculate_pots(bets, folded)
    winnings = award_pots(pots, hand_rankings)

    for seat, amount in winnings.items():
        for p in state.players:
            if p.seat == seat:
                p.stack += amount

    state.winners = winnings
    state.pot = 0
    state.hand_over = True
    state.street = Street.COMPLETE


def validate_state(state: PokerEnv) -> None:
    """Check game state invariants. Raises InvalidStateError if violated."""
    actual_total = sum(p.stack for p in state.players) + state.pot
    if state._total_chips > 0 and actual_total != state._total_chips:
        raise InvalidStateError(
            f"Chip conservation violated: expected {state._total_chips}, "
            f"got {actual_total} (stacks={sum(p.stack for p in state.players)}, pot={state.pot})"
        )

    for p in state.players:
        if p.stack < 0:
            raise InvalidStateError(f"Negative stack for seat {p.seat}: {p.stack}")

    if _count_active(state) < 1:
        raise InvalidStateError("No active players")
