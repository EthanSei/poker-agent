"""Game state machine for No-Limit Texas Hold'em."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

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


@dataclass
class GameState:
    players: list[PlayerState]
    deck: Deck
    board: list[Card]
    street: Street
    pot: int  # total pot (sum of all bets committed)
    current_bet: int  # current bet to match in this round
    min_raise: int  # minimum raise increment
    dealer_seat: int
    small_blind: int
    big_blind: int
    current_actor_idx: int  # index into players list
    last_raiser_idx: int | None  # who last raised (betting ends when action returns here)
    hand_over: bool = False
    winners: dict[int, int] = field(default_factory=dict)  # seat -> chips won
    _total_chips: int = 0  # invariant: total chips in play
    _acted_this_round: set[int] = field(default_factory=set)  # indices that have acted

    @property
    def current_actor(self) -> PlayerState:
        return self.players[self.current_actor_idx]

    def to_player_view(self, seat: int) -> dict[str, object]:
        """Return a dict with all public info + only that seat's hole cards."""
        players_view: list[dict[str, object]] = []
        for p in self.players:
            pv: dict[str, object] = {
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
            if show:
                pv["hole_cards"] = [str(c) for c in p.hole_cards]
            else:
                pv["hole_cards"] = []
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
            "current_actor_seat": self.current_actor.seat if not self.hand_over else None,
            "hand_over": self.hand_over,
            "winners": self.winners,
        }


def _find_player_idx(state: GameState, seat: int) -> int:
    for i, p in enumerate(state.players):
        return_val = i
        if p.seat == seat:
            return return_val
    raise InvalidStateError(f"Seat {seat} not found")


def _next_active_idx(state: GameState, from_idx: int) -> int:
    """Find next player index that is active and not all-in."""
    n = len(state.players)
    for offset in range(1, n + 1):
        idx = (from_idx + offset) % n
        p = state.players[idx]
        if p.is_active and not p.is_all_in:
            return idx
    return from_idx  # no one else can act


def _count_active(state: GameState) -> int:
    return sum(1 for p in state.players if p.is_active)


def _count_can_act(state: GameState) -> int:
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


def new_hand(
    seats_and_stacks: dict[int, int],
    dealer_seat: int,
    small_blind: int = 50,
    big_blind: int = 100,
) -> GameState:
    """Create a new hand. Posts blinds, deals hole cards, sets up preflop."""
    if len(seats_and_stacks) < 2:
        raise InvalidStateError("Need at least 2 players")

    sorted_seats = sorted(seats_and_stacks.keys())
    players = [PlayerState(seat=s, stack=seats_and_stacks[s]) for s in sorted_seats]

    deck = Deck()
    state = GameState(
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
    )

    n = len(players)
    dealer_idx = _find_player_idx(state, dealer_seat)

    if n == 2:
        # Heads-up: dealer posts SB, other posts BB
        sb_idx = dealer_idx
        bb_idx = (dealer_idx + 1) % n
    else:
        # 3+ players: left of dealer posts SB, next posts BB
        sb_idx = (dealer_idx + 1) % n
        bb_idx = (dealer_idx + 2) % n

    sb_posted = _post_blind(players[sb_idx], small_blind)
    bb_posted = _post_blind(players[bb_idx], big_blind)
    state.current_bet = max(sb_posted, bb_posted)
    state.pot = sb_posted + bb_posted

    # Set total chips invariant
    state._total_chips = sum(p.stack for p in players) + state.pot

    # Deal hole cards
    for p in players:
        p.hole_cards = deck.deal(2)

    # Preflop: action starts left of BB
    first_actor_idx = (bb_idx + 1) % n
    # Skip players who are all-in
    if players[first_actor_idx].is_all_in:
        first_actor_idx = _next_active_idx(state, first_actor_idx)
    state.current_actor_idx = first_actor_idx
    # No last_raiser preflop — BB option is handled by _acted_this_round
    state.last_raiser_idx = None

    # If only one player can act (other is all-in from blinds), handle it
    if _count_can_act(state) <= 1 and _count_active(state) >= 2:
        if _count_can_act(state) == 1:
            can_act_idx = next(
                i for i, p in enumerate(state.players) if p.is_active and not p.is_all_in
            )
            state.current_actor_idx = can_act_idx
        elif _count_can_act(state) == 0:
            # Everyone is all-in, advance to showdown
            _advance_street(state)

    return state


def legal_actions(state: GameState) -> list[PlayerAction]:
    """Return all legal actions for the current actor."""
    if state.hand_over:
        return []

    player = state.current_actor
    seat = player.seat
    actions: list[PlayerAction] = []
    to_call = state.current_bet - player.current_bet

    if to_call > 0:
        # Facing a bet
        actions.append(PlayerAction(seat=seat, action=Action.FOLD))

        # CALL
        call_amount = min(to_call, player.stack)
        if call_amount > 0:
            actions.append(PlayerAction(seat=seat, action=Action.CALL, amount=call_amount))

        # RAISE
        min_raise_to = state.current_bet + state.min_raise
        if player.stack + player.current_bet > state.current_bet:
            # Player can put in more than a call
            if player.stack + player.current_bet >= min_raise_to:
                # Can make a legal min-raise
                min_raise_amount = min_raise_to - player.current_bet
                actions.append(
                    PlayerAction(seat=seat, action=Action.RAISE, amount=min_raise_amount)
                )
                # All-in raise if different from min raise
                if player.stack > min_raise_amount:
                    actions.append(
                        PlayerAction(seat=seat, action=Action.ALL_IN, amount=player.stack)
                    )
            else:
                # Can only go all-in for less than a full raise
                actions.append(PlayerAction(seat=seat, action=Action.ALL_IN, amount=player.stack))
    else:
        # No bet to match
        actions.append(PlayerAction(seat=seat, action=Action.CHECK))

        # BET
        if player.stack > 0:
            min_bet = state.big_blind
            if player.stack >= min_bet:
                actions.append(PlayerAction(seat=seat, action=Action.BET, amount=min_bet))
                if player.stack > min_bet:
                    actions.append(
                        PlayerAction(seat=seat, action=Action.ALL_IN, amount=player.stack)
                    )
            else:
                # Can only go all-in for less than min bet
                actions.append(PlayerAction(seat=seat, action=Action.ALL_IN, amount=player.stack))

    return actions


def apply_action(state: GameState, action: PlayerAction) -> GameState:
    """Apply a player action and return the (mutated) state."""
    if state.hand_over:
        raise IllegalActionError("Hand is over")

    player = state.current_actor
    if action.seat != player.seat:
        raise IllegalActionError(f"Not seat {action.seat}'s turn, expected seat {player.seat}")

    # Validate action is in legal actions list
    valid = legal_actions(state)
    _validate_action(action, valid)

    # Apply the action
    if action.action == Action.FOLD:
        player.is_active = False

    elif action.action == Action.CHECK:
        pass  # nothing to do

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

    # Track that this player has acted
    state._acted_this_round.add(state.current_actor_idx)

    # Check if hand is over (only one active player)
    if _count_active(state) == 1:
        _resolve_fold_win(state)
        return state

    # Advance to next actor or next street
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
                # Allow any amount between min and all-in
                # Find the all-in action to get max
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


def _advance_action(state: GameState) -> None:
    """Move to next actor or advance street if betting round is over."""
    next_idx = _next_active_idx(state, state.current_actor_idx)

    # Check if betting round is over
    round_over = False

    if _count_can_act(state) == 0:
        # Everyone is all-in or folded
        round_over = True
    elif state.last_raiser_idx is not None:
        if next_idx == state.last_raiser_idx:
            # Action returned to last raiser
            round_over = True
        elif state.players[state.last_raiser_idx].is_all_in:
            # Last raiser is all-in; round ends when all others have acted
            all_acted = all(
                i in state._acted_this_round
                for i, p in enumerate(state.players)
                if p.is_active and not p.is_all_in
            )
            if all_acted:
                round_over = True
    else:
        # No raise this round. Round ends when all eligible players have acted.
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


def _advance_street(state: GameState) -> None:
    """Deal community cards and set up next betting round."""
    # Reset current bets and acted tracking for new round
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

    # If no one (or only one) can act but multiple players remain, run out the board
    can_act = _count_can_act(state)
    if can_act <= 1 and _count_active(state) >= 2:
        _advance_street(state)
        return

    # Set first actor for postflop: first active non-all-in player left of dealer
    dealer_idx = _find_player_idx(state, state.dealer_seat)
    n = len(state.players)
    for offset in range(1, n + 1):
        idx = (dealer_idx + offset) % n
        if state.players[idx].is_active and not state.players[idx].is_all_in:
            state.current_actor_idx = idx
            break


def _resolve_fold_win(state: GameState) -> None:
    """Everyone folded except one player — they win the pot."""
    winner = next(p for p in state.players if p.is_active)
    winner.stack += state.pot
    state.winners = {winner.seat: state.pot}
    state.pot = 0
    state.hand_over = True
    state.street = Street.COMPLETE


def _resolve_showdown(state: GameState) -> None:
    """Evaluate hands and distribute pots."""
    state.street = Street.SHOWDOWN

    # Build bets dict and folded set
    bets: dict[int, int] = {}
    folded: set[int] = set()
    for p in state.players:
        bets[p.seat] = p.total_bet
        if not p.is_active:
            folded.add(p.seat)

    # Evaluate hands
    hand_rankings: dict[int, tuple[int, ...]] = {}
    for p in state.players:
        if p.is_active:
            result = evaluate_hand(p.hole_cards + state.board)
            hand_rankings[p.seat] = (result.category, *result.tiebreakers)

    # Calculate and award pots
    pots = calculate_pots(bets, folded)
    winnings = award_pots(pots, hand_rankings)

    # Update stacks
    for seat, amount in winnings.items():
        for p in state.players:
            if p.seat == seat:
                p.stack += amount

    state.winners = winnings
    state.pot = 0
    state.hand_over = True
    state.street = Street.COMPLETE


def validate_state(state: GameState) -> None:
    """Check game state invariants. Raises InvalidStateError if violated."""
    # Total chips conservation
    total = sum(p.stack for p in state.players) + state.pot
    for p in state.players:
        total += p.current_bet  # current round bets not yet in pot... wait
    # Actually pot already includes committed bets. current_bet on player is
    # tracked for betting logic but the chips are already in pot.
    # Let's recalculate: total chips = stacks + pot
    actual_total = sum(p.stack for p in state.players) + state.pot
    if state._total_chips > 0 and actual_total != state._total_chips:
        raise InvalidStateError(
            f"Chip conservation violated: expected {state._total_chips}, "
            f"got {actual_total} (stacks={sum(p.stack for p in state.players)}, pot={state.pot})"
        )

    # No negative stacks
    for p in state.players:
        if p.stack < 0:
            raise InvalidStateError(f"Negative stack for seat {p.seat}: {p.stack}")

    # At least one active player
    if _count_active(state) < 1:
        raise InvalidStateError("No active players")
