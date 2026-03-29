"""Pot and side-pot calculation for poker hands."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Pot:
    """A main pot or side pot."""

    amount: int
    eligible_seats: frozenset[int]


def calculate_pots(bets: dict[int, int], folded: set[int]) -> list[Pot]:
    """Calculate main pot and side pots from player bets.

    Args:
        bets: seat -> total amount bet this hand
        folded: seats that folded (money in pot but can't win)

    Returns:
        List of Pots, main pot first. Each pot has amount and eligible (non-folded) seats.
    """
    if not bets:
        return []

    levels = sorted(set(bets.values()))
    pots: list[Pot] = []
    previous_level = 0

    carry = 0  # chips from levels where all contributors folded

    for level in levels:
        contribution_per_player = level - previous_level
        contributors = {seat for seat, bet in bets.items() if bet >= level}
        pot_amount = contribution_per_player * len(contributors) + carry
        eligible = frozenset(contributors - folded)

        if pot_amount > 0:
            if eligible:
                pots.append(Pot(amount=pot_amount, eligible_seats=eligible))
                carry = 0
            else:
                # All contributors folded — carry chips to next pot level
                carry = pot_amount

        previous_level = level

    # If carry remains (everyone at highest level folded), add to last pot
    if carry > 0 and pots:
        last = pots[-1]
        pots[-1] = Pot(amount=last.amount + carry, eligible_seats=last.eligible_seats)

    return pots


def award_pots(pots: list[Pot], hand_rankings: dict[int, tuple[int, ...]]) -> dict[int, int]:
    """Award pots to winners based on hand rankings.

    Args:
        pots: from calculate_pots
        hand_rankings: seat -> comparable ranking tuple (higher = better)

    Returns:
        seat -> total chips won
    """
    winnings: dict[int, int] = {}

    for pot in pots:
        eligible = {
            seat: hand_rankings[seat] for seat in pot.eligible_seats if seat in hand_rankings
        }
        if not eligible:
            raise ValueError(
                f"Pot has eligible seats {pot.eligible_seats} but none in hand_rankings"
            )

        best_rank = max(eligible.values())
        winners = sorted(seat for seat, rank in eligible.items() if rank == best_rank)

        share = pot.amount // len(winners)
        remainder = pot.amount % len(winners)

        for i, seat in enumerate(winners):
            won = share + (1 if i < remainder else 0)
            winnings[seat] = winnings.get(seat, 0) + won

    return winnings
