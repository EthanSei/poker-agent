"""Custom exceptions for the poker engine."""


class IllegalActionError(Exception):
    """Raised when a player attempts an illegal action."""


class InvalidStateError(Exception):
    """Raised when the game state is invalid."""
