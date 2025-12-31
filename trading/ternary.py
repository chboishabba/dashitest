"""
ternary.py
----------
Small helpers for ternary control logic.
"""

from __future__ import annotations


def ternary_sign(x: float, tau: float = 0.0) -> int:
    if x > tau:
        return 1
    if x < -tau:
        return -1
    return 0


def ternary_permission(p_bad: float, caution: float = 0.4, ban: float = 0.7) -> int:
    if p_bad >= ban:
        return -1
    if p_bad >= caution:
        return 0
    return 1


def ternary_controller(
    direction: int,
    edge: int,
    permission: int,
    capital_pressure: int,
    thesis: int,
) -> int:
    if permission == -1:
        return 0
    if capital_pressure == -1 and permission != 1:
        return 0
    if permission == 0:
        return 0
    if direction == 0 or edge == 0 or direction != edge:
        return 0
    return direction
