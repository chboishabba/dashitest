"""
Triadic Operations using 2-bit P/N bitplanes.
Encoding:
  00 = 0 (FLAT)
  01 = +1 (BUY)
  11 = -1 (SELL)
  10 = ⊥ (UNKNOWN)

Triadic XOR (GF(3) addition) as bitwise logic.
"""

import numpy as np

def triadic_xor_bitwise(p1, n1, p2, n2):
    """
    GF(3) addition on (P, N) bitplanes.
    0  = (0, 0)
    +1 = (1, 0)
    -1 = (0, 1)
    ⊥  = (1, 1)  # Paradox / Invalid

    Formulas for branchless triadic XOR:
    P_out = (P_a & ~P_b & ~N_b) | (~P_a & P_b & ~N_a) | (N_a & N_b)
    N_out = (N_a & ~P_b & ~N_b) | (~P_a & ~N_a & N_b) | (P_a & P_b)
    """
    p_out = (p1 & ~p2 & ~n2) | (~p1 & p2 & ~n1) | (n1 & n2)
    n_out = (n1 & ~p2 & ~n2) | (~p1 & ~n1 & n2) | (p1 & p2)
    
    # ⊥ (1,1) propagation
    is_void1 = p1 & n1
    is_void2 = p2 & n2
    any_void = is_void1 | is_void2
    
    p_out = (p_out & ~any_void) | any_void
    n_out = (n_out & ~any_void) | any_void
    
    return int(p_out) & 1, int(n_out) & 1


def triadic_step_rotation(p, n):
    """
    120° phase-advance rotation (Add +1).
    0 (00) -> 1 (10) -> -1 (01) -> 0 (00)
    ⊥ (11) remains ⊥ (11)
    """
    # Equivalent to triadic_xor_bitwise(p, n, 1, 0)
    # p1=p, n1=n, p2=1, n2=0
    # p_out = (p & 0 & 1) | (~p & 1 & ~n) | (n & 0) = (~p & ~n)
    # n_out = (n & 0 & 1) | (~p & ~n & 0) | (p & 1) = p
    
    is_void = p & n
    new_p = (~p & ~n) & 1 & ~is_void
    new_n = p & 1 & ~is_void
    
    new_p |= is_void
    new_n |= is_void
    
    return int(new_p) & 1, int(new_n) & 1


def score_to_planes(score: float, threshold: float = 0.1) -> tuple[int, int]:
    """
    Collapse a continuous influence score into the P/N bitplanes.
    Scores above +threshold become +1, below -threshold become -1, and small scores stay 0.
    NaNs or infinities map to ⊥ (semantic poison).
    """
    if score is None or not np.isfinite(score):
        return 1, 1
    if score >= threshold:
        return 1, 0
    if score <= -threshold:
        return 0, 1
    return 0, 0


def planes_to_symbol(p: int, n: int) -> str:
    """
    Human-readable symbol for a given P/N combination.
    """
    if p & n:
        return "⊥"
    if p:
        return "+1"
    if n:
        return "-1"
    return "0"
