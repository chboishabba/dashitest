from __future__ import annotations

"""
p-adic path utilities for GPU-friendly beam representations.

We encode the action history in base-3 digits so path similarity and prefix
classes can be computed with integer arithmetic.

Encoding choice:
- Depth 0 digit is the least-significant base-3 digit.
- This makes agreement depth correspond to the 3-adic valuation v3(id_a-id_b):
  the number of trailing base-3 zeros in the difference, capped at max_depth.
"""

import math


def action_to_digit(action: int) -> int:
    """
    Map beam actions {-1,0,+1} to base-3 digits {0,1,2}.
    """
    if action == -1:
        return 0
    if action == 0:
        return 1
    if action == 1:
        return 2
    raise ValueError(f"unsupported action for p-adic path encoding: {action!r}")


def digit_to_action(digit: int) -> int:
    if digit == 0:
        return -1
    if digit == 1:
        return 0
    if digit == 2:
        return 1
    raise ValueError(f"unsupported digit for p-adic path decoding: {digit!r}")


def pow3_upto(max_depth: int) -> list[int]:
    """
    Precompute [3^0, 3^1, ..., 3^max_depth].

    Uses Python ints (unbounded); callers may clamp depth to fit in int64.
    """
    if max_depth < 0:
        raise ValueError("max_depth must be >= 0")
    out = [1]
    for _ in range(max_depth):
        out.append(out[-1] * 3)
    return out


def push_digit(path_id: int, depth: int, digit: int, pow3: list[int] | None = None) -> int:
    """
    Append (depth,digit) into the path id using the fixed base-3 positional encoding.
    """
    if depth < 0:
        raise ValueError("depth must be >= 0")
    if digit not in (0, 1, 2):
        raise ValueError("digit must be 0,1,2")
    if pow3 is not None and depth < len(pow3):
        return int(path_id) + int(digit) * int(pow3[depth])
    return int(path_id) + int(digit) * (3**depth)


def agreement_depth(id_a: int, id_b: int, *, max_depth: int) -> int:
    """
    Compute v3(id_a-id_b) capped to max_depth.

    If the ids are identical, returns max_depth.
    """
    if max_depth < 0:
        raise ValueError("max_depth must be >= 0")
    diff = int(id_a) - int(id_b)
    if diff == 0:
        return int(max_depth)
    diff = abs(diff)
    depth = 0
    # trailing base-3 zeros in diff
    while depth < max_depth and diff % 3 == 0:
        diff //= 3
        depth += 1
    return int(depth)


def prefix_bucket(path_id: int, *, depth: int, pow3: list[int] | None = None) -> int:
    """
    Bucket key for the prefix class up to `depth` digits: id mod 3^depth.
    """
    if depth < 0:
        raise ValueError("depth must be >= 0")
    if depth == 0:
        return 0
    mod = pow3[depth] if pow3 is not None and depth < len(pow3) else 3**depth
    return int(path_id) % int(mod)


def decode_actions(path_id: int, *, depth: int) -> list[int]:
    """
    Decode the first `depth` digits of a path id into beam actions.

    This is intended for tests and debugging.
    """
    if depth < 0:
        raise ValueError("depth must be >= 0")
    x = int(path_id)
    actions: list[int] = []
    for _ in range(depth):
        digit = x % 3
        actions.append(digit_to_action(digit))
        x //= 3
    return actions


def max_safe_depth_int64() -> int:
    """
    Maximum depth such that 3^depth fits in signed int64.
    """
    # 3^39 is 4.05e18 < 2^63-1, 3^40 is 1.21e19 > 2^63-1.
    return 39

