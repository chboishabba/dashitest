from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class BasinMass:
    long_mass: float
    short_mass: float
    flat_mass: float


class BasinClassifier:
    """
    Classify terminal beam nodes by basin mass, normalized to one.
    """

    def classify(self, beam_nodes: Iterable) -> BasinMass:
        long_mass = 0.0
        short_mass = 0.0
        flat_mass = 0.0
        for node in beam_nodes:
            weight = float(getattr(node, "weight", 0.0))
            predicted_return = float(getattr(node, "predicted_return", 0.0))
            if predicted_return > 0:
                long_mass += weight
            elif predicted_return < 0:
                short_mass += weight
            else:
                flat_mass += weight
        total = long_mass + short_mass + flat_mass
        if total <= 0:
            return BasinMass(0.0, 0.0, 1.0)
        inv_total = 1.0 / total
        return BasinMass(long_mass * inv_total, short_mass * inv_total, flat_mass * inv_total)


def normalized_entropy(weights: Iterable[float]) -> float:
    items = [float(w) for w in weights if w > 0]
    total = sum(items)
    if total <= 0 or len(items) <= 1:
        return 0.0
    probs = [item / total for item in items]
    entropy = -sum(p * math.log(p) for p in probs)
    max_entropy = math.log(len(probs))
    if max_entropy <= 0:
        return 0.0
    return entropy / max_entropy
