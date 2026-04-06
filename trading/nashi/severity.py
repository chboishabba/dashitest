from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class SeverityLevel(IntEnum):
    NORMAL = 0
    INFO = 1
    CAUTION = 2
    HOLD = 3
    BAN = 4
    PARADOX = 5


@dataclass(frozen=True)
class SeverityCode:
    level: SeverityLevel
    label: str
    detail: str = ""

    @property
    def is_blocking(self) -> bool:
        return self.level >= SeverityLevel.HOLD


def combine_codes(*codes: SeverityCode | None) -> SeverityCode:
    chosen = SeverityCode(SeverityLevel.NORMAL, "normal", "")
    for code in codes:
        if code is None:
            continue
        if code.level >= chosen.level:
            chosen = code
    return chosen
