from __future__ import annotations

from dataclasses import dataclass, field

from .family_memory import FamilyMemory
from .severity import SeverityCode, SeverityLevel


@dataclass(frozen=True)
class NashiStepContext:
    ts: int
    price: float
    bid: float
    ask: float
    spread: float
    spread_bps: float
    price_return: float
    realized_vol: float
    actionability: float
    edge: float
    edge_persistence: float
    edge_shock: float
    stress: float
    microstructure_pressure: float
    cost_survival_ratio: float
    drawdown: float
    current_exposure: float
    family_cooldown: float = 0.0
    hazard: float = 0.0
    hazard_regime: str = "calm"
    hazard_p_bad: float = 0.0
    hazard_bad_flag: bool = False
    hazard_contextual_pressure: float = 0.0
    hazard_contextual_active: bool = False
    hazard_contextual_label: str = ""
    hazard_ema: float = 0.0
    hazard_persistence: float = 0.0
    hazard_trend: float = 0.0
    hazard_cooldown: float = 0.0


@dataclass
class NashiState:
    capital: float = 100000.0
    cash: float = 100000.0
    exposure: float = 0.0
    last_price: float = 0.0
    last_arrow: float = 0.0
    last_mdl: float = 0.0
    capital_drawdown: float = 0.0
    refusal: SeverityCode = field(
        default_factory=lambda: SeverityCode(SeverityLevel.NORMAL, "normal")
    )
    family_memory: FamilyMemory = field(default_factory=FamilyMemory)

    @property
    def equity(self) -> float:
        return self.cash + self.exposure * self.last_price
