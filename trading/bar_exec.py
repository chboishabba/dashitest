try:
    from trading.base import BaseExecution
except ModuleNotFoundError:
    from base import BaseExecution


class BarExecution(BaseExecution):
    """
    Simple bar-level execution that moves exposure toward intent.target_exposure
    at the given bar price, with flat slippage/fees.
    """

    def __init__(self, fee_rate: float = 0.0005, slippage: float = 0.0003, min_trade: float = 0.02):
        self.fee_rate = fee_rate
        self.slippage = slippage
        self.exposure = 0.0  # current portfolio fraction
        self.min_trade = min_trade

    def execute(
        self,
        intent,
        price: float,
        *,
        bid: float | None = None,
        ask: float | None = None,
        capital: float | None = None,
    ):
        if getattr(intent, "hold", False):
            return {
                "requested_delta": 0.0,
                "filled": 0.0,
                "fill_ratio": 0.0,
                "fill_price": price,
                "fee": 0.0,
                "pnl": 0.0,
                "exposure": self.exposure,
                "slippage": 0.0,
                "slippage_cost": 0.0,
                "urgency": getattr(intent, "urgency", 0.0),
            }

        target = intent.target_exposure * intent.direction
        delta = target - self.exposure

        if abs(delta) < self.min_trade:
            return {
                "requested_delta": delta,
                "filled": 0.0,
                "fill_ratio": 0.0,
                "fill_price": price,
                "fee": 0.0,
                "pnl": 0.0,
                "exposure": self.exposure,
                "slippage": 0.0,
                "slippage_cost": 0.0,
                "urgency": getattr(intent, "urgency", 0.0),
            }

        if bid is not None and ask is not None and ask >= bid > 0:
            fill_price = ask if delta > 0 else bid
            slip = (fill_price / max(price, 1e-9)) - 1.0
        else:
            slip = self.slippage * (1 if delta > 0 else -1)
            fill_price = price * (1 + slip)
        notional_capital = max(float(capital), 0.0) if capital is not None else 1.0
        fee = abs(delta) * notional_capital * self.fee_rate
        slippage_cost = abs(delta) * notional_capital * abs(slip)
        pnl = -(fee + slippage_cost)  # mark-to-market handled elsewhere

        self.exposure += delta

        return {
            "requested_delta": delta,
            "filled": delta,
            "fill_ratio": 1.0,
            "fill_price": fill_price,
            "fee": fee,
            "pnl": pnl,
            "exposure": self.exposure,
            "slippage": slip,
            "slippage_cost": slippage_cost,
            "urgency": getattr(intent, "urgency", 0.0),
        }
