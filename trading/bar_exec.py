from trading.base import BaseExecution


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

    def execute(self, intent, price: float):
        if getattr(intent, "hold", False):
            return {
                "filled": 0.0,
                "fill_price": price,
                "fee": 0.0,
                "pnl": 0.0,
                "exposure": self.exposure,
                "slippage": 0.0,
                "urgency": getattr(intent, "urgency", 0.0),
            }

        target = intent.target_exposure * intent.direction
        delta = target - self.exposure

        if abs(delta) < self.min_trade:
            return {
                "filled": 0.0,
                "fill_price": price,
                "fee": 0.0,
                "pnl": 0.0,
                "exposure": self.exposure,
                "slippage": 0.0,
                "urgency": getattr(intent, "urgency", 0.0),
            }

        # simulate fill with simple slippage
        slip = self.slippage * (1 if delta > 0 else -1)
        fill_price = price * (1 + slip)
        fee = abs(delta) * self.fee_rate
        pnl = -fee  # mark-to-market handled elsewhere

        self.exposure += delta

        return {
            "filled": delta,
            "fill_price": fill_price,
            "fee": fee,
            "pnl": pnl,
            "exposure": self.exposure,
            "slippage": slip,
            "urgency": getattr(intent, "urgency", 0.0),
        }
