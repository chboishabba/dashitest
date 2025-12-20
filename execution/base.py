class BaseExecution:
    """Abstract execution backend."""

    def execute(self, intents):
        """
        Args:
            intents: list of dicts with keys
                ts, side {-1,0,+1}, target_exposure, order_style, urgency, ttl
        Returns:
            fills: list of dicts {ts, qty, price}
            summary: dict with metrics (fees, slippage, fill_ratio, queue_delay, impact)
        """
        raise NotImplementedError
