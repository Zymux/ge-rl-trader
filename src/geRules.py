from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class GERules:
    """
    OSRS GE rules for the simulator:

    - Buy: no tax
    - Sell: GE tax (default 2%), rounded down to whole coins, with optional cap per trade
    - buy_limit_window_seconds (4h) is enforced in the env, not here
    """

    sell_tax_rate: float = 0.02
    apply_tax_cap: bool = True
    tax_cap_gp: int = 5_000_000

    def sell_tax(self, gross_proceeds: float) -> float:
        tax = self.sell_tax_rate * gross_proceeds
        tax = math.floor(tax)  # GE rounds down to whole coins
        if self.apply_tax_cap:
            tax = min(tax, self.tax_cap_gp)
        return float(tax)

    def sell_net_proceeds(self, gross_proceeds: float) -> float:
        return float(gross_proceeds) - self.sell_tax(gross_proceeds)

    def buy_cost(self, cash_spent: float) -> float:
        # No tax on buys in GE
        return 0.0

    # ---- Compatibility aliases (so env/baselines can call older names) ----
    def apply_sell_tax(self, gross_proceeds: float) -> float:
        """Return net proceeds after sell tax."""
        return self.sell_net_proceeds(gross_proceeds)

    def apply_buy_tax(self, cash_spent: float) -> float:
        """Return extra cost due to buy tax (0 for GE)."""
        return self.buy_cost(cash_spent)