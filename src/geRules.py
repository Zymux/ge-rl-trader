from __future__ import annotations
from dataclasses import dataclass
import math

@dataclass(frozen=True)
class GERules:
    # Simplifying the GE rules for trading: GE Tax is applied on sell (2%) / Rounded down in coins / Optional cap (5,000,000 GP per item)
    
    sell_tax_rate: float = 0.02
    apply_tax_cap: bool = False
    tax_cap_gp: int = 5_000_000
    
    def sell_tax(self, gross_proceeds: float) -> float:
        tax = self.sell_tax_rate * gross_proceeds
        # GE rounding down to whole coins
        tax = math.floor(tax)
        if self.apply_tax_cap:
            tax = min(tax, self.tax_cap_gp)
        return float(tax)
    

    def buy_cost(self, cash_spent: float) -> float:
        # No tax on buys in GE
        return 0.0
    
    def sell_net_proceeds(self, gross_proceeds: float) -> float:
        return gross_proceeds - self.sell_tax(gross_proceeds)