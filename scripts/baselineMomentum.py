from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

TS_PATH = Path("data") / "timeseries" / "ge_item_timeseries.csv"
ASSETS_DIR = Path("docs") / "assets"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

def load_ts() -> pd.DataFrame:
    df = pd.read_csv(TS_PATH, parse_dates=["parsed_utc"])
    df = df.sort_values(["item_id", "parsed_utc"])
    return df

def compute_momentum(df: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    df["mom"] = (
        df.groupby('item_id')["log_return"]
        .rolling(window=window, min_periods=window)
        .sum()
        .reset_index(level=0, drop=True)
    )
    return df

def run_strategy(df: pd.DataFrame, threshold: float = 0.01, max_spread_pct: float = 0.05, fee_pct: float = 0.002) -> pd.DataFrame:
    # starting position / buy when momentum > threshold / exit when moment < threshold / filters illiquid items by spread_pct and applies the small trx fee/slippage on buys/sells
    cash = 1.0
    position_units = 0.0
    position_item = None
    held_last_price = np.nan

    equity_curve = []

    df = df.dropna(subset=["parsed_utc", "item_id", "mid", "mom", "spread_pct"])
    df = df.sort_values("parsed_utc")

    for _, row in df.iterrows():
        item_id = int(row["item_id"])
        price = float(row["mid"])
        mom = float(row["mom"])
        spread_pct = float(row["spread_pct"])

        # Update last seen price if this row corresponds to our held item
        if position_item == item_id:
            held_last_price = price

        # Compute equity using the last known held price (if holding)
        if position_item is None:
            net_worth = cash
        else:
            if np.isnan(held_last_price):
                # fallback: if we somehow never saw a price for held item, treat as cash (should be rare)
                net_worth = cash
            else:
                net_worth = cash + position_units * held_last_price

        # Skip illiquid items for ENTRY/EXIT decisions (but still record equity)
        if spread_pct > max_spread_pct:
            equity_curve.append(net_worth)
            continue

        # BUY
        if position_item is None and mom > threshold:
            cash_after_fee = cash * (1.0 - fee_pct)
            position_units = cash_after_fee / price
            cash = 0.0
            position_item = item_id
            held_last_price = price  # initialize held price at entry

            # Recompute equity immediately after trade
            net_worth = cash + position_units * held_last_price

        # SELL
        elif position_item == item_id and mom < -threshold and position_units > 0:
            proceeds = position_units * price
            cash = proceeds * (1.0 - fee_pct)
            position_units = 0.0
            position_item = None
            held_last_price = np.nan
            net_worth = cash

        equity_curve.append(net_worth)

    out = df.iloc[:len(equity_curve)].copy()
    out["equity"] = equity_curve
    return out
            
def main():
    df = load_ts()
    df = compute_momnetum(df, window=3)
    result = run_strategy(df, threshold=0.01, max_spread_pct=0.05, fee_pct=0.002)    
    print("Equity min/max:", result["equity"].min(), result["equity"].max())
    print("Final equity:", result["equity"].iloc[-1])

    out_plot = ASSETS_DIR / "baseline_momentum_equity.png"
    
    plt.figure()
    plt.plot(result["parsed_utc"], result["equity"])
    plt.xlabel("Time")
    plt.ylabel("Equity (normalized)")
    plt.title("Baseline Momentum Strategy (single-position)")
    plt.tight_layout()
    plt.savefig(out_plot, dpi=200)
    plt.close()

    print(f"Saved equity curve: {out_plot}")
    print(f"Final equity: {result['equity'].iloc[-1]:.4f}")

if __name__ == "__main__":
    main()