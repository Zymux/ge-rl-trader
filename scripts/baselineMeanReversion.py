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

def compute_zscore(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    # Mean reversion signal: z = (mid - rolling_mean(mid)) / rolling_std(mid)
    g = df.groupby("item_id")["mid"]
    roll_mean = g.rolling(window=window, min_periods=window).mean().reset_index(level=0, drop=True)
    roll_std = g.rolling(window=window, min_periods=window).std().reset_index(level=0, drop=True)
    
    df["z"] = (df["mid"] - roll_mean) / roll_std
    
    return df

def run_strategy(
    df: pd.DataFrame,
    entry_z: float = -1.0,
    exit_z: float = 0.0,
    max_spread_pct: float = 0.05,
    fee_pct: float = 0.002,
) -> pd.DataFrame:
    cash = 1.0
    position_units = 0.0
    position_item = None
    held_last_price = np.nan
    equity_curve = []

    df = df.dropna(subset=["parsed_utc", "item_id", "mid", "z", "spread_pct"])
    df = df.sort_values("parsed_utc")

    for _, row in df.iterrows():
        item_id = int(row["item_id"])
        price = float(row["mid"])
        z = float(row["z"])
        spread_pct = float(row["spread_pct"])

        if position_item == item_id:
            held_last_price = price

        # mark-to-market
        if position_item is None:
            net_worth = cash
        else:
            net_worth = cash + position_units * held_last_price

        # liquidity filter for decisions
        if spread_pct > max_spread_pct:
            equity_curve.append(net_worth)
            continue

        # ENTRY: buy when price is "cheap" vs mean
        if position_item is None and z <= entry_z:
            cash_after_fee = cash * (1.0 - fee_pct)
            position_units = cash_after_fee / price
            cash = 0.0
            position_item = item_id
            held_last_price = price
            net_worth = cash + position_units * held_last_price

        # EXIT: sell when reverted back toward/above mean
        elif position_item == item_id and z >= exit_z and position_units > 0:
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
    df = compute_zscore(df, window=5)
    result = run_strategy(df, entry_z=-1.0, exit_z=0.0, max_spread_pct=0.05, fee_pct=0.002)

    out_plot = ASSETS_DIR / "baseline_mean_reversion_equity.png"

    plt.figure()
    plt.plot(result["parsed_utc"], result["equity"])
    plt.xlabel("Time")
    plt.ylabel("Equity (normalized)")
    plt.title("Baseline Mean Reversion Strategy (single-position)")
    plt.tight_layout()
    plt.savefig(out_plot, dpi=200)
    plt.close()

    print("Equity min/max:", result["equity"].min(), result["equity"].max())
    print(f"Final equity: {result['equity'].iloc[-1]:.4f}")
    print(f"Saved equity curve: {out_plot}")


if __name__ == "__main__":
    main()