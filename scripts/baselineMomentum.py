from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.geRules import GERules  

TS_PATH = Path("data") / "timeseries" / "ge_item_timeseries.csv"
ASSETS_DIR = Path("docs") / "assets"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)


def load_ts() -> pd.DataFrame:
    df = pd.read_csv(TS_PATH, parse_dates=["parsed_utc"])
    df = df.sort_values(["item_id", "parsed_utc"])
    return df


def compute_momentum(df: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    df["mom"] = (
        df.groupby("item_id")["log_return"]
        .rolling(window=window, min_periods=window)
        .sum()
        .reset_index(level=0, drop=True)
    )
    return df


def run_strategy(
    df: pd.DataFrame,
    threshold: float = 0.01,
    max_spread_pct: float = 0.05,
    rules: GERules = GERules(),
) -> pd.DataFrame:
    """
    Single-position momentum baseline.

    Decisions:
      - BUY if mom > threshold (no tax on buy)
      - SELL if mom < -threshold (apply GE sell tax)

    Reality knobs:
      - max_spread_pct filters illiquid rows
      - mark-to-market uses last seen price for held item
    """
    cash = 1.0
    position_units = 0.0
    position_item: int | None = None
    held_last_price = np.nan

    equity_curve: list[float] = []

    df = df.dropna(subset=["parsed_utc", "item_id", "mid", "mom", "spread_pct"])
    df = df.sort_values("parsed_utc")

    for _, row in df.iterrows():
        item_id = int(row["item_id"])
        price = float(row["mid"])
        mom = float(row["mom"])
        spread_pct = float(row["spread_pct"])

        # update held mark-to-market price when we see the held item again
        if position_item == item_id:
            held_last_price = price

        # mark-to-market equity
        if position_item is None:
            net_worth = cash
        else:
            net_worth = cash + position_units * held_last_price if not np.isnan(held_last_price) else cash

        # liquidity filter for entry/exit (still record equity)
        if spread_pct > max_spread_pct:
            equity_curve.append(net_worth)
            continue

        # BUY (no tax on buy)
        if position_item is None and mom > threshold:
            position_units = cash / price
            cash = 0.0
            position_item = item_id
            held_last_price = price
            net_worth = cash + position_units * held_last_price

        # SELL (GE sell tax)
        elif position_item == item_id and mom < -threshold and position_units > 0:
            gross = position_units * price
            cash = rules.sell_net_proceeds(gross)
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
    df = compute_momentum(df, window=3)

    rules = GERules()  # sell tax on SELL only
    result = run_strategy(df, threshold=0.01, max_spread_pct=0.05, rules=rules)

    print("Equity min/max:", result["equity"].min(), result["equity"].max())
    print("Final equity:", float(result["equity"].iloc[-1]))

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