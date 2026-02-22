from __future__ import annotations

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ASSETS_DIR = Path("docs") / "assets"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

MOM_PATH = ASSETS_DIR / "baseline_momentum_equity.png"  # only for reference; we recompute curves below
MR_PATH = ASSETS_DIR / "baseline_mean_reversion_equity.png"

TS_PATH = Path("data") / "timeseries" / "ge_item_timeseries.csv"


# --- import the two strategy modules (so we reuse the exact logic you ran) ---
from scripts import baselineMomentum as mom
from scripts import baselineMeanReversion as mr


def main() -> None:
    df = pd.read_csv(TS_PATH, parse_dates=["parsed_utc"]).sort_values(["item_id", "parsed_utc"])

    # recompute both curves from the same underlying df
    df_m = mom.compute_momentum(df.copy(), window=3)
    res_m = mom.run_strategy(df_m, threshold=0.01, max_spread_pct=0.05, fee_pct=0.002)

    df_r = mr.compute_zscore(df.copy(), window=5)
    res_r = mr.run_strategy(df_r, entry_z=-1.0, exit_z=0.0, max_spread_pct=0.05, fee_pct=0.002)

    out_plot = ASSETS_DIR / "baselines_compare_equity.png"

    plt.figure()
    plt.plot(res_m["parsed_utc"], res_m["equity"], label="Momentum")
    plt.plot(res_r["parsed_utc"], res_r["equity"], label="Mean Reversion")
    plt.xlabel("Time")
    plt.ylabel("Equity (normalized)")
    plt.title("Baseline Strategies Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_plot, dpi=200)
    plt.close()

    print(f"Saved comparison plot: {out_plot}")
    print(f"Momentum final: {res_m['equity'].iloc[-1]:.4f}")
    print(f"MeanRev  final: {res_r['equity'].iloc[-1]:.4f}")


if __name__ == "__main__":
    main()