# Step 2: Print raw CSV rows for exact timestamp(s) + item_id to verify mid values.
# Usage: python -m scripts.check_ts_item_rows [--ts "2026-02-22 20:41:40+00:00" "2026-02-22 20:42:41+00:00"] [--item 10322] [--ts-path path]

import argparse
import pandas as pd

DEFAULT_TS = "data/timeseries/ge_item_timeseries.csv"
DEFAULT_ITEM = 10322
DEFAULT_TIMES = ["2026-02-22 20:41:40+00:00", "2026-02-22 20:42:41+00:00"]


def main():
    p = argparse.ArgumentParser(description="Inspect timeseries rows for (parsed_utc, item_id).")
    p.add_argument("--ts-path", default=DEFAULT_TS, help="Timeseries CSV path")
    p.add_argument("--item", type=int, default=DEFAULT_ITEM, help="item_id to filter")
    p.add_argument("--ts", nargs="+", default=DEFAULT_TIMES, help="Timestamp(s) to inspect")
    args = p.parse_args()

    df = pd.read_csv(args.ts_path, parse_dates=["parsed_utc"])
    df = df.dropna(subset=["parsed_utc", "item_id", "mid"])
    df["item_id"] = df["item_id"].astype(int)

    for t in args.ts:
        ts = pd.Timestamp(t)
        sub = df[(df["parsed_utc"] == ts) & (df["item_id"] == args.item)]
        print("\n", ts, "item_id=%d rows:" % args.item, len(sub))
        if len(sub) > 0:
            cols = [c for c in ["parsed_utc", "item_id", "mid", "spread_pct", "vol_5"] if c in sub.columns]
            print(sub[cols].head(20).to_string())
        else:
            print("  (no rows)")


if __name__ == "__main__":
    main()
