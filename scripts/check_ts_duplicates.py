# Step 3: Check for duplicate (parsed_utc, item_id) in timeseries CSV.
# If duplicates exist, valuation can pick the wrong row (iloc[0]) and show nonsense.
# Usage: python -m scripts.check_ts_duplicates [--ts-path path]

import argparse
import pandas as pd

DEFAULT_TS = "data/timeseries/ge_item_timeseries.csv"


def main():
    p = argparse.ArgumentParser(description="Report duplicate (parsed_utc, item_id) in timeseries.")
    p.add_argument("--ts-path", default=DEFAULT_TS, help="Timeseries CSV path")
    args = p.parse_args()

    df = pd.read_csv(args.ts_path, parse_dates=["parsed_utc"])
    df = df.dropna(subset=["parsed_utc", "item_id", "mid"])
    df["item_id"] = df["item_id"].astype(int)

    dup = df.groupby(["parsed_utc", "item_id"]).size().reset_index(name="n")
    dup = dup[dup["n"] > 1].sort_values("n", ascending=False)

    print("Duplicate (parsed_utc, item_id) pairs: %d" % len(dup))
    if len(dup) > 0:
        print(dup.head(20).to_string())
        print("\nFix: deduplicate in buildTimeseries (e.g. one row per (parsed_utc, item_id)).")
    else:
        print("None found.")


if __name__ == "__main__":
    main()
