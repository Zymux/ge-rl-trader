# Split timeseries into train and eval by date (eval = later dates).
# Run once before training so you can train on _train.csv and eval on _eval.csv.
#
# Usage:
#   python -m scripts.splitTimeseries [--input path] [--eval-frac 0.2]
#
# Creates:
#   data/timeseries/ge_item_timeseries_train.csv   (earlier dates)
#   data/timeseries/ge_item_timeseries_eval.csv    (later dates)
#
# Then: train with ts_path=train CSV, and run:
#   python -m scripts.verifyLearning --eval-ts data/timeseries/ge_item_timeseries_eval.csv

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

TS_DIR = Path("data") / "timeseries"
DEFAULT_INPUT = TS_DIR / "ge_item_timeseries.csv"
TRAIN_OUT = TS_DIR / "ge_item_timeseries_train.csv"
EVAL_OUT = TS_DIR / "ge_item_timeseries_eval.csv"


def main():
    parser = argparse.ArgumentParser(description="Split timeseries into train (earlier) and eval (later) by date.")
    parser.add_argument("--input", type=str, default=str(DEFAULT_INPUT), help="Input timeseries CSV")
    parser.add_argument("--eval-frac", type=float, default=0.2, help="Fraction of unique dates for eval (later); rest for train")
    args = parser.parse_args()

    path = Path(args.input)
    if not path.exists():
        print(f"Input not found: {path}")
        return
    eval_frac = max(0.05, min(0.5, args.eval_frac))

    df = pd.read_csv(path, parse_dates=["parsed_utc"])
    if "parsed_utc" not in df.columns:
        print("CSV must have column parsed_utc")
        return
    df = df.dropna(subset=["parsed_utc", "item_id", "mid"])
    dates = sorted(df["parsed_utc"].dt.normalize().unique())
    n_dates = len(dates)
    if n_dates < 2:
        print("Need at least 2 distinct dates to split")
        return

    n_eval_dates = max(1, int(n_dates * eval_frac))
    n_train_dates = n_dates - n_eval_dates
    train_dates = set(dates[:n_train_dates])
    eval_dates = set(dates[n_train_dates:])

    df["_date"] = df["parsed_utc"].dt.normalize()
    train_df = df[df["_date"].isin(train_dates)].drop(columns=["_date"])
    eval_df = df[df["_date"].isin(eval_dates)].drop(columns=["_date"])

    TS_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(TRAIN_OUT, index=False)
    eval_df.to_csv(EVAL_OUT, index=False)

    n_train_ts = train_df["parsed_utc"].nunique()
    n_eval_ts = eval_df["parsed_utc"].nunique()
    min_ts = 76  # episode_len 75 + 1
    print(f"Split by date:")
    print(f"  Train: {dates[0]} .. {dates[n_train_dates-1]}  ({n_train_dates} dates, {n_train_ts} timestamps, {len(train_df):,} rows) -> {TRAIN_OUT}")
    print(f"  Eval:  {dates[n_train_dates]} .. {dates[-1]}  ({n_eval_dates} dates, {n_eval_ts} timestamps, {len(eval_df):,} rows) -> {EVAL_OUT}")
    if n_train_ts < min_ts or n_eval_ts < min_ts:
        print(f"  WARNING: Env needs >= {min_ts} timestamps per file. Train will fall back to full timeseries if train split is used.")
    print()
    print("Next steps:")
    print("  1) Train using ts_path='data/timeseries/ge_item_timeseries_train.csv'")
    print("  2) Run: python -m scripts.verifyLearning --eval-ts data/timeseries/ge_item_timeseries_eval.csv")


if __name__ == "__main__":
    main()
