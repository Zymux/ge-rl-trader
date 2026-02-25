from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np

PROC_DIR = Path("data") / "processed"
TS_DIR = Path("data") / "timeseries"
TS_DIR.mkdir(parents=True, exist_ok=True)

def load_all_parsed() -> pd.DataFrame:
    files = sorted(PROC_DIR.glob("latest_parsed_*.csv"))
    if not files:
        raise FileNotFoundError("No parsed CSV found.")
    
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        dfs.append(df)
        
        
    all_df = pd.concat(dfs, ignore_index=True)
    return all_df

def build_timeseries(df: pd.DataFrame) -> pd.DataFrame:
    # validating correct types
    df["parsed_utc"] = pd.to_datetime(df["parsed_utc"], utc=True)
    
    # sorting for time-series operation
    df = df.sort_values(["item_id", "parsed_utc"]) # ordering prices correctly since timeseries depends on order. sorting by item_id so grouped together and parsed so time increases correctly)
    
    # computing log returns per item -> for each "item_id" take "mid" column values and apply this function, so x = all mid val for item_id and lambda tells pandas what to do with vals.
    df["log_mid"] = df.groupby("item_id")["mid"].transform(
        lambda x: np.log(x.where(x > 0)) # cant take log of 0 or negative #s
    )
    
    df ["log_return"] = df.groupby("item_id")["log_mid"].diff() # chosen because price changes are multiplicative, and log returns are scale-invariant / additive over time. RL learns better from normalized signals
    
    # vol_5 = rolling volatility (std of log returns over 5 steps). Not volume/liquidity — for that you'd need GE DB volume.
    df["vol_5"] = (
        df.groupby("item_id")["log_return"]
        .rolling(window=5, min_periods=3) 
        .std()
        .reset_index(level=0, drop=True)
    )
    
    return df


def main():
    raw = load_all_parsed()
    ts = build_timeseries(raw)
    
    out = TS_DIR / "ge_item_timeseries.csv"
    ts.to_csv(out, index=False)
    
    print(f"Built time series:")
    print(f"Rows: {len(ts):,}")
    print(f"Items: {ts['item_id'].nunique():,}")
    print(f"Saved: {out}")
    
    # validity check
    print("\nExample rows:")
    example = ts[["parsed_utc", "item_id", "mid", "log_return", "vol_5"]]
    print(example.head(10).to_string(index=False))
    
    print("\nNon-null counts:")
    print(example.notna().sum())

if __name__ == "__main__":
    main()