from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd

RAW_DIR = Path("data") / "raw"
PROC_DIR = Path("data") / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)

def newest_file(pattern: str) -> Path:
    # Picking the most recent snapshot. pullPrices.py to write timestamped JSON -> parseLatest.py to grab the latest
    
    files = sorted(RAW_DIR.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files found in {RAW_DIR} matching the {pattern}")
    return files[-1]



def utc_now_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

def main() -> None:
    # loading the newest snapshot || to be able to pricePull multiple times always parsed with newest data.
    latestPath = newest_file("latest_*.json")
    payload = json.loads(latestPath.read_text(encoding="utf-8"))
    
    # API structure expected is {"data": {item_id}: {high, low, ...}, ...}, ...} || later joining with metadata table like names, items, catoegories.
    if "data" not in payload or not isinstance(payload["data"], dict):
        raise ValueError("Unexpected JSON structure: expected top-level key'data' as dict")
    
    # converting dict-of-dicts into table
    rows = []
    for itemIdStr, stats, in payload["data"].items():
        # item_id is the key but keep as int for joins
        try:
            item_id = int(itemIdStr)
        except ValueError:
            continue
        
        # stats is in dict {"high": 123, "highTime": ..., "low":...}
        row = {"item_id": item_id}
        if isinstance(stats, dict):
            row.update(stats)
        rows.append(row)
        
    df = pd.DataFrame(rows)
    
    # Cleaning / typing but not over-normalizing -> converting numeric columns if present. || for RL need to maintain consistent table, repeatable pipeline, and features to feed to environment
    for col in ["high", "low", "highTime", "lowTime"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            
    # adding metadata of where snapshot came from
    df["snapshot_file"] = latestPath.name
    df["parsed_utc"] = utc_now_tag()
    
    # deriving columsn that are useful immediately for analytics/backtesting || these will be used in reward calculations, transaction costs, and policy behavior later.
    # mid price for simple mark-to-market proxy
    if "high" in df.columns and "low" in df.columns:
        df["mid"] = (df["high"] + df["low"]) / 2.0
        
        # spreading proxy in gp and %, trading feature
        df["spread_gp"] = df["high"] - df["low"]
        df["spread_pct"] = df["spread_gp"] / df["mid"]
        
        
    # saving processed outputs || setup to assist with debugging in teh future and performance.
    out_tag = df["parsed_utc"].iloc[0]
    out_csv = PROC_DIR / f"latest_parsed_{out_tag}.csv"
    out_parquet = PROC_DIR / f"latest_parsed_{out_tag}.parquet"
    
    df.to_csv(out_csv, index=False)
    
    # parquet can be used for speed later
    try:
        df.to_parquet(out_parquet, index=False)
        parquet_ok = True
    except Exception:
        parquet_ok = False
        
    # printing a summary to validate
    print(f"Laoded: {latestPath}")
    print(f"Rows: {len(df):,} | Cols: {len(df.columns)}")
    print(f"Saved CSV: {out_csv}")
    if parquet_ok:
        print(f"Saved Parquet: {out_parquet}")
    else:
        print("Parquet not saved, missing the parquet engine")
        
    # showing top 10 of widest spread % items (low liquidity / risk)
    if "spread_pct" in df.columns:
        top = df.sort_values("spread_pct", ascending=False).head(10)[
            ["item_id", "high", "low", "mid", "spread_gp", "spread_pct"]
        ]
        print("\nTop 10 by spread_pct (validation check):")
        print(top.to_string(index=False))
        
        
if __name__ == "__main__":
    main()