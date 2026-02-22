from __future__ import annotations

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

PROC_DIR = Path("data") / "processed"
ASSETS_DIR = Path("docs") / "assets"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

def newest_file(pattern: str) -> Path:
    files = sorted(PROC_DIR.glob(pattern))
    if not files:
        return FileNotFoundError(f"No files found in {PROC_DIR} matching {pattern}")
    return files[-1]


def load_latest_parsed() -> pd.DataFrame:
    path = newest_file("latest_parsed_*.csv")
    df = pd.read_csv(path)
    
    # making sure required columns exist
    required = {"item_id", "high", "low", "mid", "spread_gp", "spread_pct"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in parsed CSV: {missing}")
    return df, path


def filter_for_plotting(df: pd.DataFrame) -> pd.DataFrame:
    # Filtering the extreme/invalid values to avoid the chart looking broken
    # Keeping positive prices / removing absurd spreads where low is tiny / keeping mid with a reasonable range
    x = df.copy()
    x = x.dropna(subset=["high", "low", "mid", "spread_gp", "spread_pct"])
    
    x = x[(x["high"] > 0) & (x["low"] > 0) & (x["mid"] > 0)]
    x = x[x["high"] >= x["low"]]
    
    # removing the extreme spread% outlier, preventing noise
    x = x[x["spread_pct"] < 0.50] # 50% spread cap
    
    # keeping very tiny mids out to avoid 1gp items taking over
    x = x[x["mid"] >= 10]
    
    return x

def plot_top_spread_pct(df: pd.DataFrame, out_path: Path) -> None:
    top = df.sort_values("spread_pct", ascending=False).head(20)
    
    plt.figure()
    plt.bar([str(i) for i in top["item_id"]], top["spread_pct"])
    plt.xticks(rotation=90)
    plt.ylabel("Spread % ( (high-low)/mid )")
    plt.title("Top 20 Items by Spread % (filtered snapshot)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    
def plot_spread_vs_mid(df: pd.DataFrame, out_path: Path) -> None:
    plt.figure()
    plt.scatter(df["mid"], df["spread_pct"], s=10)
    plt.xscale("log")
    plt.xlabel("Mid price (log scale)")
    plt.ylabel("Spread %")
    plt.title("Spread % vs Mid Price (filtered snapshot)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    
def main() -> None:
    df, src_path = load_latest_parsed()
    snap_tag = Path(src_path).stem.replace("latest_parsed_", "")
    
    df_f = filter_for_plotting(df)
    
    out1 = ASSETS_DIR / f"snapshot_top_spread_pct{snap_tag}.png"
    out2 = ASSETS_DIR / f"snapshot_spread_vs_mid_{snap_tag}.png"
    
    plot_top_spread_pct(df_f, out1)
    plot_spread_vs_mid(df_f, out2)
    
    print(f"Source: {src_path}")
    print(f"Filtered rows: {len(df_f):,} / {len(df):,}")
    print(f"Saved: {out1}")
    print(f"Saved: {out2}")
    
if __name__ == "__main__":
    main()