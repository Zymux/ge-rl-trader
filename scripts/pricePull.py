import json
from pathlib import Path
import requests # for HTTP api requests
from datetime import datetime, timezone

OUT_DIR = Path("data") / "raw"
OUT_DIR.mkdir(parents=True, exist_ok=True)

URL = "https://prices.runescape.wiki/api/v1/osrs/latest"

def main():
    r = requests.get(URL, headers={"User-Agent": "ge-rl-trader"}, timeout=30)
    r.raise_for_status()
    data = r.json()
    
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = OUT_DIR / f"latest_{ts}.json"
    out_path.write_text(json.dumps(data, indent=2))
    print(f"Saved: {out_path} | keys: {list(data.keys())}")

    
if __name__ == "__main__":
    main()