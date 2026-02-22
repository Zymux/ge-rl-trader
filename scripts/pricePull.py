from __future__ import annotations

import json
from pathlib import Path
import requests # for HTTP api requests
from datetime import datetime, timezone

RAW_DIR = Path("data") / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

URL = "https://prices.runescape.wiki/api/v1/osrs/latest"
USER_AGENT = "ge-rl-trader"

def utc_now_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

def main():
    ts = utc_now_tag()
    out_path = RAW_DIR / f"latest_{ts}.json"
    
    if out_path.exists():
        raise FileExistsError(f"Snapshot already exists: {out_path}")
    
    r = requests.get(
        URL,
        headers={"User-Agent": USER_AGENT},
        timeout=30,
    )
    r.raise_for_status()
    payload = r.json()
    
    # embeding the metadata so downstream steps dont rely on filenames || for later on, when reprocessing old file,ds moving data, and loading from SR -> to not lose timing context
    wrapped = {
        "meta": {
            "pulled_utc": ts,
            "source": URL,
        },
        "data": payload.get("data", {}),
    }
    
    out_path.write_text(json.dumps(wrapped, indent=2))
    print(f"Saved snapshot: {out_path}")
    print(f"Items: {len(wrapped['data'])}")
    
    

    
if __name__ == "__main__":
    main()