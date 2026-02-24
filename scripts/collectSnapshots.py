# Created a new script to collect snapshots to avoid powershell/cmd issues.

import time
import subprocess
import sys
from datetime import datetime, timezone

# 3 hours = 180 minutes
RUN_MINUTES = 180
SLEEP_SECONDS = 60

def main():
    end_time = time.time() + RUN_MINUTES * 60
    i = 0

    while time.time() < end_time:
        i += 1
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
        print(f"\n[{i}] {ts} Pull + Parse")

        subprocess.check_call([sys.executable, "scripts/pricePull.py"])
        subprocess.check_call([sys.executable, "scripts/parseLatest.py"])

        time.sleep(SLEEP_SECONDS)

    print("\nDone collecting snapshots.")

if __name__ == "__main__":
    main()