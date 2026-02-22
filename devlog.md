# Devlog GE Trade - RL + LLM Signals Project

## 2026-02-21 - Project Structure
- Created repo structure, src/, scripts/, docs/, data/
- Set up the Python environment and requirements
- Added the first price pull to save raw JSON snapshots.

## 2026-02-22 - parseLatest.py:
- Parsed the latest snapshot into a dataframe
- Created a first plot: top movers / volume leaders
- Implemented caching + timeseries collection
- Generated the initial market microstructure plots:
    - Top items by spread %
    - Spread % vs mid price (log scale)
- Observed strong inverse relationship between item price and spread %
- High spread items tended to be low-price and likely illiquid
- Concluded that spread % should be treated as a risk / cost feature, not a signal for profit.