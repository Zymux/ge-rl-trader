# Devlog GE Trade - RL + LLM Signals Project

## 2026-02-21 - Project Structure
- Created repo structure, src/, scripts/, docs/, data/
- Set up the Python environment and requirements
- Added the first price pull to save raw JSON snapshots.

## 2026-02-22 - 
### parseLatest.py:
- Parsed the latest snapshot into a dataframe
- Created a first plot: top movers / volume leaders
- Implemented caching + timeseries collection
- Generated the initial market microstructure plots:
    - Top items by spread %
    - Spread % vs mid price (log scale)
- Observed strong inverse relationship between item price and spread %
- High spread items tended to be low-price and likely illiquid
- Concluded that spread % should be treated as a risk / cost feature, not a signal for profit.

### Time series edge cases & feature validation
- Built a per-item time series from multiple GE snapshots (10+)
- Observed expected NaNs in log returns and rolling vaoltility due to limited history
- Fixed log-return computation by masking non-positive prices before applying log
- Confirmed that rolling volatility features require more snapshots to populate
- Decided to keep NaNs at this stage rather than forward-fill or drop rows
- Verified successive GER snapshot payloads differ using different content hashes / confirmed API responses are not cached or duplicated.
- Dataset suitable for baseline trading strategies

### Momentum baseline validated
- Implemented a single-position momentume baseline with spread filtering + transaction fee
- Fixed portfolio valuation bug by tracking last seen price of held item
- Generated equity curve, short-horizon run ended at around ~0.974 (costs dominated)
- Mean reversion outperformed momentum over short sample window


## Baseline freeze
- Updated all baseline strategies to apply the GE sell tax (2%) and SELL only
- Converted `src/` and `src/` into Python packages and switched to module-based execution (`python -m`)
- Regenerated baseline equity curves and comparison plot under consistent GE rules