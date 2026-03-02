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


### Baseline freeze
- Updated all baseline strategies to apply the GE sell tax (2%) and SELL only
- Converted `src/` and `src/` into Python packages and switched to module-based execution (`python -m`)
- Regenerated baseline equity curves and comparison plot under consistent GE rules


## 2026-02-23 - 
### GE Execution + env validation + PPO readiness
- Created `geEnv.py` and `testEnv.py`to implement Gym environemnt using GE rules, random policy test passes.

### Environment (GE Simulator)
- Implemented the GE-simulator execution costs:
    - BUY executing at ask derived from mid + half-spread
    - SELL executing at bid derived from mid - half-spread
    - SELL applying 2% GE tax through GERules
- Normalized reward to be relative to starting_cash to stabilize PPO training.
- Identified and fixed RL stability issues:
    - SELL shoudl always liquidate the held item (not depend on top-K candidate list)

### Validation
- Forced buy/sell test confirmed tax behavior:
    - Starting cash of 10,000,000 (gp) after sell cash it displayed 9,800,000 (gp) (2% tax hit)
- Confirmed pipeline produces time-varying snapshots and a useable timeseries CSV

### Data pipeline / baselines
- Built minute-level snapshot collection (pricePull + parseLatest)
- Generated baseline equity curves

## 2026-02-24 -
### PPO model evaluation and certain bugs in trading logic/GE simulator
- Created `evalPolicyEquity.py` and `evalPPO.py`to evaluate the trained PPO model.
- `evalPolicyEquity.py` analysis over 50 episodes:
    - Displayed that dataset was fine since it dispalyed 191 timesteps across different times
    - Within Episode 0, environment/episode produced zero net-worth movement __min=max=mean=10_000_000__
    - 37~ trades per episode / normalized equity remained 1. throughout all episodes.
    - Conclusion: 
        - Not holding across time, policy alternates BUY then SELL so quickly the expusre is most likely 0 most of the time
        - Environment logic effectively flattened the position at every step
        - Action execution was "all-in" then "all-out" within the same timestep
        - Printed Episode 0 for the fraction of steps where inventory != 0, 
- Confirmed MTM (Market-to-Market) accouintign is correct
- 
- Fixed equity only updating on episode end (but positions remianing flat) -> Δequity = 0 every episode
- Fixed transaction costs canceling out gains -> Net equity never moved
- Fixed policy evaluation using dummy / frozen price -> RL cant extract value from static prices


## 2026-02-25 -
### Fixing GE Environment/Simulation mechanics
- PPO results appeared "too good", which led to a more indepth analysis of the GE.
- Fixed "all-in" buying -> This enocuraged degenerate behavior
- Fixed net-worth evaluation bug caused by top-K candidate filtering
- Added full-universe snapshot for valuation and liquidation
- Added carry-forward pricing(last_pos_price)
- Confirmed PPO stability after env fixes
- Ran full-episode equity evaluation
- Verified strong out-of-sample performance
- Explicitly mapped missing GE mechanics to OSRS Wiki

### v2 baseline lock (resting orders, partial fills, sell sanity)
- **V2 microstructure**: Resting buy/sell offers, probabilistic partial fills, discrete price offsets; single active buy + single active sell; CANCEL_BUY / CANCEL_SELL (v2.1). No instant mid execution.
- **Valuation**: Single source of truth via `_get_position_mid_from_slice()`; full slice for MTM; off-by-one fixed (worth_ts = step_ts). acted_mid now matches acted_item_id (BUY = candidate mid, SELL = held-item mid from same slice). Valuation sanity treats large reprices as MTM/regime change, not bugs.
- **Sell sanity**: evalPolicyEquity counts requested SELL, executed SELL (fills), rejected (blocked) with `sell_blocked_*` histogram. Blocked overrides (no position, active order) set consistent prefixes and small feasibility penalties.
- **Blocked-sell penalties**: `sell_blocked_active_order` → -0.001; `sell_blocked_no_position` → -0.003. Reduces SELL spam; blocked sells dropped from ~65% to ~13% of requested SELL.
- **Results (post-penalties, 50 eps)**: requested SELL 697, executed SELL 181, rejected 91 (sell_blocked_active_order 88, sell_blocked_no_position 3). Valuation (pos_mid vs acted_mid): OK. Final equity mean 1.019, std 0.093, min/max 0.86 / 1.33 — realistic flipping profile. Locked as good v2 baseline (potentially v2.2).


## PENDING: 
- Collecting 3+ hours of snapshots to increase unique timestamps for longer episodes
- Updated GE to have real OSRS 4-hour buy limits (per item)
- Modeled the GE offer mechanics such as agent learning actual GE behavior of margin checking or undercutting/overcutting.
- Retraining PPO with stbale env (sell-held-item semantics + normalized reward)
- Add evaluation script for PPO vs Momentum vs MeanReversion (same episode windows) + metrics table
- Integrating LLM sentiment after RL work