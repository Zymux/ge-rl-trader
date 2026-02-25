---

## docs/design.md (documenting on the reasoning)

```md
# Design Notes - GE RL Trader

## Problem statement
Agent needs to allocate capital across trade opportunities in a simplified Grand Exchange market model.
Agent also acts in discrete time steps using only observable public signals (prices + engineered featueres + sentiment)

## Data sources
- OSRS Wiki endpoints for latest and/or timesteps snapshots.
- Eventually provide reddit threads, and patch notes for the sentiment extraction.

## Core assumptions
- Simulate fills at observed prices (plus transaction cost and the slippage if any)
- Inventory limited, cash limited.
- No order book
- Single-unit Trades
- Public price snapshots only
- Sell-side tax only

## Market mechanics modeled (Grand Exchange)

This project models OSRS Grand Exchange trading at an abstracted but realistic level:

- Agent observes only public price snapshots (high, low, mid)
- No access to internal order book depth or queue priority
- Trades execute at observed mid-price (proxy for matched offers)
- Single-unit, single-position trading (simplified capital allocation)
- Illiquid items filtered using spread percentage
- GE sell tax applied at 2% on SELL actions only (rounded down)
- No tax or fee applied on BUY actions
- No explicit buy-limit enforcement (documented simplification)
- BUY: execute at ask = mid * (1 + 0.5 * spread_pct)
  - No GE tax on buy
- SELL: execute at bid = mid * (1 − 0.5 * spread_pct)
  - Apply 2% GE tax on sell proceeds (integer rounding / floor)

## Action semantics (important for RL stability)
Action = (candidate_index, action_type)

- HOLD: no trade
- BUY: uses candidate_index to select an item; opens a single position using all cash
- SELL: ignores candidate_index and always liquidates the held item at current timestep

Rationale: the candidate set changes over time; SELL must not depend on whether the held item appears in top-K candidates.


## Environment definition for gymnasium
### State (Observation)
- Observation = [K candidates x (log(mid), return, spread) + (cash, holding)]

- Price features for each item such as; return, moving averages, volatility.
- Portfolio features such as; cash balance, inventory counts, current net worth
- Sentiment features such as; sentiment score, topic tags, confidence
- State features include log returns and rolling volatility, which become available as sufficient historical context is accumulated.

### Action space (MVP)
- Action = (candidate_index, HOLD/BUY/SELL)
- UPDATED:
    Action = (candidate_index, action_type)

    - HOLD: no trade
    - BUY: uses candidate_index to select an item; opens a single position using all cash
    - SELL: ignores candidate_index and always liquidates the held item at current timestep

    Rationale: the candidate set changes over time; SELL must not depend on whether the held item appears in top-K candidates.


- Choosing 1 item per step
- Choosing action such as BUY, SELL HOLD
- Set a fixed trade size such as 1 unit or fixed gp amount
Future upgrades:
- Variable sizing
- Multi-item actions
- Risk constraints

### Liquidity considerations
Exploratory analysis shows that spread % varies strongly with item price,
with low-priced items exhibiting extremely high relative spreads.
This implies that transaction cost modeling and liquidity filtering
are necessary to prevent degenerate trading behavior in the agent.

Spread-related features (spread_gp, spread_pct) will be included
in the observation space and/or reward function to penalize illiquid trades.

### Time-dependent features
Rolling statistics (e.g., volatility) require a minimum number of historical
snapshots per item. During early data collection, these features may be NaN
and are handled explicitly rather than imputed.

### Reward
MVP:
- reward = Δ(netWorth) per step - transactionCost
where netWorth = cash + sum (inventory_i * price_i)

reward_t = (net_worth_t − net_worth_{t−1}) / starting_cash
    This keeps reward scale stable for PPO and prevents large GP-denominated spikes.



Shaping:
- small penalty for excessive trades
- penalty for inventory concentration (risk)

### Episode termination
- fixed horizon such as N steps
- Bankrupt / no-cash or no-inventory end

## Baselines, mostly for credibility
- Buy and Hold (1 item)
- Momentum (buy if short MA > long MA)
- Mean reversion (buy if price drops below band)

## Evaluation plan
- Compare baselines vs RL on the same episodes
- Multiple Seeds
- Train/val/test split on time
- Ablation: RL without sentiment vs. RL with sentiment
- Baselines include transaction costs and liquidity filters to prevent illiquid-item artifacts

---

## Before you train: out-of-sample eval and learning checks

### 2) Evaluate properly (out-of-sample)

**Before trusting any run:** Train on one date range, then run evaluation on a **different** range (or different timeseries file) than training. If you only train and eval on the same window, PPO can “memorize the regime.” Use `--eval-ts <path_to_different_dates.csv>` with verifyLearning.py.

### 3) Three checks to confirm learning is real

After training, verify:

1. **Trade rate:** Policy is not always HOLD, and not constant churn. (e.g. executed BUY+SELL per episode in a sensible range.)
2. **Blocked actions:** `blocked_reason` frequency goes down over training. (Log blocked-reason counts in eval callbacks or rollouts; early training should have more blocks than late.)
3. **Equity curve:** Policy equity beats a baseline (buy-and-hold one item, or random valid policy).

The env already emits `executed`, `blocked_reason`, `realized_pnl`, `net_worth` in `info` — use these to compute all three.

### 4) Baselines (important)

Before trusting PPO results, compare against:

- **Always HOLD:** No trades; equity = 1.0 (minus any forced liquidation at episode end if you had a position from a previous run).
- **Random valid actions:** Buy only when flat (no position), sell only when in position; otherwise HOLD. Random choice among valid actions.
- **Simple heuristic:** e.g. buy when last return &gt; 0 and sell when &lt; 0 (momentum), or the reverse (mean reversion). Use one item (e.g. candidate 0) or aggregate.

Use the same eval episodes (and if possible out-of-sample window) for policy and baselines so the comparison is fair.

---

## What we need to do for #3 (minimal viable “GE-like” v2)

Current env is v1: instant fill at synthetic bid/ask. To learn actual OSRS flipping (place, wait, cancel, partial fill), we need the following.

### A) Change the agent’s actions (critical)

Right now actions are: **HOLD / BUY candidate / SELL current**.

For GE-like flipping, actions must become **order placement + cancellation**:

**Minimum action set:**
- **PLACE_BUY**: (item_id, price, qty)
- **PLACE_SELL**: (item_id, price, qty) — typically only if you hold inventory
- **CANCEL_ORDER**: cancel one of your active offers
- **HOLD**

Enforce slot limits (8/3) later; for now, keep “1 active buy + 1 active sell” to make it tractable.

### B) Add “resting offers” state to the environment

Add env state like:
- **active_buy**: {item_id, price, qty_remaining}
- **active_sell**: {item_id, price, qty_remaining}
- (optionally multiple slots)

This is the heart of GE flipping (orders persist).

### C) Add a matching / fill model (no full order book needed)

We can’t see real GE book depth, so approximate fills with a **probabilistic model** using only the time series:

At each timestep for an item:
- Compute a reference price (mid).
- Define a **fill probability** that increases when the offer is aggressive:
  - Place buy **above** mid → higher chance to fill.
  - Place sell **below** mid → higher chance to fill.
  - Undercut/overcut the wrong way → near-zero fill.

A simple fill curve:
- `p_fill = sigmoid(k * (mid - buy_price) / mid)` for buys (adjust sign by convention).
- Similar for sells.

Allow **partial fills**:
- `filled_qty ~ Binomial(qty_remaining, p_fill)` (or Poisson).

This gives RL the “wait vs improve price” tradeoff that makes flipping interesting.

### D) Execution price logic: “who posted first”

Wiki: execution price depends on which side existed first. In the sim (no book), approximate:
- If your **buy** fills: you pay your **limit price** (or slightly better with small improvement).
- If your **sell** fills: you receive your **limit price**.
- (Optional) small “price improvement” noise.

That’s enough for RL.

### E) Reward stays net-worth based (already good)

Current reward is equity log return. We already enforce:
- Integer units
- Buy limits with connected buckets
- Sell tax

So reward is fine. The missing piece is **how trades happen** (resting offers + fill model instead of instant fill).

