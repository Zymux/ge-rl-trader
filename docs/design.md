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


## Environment definition for gymnasium
### State (Observation)
- Price features for each item such as; return, moving averages, volatility.
- Portfolio features such as; cash balance, inventory counts, current net worth
- Sentiment features such as; sentiment score, topic tags, confidence
- State features include log returns and rolling volatility, which become available as sufficient historical context is accumulated.

### Action space (MVP)
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

