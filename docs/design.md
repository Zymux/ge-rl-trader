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

## Environment definition for gymnasium
### State (Observation)
- Price features for each item such as; return, moving averages, volatility.
- Portfolio features such as; cash balance, inventory counts, current net worth
- Sentiment features such as; sentiment score, topic tags, confidence

### Action space (MVP)
- Choosing 1 item per step
- Choosing action such as BUY, SELL HOLD
- Set a fixed trade size such as 1 unit or fixed gp amount
Future upgrades:
- Variable sizing
- Multi-item actions
- Risk constraints

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

