# src/geEnv.py
"""
GE-style trading environment for RL.

What real GE does (we do *not* simulate this):
  - Resting offers (buy/sell orders sit in the book until matched).
  - Matching when buy_price >= sell_price; execution price and time depend on which side
    posted first; same-price orders are loosely time-priority.
  - Partial fills over time. Flipping = undercut/overcut and wait.
  See: Grand Exchange - OSRS Wiki (and wiki PDF).

What this env does (v1 — instant fill):
  - "Market order at synthetic bid/ask each step": BUY at ask, SELL at bid; no resting orders,
    no order book, no time priority, no partial fills. RL will not learn actual OSRS flipping
    behavior (place, wait, cancel, partial fill) with this model.

To learn actual OSRS flipping (v2 / #3), the env must move from the above to:
  - Place limit offers (buy/sell at chosen price) instead of instant execution.
  - Wait for fills: matching when buy_price >= sell_price; time priority.
  - Manage cancellations (cancel stale offers).
  - Handle partial fills over time.
  That implies: resting orders, an order book (or simplified fill model), and step-level
  fill logic instead of instant fill. Until then, v1 remains "market order at synthetic spread."

  Minimal viable v2 design (actions, resting offers, fill model, execution price): see docs/design.md § "What we need to do for #3".
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from src.reward import equity_log_reward

from .geRules import GERules


@dataclass
class GEEnvConfig:
    ts_path: str = "data/timeseries/ge_item_timeseries.csv"
    starting_cash: float = 10_000_000.0

    # episode settings
    episode_len: int = 75          # how many timestamps per episode
    max_candidates: int = 25       # top-N items considered per step (by liquidity filter)

    # trading / liquidity filters
    max_spread_pct: float = 0.05   # skip illiquid rows
    min_mid: float = 1.0           # skip junk
    min_spread_pct_floor: float = 0.005  # min spread for bid/ask (e.g. 0.5%) when spread_pct missing or tiny
    fee_sell_tax: float = 0.02     # 2% GE tax on sells ONLY (buy has no tax)
    fee_sell_tax_cap_gp: int = 5_000_000  # GE tax cap per trade
    apply_sell_tax_cap: bool = True

    # buy limits (GE 4h window): required CSV with item_id, limit; optional bucket_id for connected limits
    # (e.g. potion doses share one bucket: same bucket_id → buying any of them consumes the same limit)
    buy_limit_csv: str = "data/buy_limits.csv"
    buy_limit_window_seconds: int = 4 * 3600  # reset by real elapsed time (parsed_utc), not step count

    # position sizing: cap fraction of cash per BUY (stops "all-in" exploit)
    max_cash_fraction_per_trade: float = 0.25  # e.g. 0.25 = spend at most 25% of cash per buy

    # v2.0/2.1 action space (limit-order microstructure):
    #   [candidate_index, act_type, price_offset_idx, qty_idx]
    #   act_type: 0=HOLD, 1=PLACE_BUY, 2=PLACE_SELL, 3=CANCEL_BUY, 4=CANCEL_SELL
    #   price_offset_idx ∈ {0..6} → offsets {-3..+3} * price_offset_pct_step around mid
    #   qty_idx ∈ {0..2} → fractions {0.25, 0.5, 1.0} of max_cash_fraction_per_trade / position_units
    price_offset_pct_step: float = 0.002  # ~0.2% per offset step around mid
    fill_slope: float = 5.0              # sigmoid slope for probabilistic fills
    seed: int = 123


class GEEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, cfg: Optional[GEEnvConfig] = None):
        super().__init__()
        self.cfg = cfg or GEEnvConfig()
        self.rules = GERules(
            sell_tax_rate=self.cfg.fee_sell_tax,
            apply_tax_cap=self.cfg.apply_sell_tax_cap,
            tax_cap_gp=self.cfg.fee_sell_tax_cap_gp,
        )

        self.df = pd.read_csv(self.cfg.ts_path, parse_dates=["parsed_utc"])
        # Expect columns: parsed_utc, item_id, mid, log_return, vol_5 (rolling volatility, not volume), spread_pct (optional)
        if "spread_pct" not in self.df.columns:
            # if you don't have spread_pct in your timeseries, set to 0
            self.df["spread_pct"] = 0.0

        self.df = self.df.dropna(subset=["parsed_utc", "item_id", "mid"])
        self.df = self.df.sort_values(["parsed_utc", "item_id"]).reset_index(drop=True)

        self.times = sorted(self.df["parsed_utc"].unique().tolist())

        self.rng = np.random.default_rng(self.cfg.seed)

        # buy limits (4h window): required CSV. Optional column bucket_id = connected limits (e.g. potion doses share a bucket).
        buy_limit_path = Path(self.cfg.buy_limit_csv)
        if not buy_limit_path.exists():
            raise FileNotFoundError(
                f"buy_limit_csv is required and must exist: {self.cfg.buy_limit_csv}. "
                "Add a CSV with columns item_id, limit (and optional bucket_id for connected limits)."
            )
        bl = pd.read_csv(buy_limit_path)
        if "item_id" not in bl.columns or "limit" not in bl.columns:
            raise ValueError(
                f"buy_limit_csv must have columns 'item_id' and 'limit': {self.cfg.buy_limit_csv}"
            )
        # Connected limits: item_id -> bucket_id; bucket_id -> limit. Same bucket_id = shared 4h limit.
        # If CSV has no bucket_id column, bucket_id = item_id (per-item limit). Else e.g. potion doses share a bucket.
        self._item_to_bucket: Dict[int, Any] = {}
        self._bucket_limit: Dict[Any, int] = {}
        for _, row in bl.iterrows():
            iid = int(row["item_id"])
            limit_val = int(row["limit"])
            bucket = row["bucket_id"] if "bucket_id" in bl.columns and pd.notna(row.get("bucket_id")) else iid
            # normalize bucket for dict key (e.g. "1" vs 1)
            bucket = int(bucket) if isinstance(bucket, (int, float)) and not isinstance(bucket, bool) else bucket
            self._item_to_bucket[iid] = bucket
            self._bucket_limit[bucket] = limit_val
        if not self._item_to_bucket:
            raise ValueError(f"buy_limit_csv has no rows: {self.cfg.buy_limit_csv}")

        # Action (v2.1): [candidate_index, act_type, price_offset_idx, qty_idx]
        # act_type: 0=HOLD, 1=PLACE_BUY, 2=PLACE_SELL, 3=CANCEL_BUY, 4=CANCEL_SELL
        self.action_space = spaces.MultiDiscrete([self.cfg.max_candidates, 5, 7, 3])
        self._price_offset_grid = np.array([-3, -2, -1, 0, 1, 2, 3], dtype=np.int32)
        self._qty_grid = np.array([0.25, 0.5, 1.0], dtype=np.float32)

        # Observation: features per candidate + portfolio + resting order state
        # For each candidate: [mid_norm, logret, vol_5 (volatility), spread_pct]
        # Portfolio: [cash_norm, has_pos, pos_units_norm, pos_mid_norm,
        #             has_active_buy, buy_price_rel_mid, buy_qty_norm, buy_age_norm,
        #             has_active_sell, sell_price_rel_mid, sell_qty_norm, sell_age_norm]
        self.obs_dim_per_item = 4
        self.port_dim = 12
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.cfg.max_candidates * self.obs_dim_per_item + self.port_dim,),
            dtype=np.float32,
        )

        # state
        self.t0 = 0
        self.t = 0
        self.cash = self.cfg.starting_cash
        self.position_item: Optional[int] = None
        self.position_units: float = 0.0
        self.prev_worth: float = self.cfg.starting_cash
        self.last_pos_price: Optional[float] = None  # fallback when position item missing from slice
        # buy limit window: bucket_id -> (bought_qty_in_window, window_start_ts); shared buckets = connected limits
        self._buy_window: Dict[Any, Tuple[int, Any]] = {}
        self.position_cost_basis: float = 0.0  # cash spent to open position (for realized_pnl)
        # v2.0 resting orders: at most one buy + one sell offer
        self.active_buy: Optional[Dict[str, Any]] = None
        self.active_sell: Optional[Dict[str, Any]] = None
        self.step_index: int = 0

    # ---------- helpers ----------

    def _bid_ask(self, mid: float, spread_pct: float) -> Tuple[float, float]:
        """Bid and ask from mid and spread; use floor so we always pay/take spread."""
        s = max(float(spread_pct), self.cfg.min_spread_pct_floor)
        bid = mid * (1.0 - s / 2.0)
        ask = mid * (1.0 + s / 2.0)
        return (bid, ask)

    def _full_slice(self, time_idx: int) -> pd.DataFrame:
        """Full dataframe slice for this timestamp (all items). Use for MTM and liquidation, not obs."""
        ts = self.times[time_idx]
        return self.df[self.df["parsed_utc"] == ts]

    def _get_position_mid_from_slice(
        self, slice_df: pd.DataFrame
    ) -> Tuple[Optional[float], bool, int]:
        """Get mid price for position_item from slice. Returns (mid, used_fallback, match_count).
        Uses strict int match; only one row should match per (timestamp, item_id) in clean data."""
        if self.position_item is None or slice_df.empty:
            return (self.last_pos_price, True, 0)
        # Drop rows with missing item_id so we don't match wrong
        clean = slice_df.dropna(subset=["item_id"])
        if clean.empty:
            return (self.last_pos_price, True, 0)
        pid = int(self.position_item)
        try:
            match = clean.loc[clean["item_id"].astype(int) == pid]
        except (TypeError, ValueError):
            return (self.last_pos_price, True, 0)
        n = len(match)
        if n == 0:
            return (self.last_pos_price, True, 0)
        mid_val = float(match.iloc[0]["mid"])
        if n > 1:
            # Duplicate (ts, item_id) rows: take first, caller can log
            pass
        return (mid_val, False, n)

    def _snapshot(self, time_idx: int) -> pd.DataFrame:
        # time_idx is index into self.times
        ts = self.times[time_idx]
        snap = self.df[self.df["parsed_utc"] == ts].copy()

        # filters
        snap = snap[(snap["mid"] >= self.cfg.min_mid)]
        snap = snap[(snap["spread_pct"] <= self.cfg.max_spread_pct)]

        # if nothing left, fall back to unfiltered snapshot to avoid empty obs
        if snap.empty:
            snap = self.df[self.df["parsed_utc"] == ts].copy()

        # rank candidates: vol_5 is rolling std of log returns (volatility), not volume/liquidity
        # (real liquidity would need volume from GE DB; wiki notes volume exists but isn't shown in-game)
        if "vol_5" in snap.columns and snap["vol_5"].notna().any():
            snap = snap.sort_values(["vol_5", "mid"], ascending=[False, False])
        else:
            snap = snap.sort_values(["mid"], ascending=[False])

        return snap.head(self.cfg.max_candidates).reset_index(drop=True)

    def _get_obs(self) -> np.ndarray:
        snap = self._snapshot(self.t)
        # pad if fewer than max_candidates
        if len(snap) < self.cfg.max_candidates:
            pad_n = self.cfg.max_candidates - len(snap)
            pad = pd.DataFrame(
                {
                    "item_id": [-1] * pad_n,
                    "mid": [0.0] * pad_n,
                    "log_return": [0.0] * pad_n,
                    "vol_5": [0.0] * pad_n,
                    "spread_pct": [0.0] * pad_n,
                }
            )
            snap = pd.concat([snap, pad], ignore_index=True)

        mids = snap["mid"].astype(float).to_numpy()
        logret = snap["log_return"].fillna(0.0).astype(float).to_numpy()
        vol5 = snap["vol_5"].fillna(0.0).astype(float).to_numpy()
        spread = snap["spread_pct"].fillna(0.0).astype(float).to_numpy()

        # normalize mids by median to keep scale tame
        med = np.median(mids[mids > 0]) if np.any(mids > 0) else 1.0
        mid_norm = mids / max(med, 1e-9)

        feats = np.stack([mid_norm, logret, vol5, spread], axis=1).reshape(-1)

        # portfolio features
        cash_norm = self.cash / self.cfg.starting_cash
        has_pos = 1.0 if self.position_item is not None else 0.0
        pos_units_norm = self.position_units / 1_000_000.0  # arbitrary scale
        pos_mid_norm = 0.0
        if self.position_item is not None:
            # Use full slice so position item is always findable (not just top-N candidates)
            full_at_t = self._full_slice(self.t)
            pos_mid_val, _, _ = self._get_position_mid_from_slice(full_at_t)
            if pos_mid_val is not None and pos_mid_val > 0:
                pos_mid_norm = pos_mid_val / max(med, 1e-9)

        # resting order features (single-slot: 1 active buy + 1 active sell)
        has_active_buy = 1.0 if self.active_buy is not None else 0.0
        buy_price_rel_mid = 0.0
        buy_qty_norm = 0.0
        buy_age_norm = 0.0
        if self.active_buy is not None:
            buy_item_id = int(self.active_buy.get("item_id", -1))
            buy_mid = None
            if buy_item_id != -1:
                match = snap[snap["item_id"].astype(int) == buy_item_id]
                if not match.empty:
                    buy_mid = float(match.iloc[0]["mid"])
            if buy_mid is None:
                buy_mid = med if med > 0 else 1.0
            buy_price_rel_mid = (float(self.active_buy["price"]) - buy_mid) / max(buy_mid, 1e-9)
            buy_qty_norm = float(self.active_buy.get("qty_remaining", 0)) / 1_000_000.0
            age_steps = max(0, self.t - int(self.active_buy.get("created_step", self.t)))
            buy_age_norm = min(1.0, age_steps / max(self.cfg.episode_len, 1))

        has_active_sell = 1.0 if self.active_sell is not None else 0.0
        sell_price_rel_mid = 0.0
        sell_qty_norm = 0.0
        sell_age_norm = 0.0
        if self.active_sell is not None:
            sell_item_id = int(self.active_sell.get("item_id", -1))
            sell_mid = None
            if sell_item_id != -1:
                match = snap[snap["item_id"].astype(int) == sell_item_id]
                if not match.empty:
                    sell_mid = float(match.iloc[0]["mid"])
            if sell_mid is None:
                sell_mid = med if med > 0 else 1.0
            sell_price_rel_mid = (sell_mid - float(self.active_sell["price"])) / max(sell_mid, 1e-9)
            sell_qty_norm = float(self.active_sell.get("qty_remaining", 0)) / 1_000_000.0
            age_steps = max(0, self.t - int(self.active_sell.get("created_step", self.t)))
            sell_age_norm = min(1.0, age_steps / max(self.cfg.episode_len, 1))

        port = np.array(
            [
                cash_norm,
                has_pos,
                pos_units_norm,
                pos_mid_norm,
                has_active_buy,
                buy_price_rel_mid,
                buy_qty_norm,
                buy_age_norm,
                has_active_sell,
                sell_price_rel_mid,
                sell_qty_norm,
                sell_age_norm,
            ],
            dtype=np.float32,
        )
        obs = np.concatenate([feats.astype(np.float32), port], axis=0)
        return obs

    def _net_worth(self, slice_df: pd.DataFrame) -> float:
        worth = self.cash
        if self.position_item is not None and self.position_units > 0:
            price, used_fallback, match_count = self._get_position_mid_from_slice(slice_df)
            if price is not None:
                if match_count == 1:
                    self.last_pos_price = price
                worth += price * self.position_units
        return float(worth)

    def _force_liquidate(self, slice_df: pd.DataFrame) -> None:
        """Sell any open position at bid, applying sell tax. Uses full slice so item is findable."""
        if self.position_item is None or self.position_units <= 0:
            return
        mid, used_fallback, _ = self._get_position_mid_from_slice(slice_df)
        if mid is None and self.last_pos_price is not None:
            mid = self.last_pos_price
        bid: Optional[float] = None
        if mid is not None:
            # Use spread from slice if we have a match for this item
            clean = slice_df.dropna(subset=["item_id"])
            match = clean.loc[clean["item_id"].astype(int) == int(self.position_item)] if not clean.empty else pd.DataFrame()
            sp = float(match.iloc[0].get("spread_pct", self.cfg.min_spread_pct_floor)) if len(match) > 0 else self.cfg.min_spread_pct_floor
            bid, _ = self._bid_ask(mid, sp)
        if bid is None:
            return
        proceeds = self.position_units * bid
        proceeds_after_tax = self.rules.apply_sell_tax(proceeds)
        self.cash += proceeds_after_tax
        self.position_item = None
        self.position_units = 0.0
        self.last_pos_price = None
        self.position_cost_basis = 0.0

    # ---------- gym API ----------

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)

        if len(self.times) < self.cfg.episode_len + 1:
            raise ValueError(
                f"Not enough timestamps ({len(self.times)}) for episode_len={self.cfg.episode_len}"
            )

        # choose random start so we have enough room for episode
        self.t0 = int(self.rng.integers(0, len(self.times) - self.cfg.episode_len))
        self.t = self.t0
        self.step_index = 0

        self.cash = self.cfg.starting_cash
        self.position_item = None
        self.position_units = 0.0
        self.position_cost_basis = 0.0
        self.last_pos_price = None
        self._buy_window.clear()
        self.active_buy = None
        self.active_sell = None

        snap = self._snapshot(self.t)
        self.prev_worth = self._net_worth(snap)

        return self._get_obs(), {}

    def _simulate_fills(self, snap: pd.DataFrame) -> Tuple[int, int]:
        """Probabilistic partial fills for at most one buy + one sell offer."""
        buy_filled = 0
        sell_filled = 0

        if self.active_buy is not None:
            qty = int(self.active_buy.get("qty_remaining", 0))
            if qty > 0:
                item_id = int(self.active_buy.get("item_id", -1))
                match = snap[snap["item_id"].astype(int) == item_id]
                if not match.empty:
                    mid = float(match.iloc[0]["mid"])
                    if mid > 0:
                        price = float(self.active_buy["price"])
                        edge = (price - mid) / mid
                        k = float(self.cfg.fill_slope)
                        p_fill = 1.0 / (1.0 + np.exp(-k * edge))
                        p_fill = float(np.clip(p_fill, 0.0, 1.0))
                        if p_fill > 0.0:
                            fill_qty = int(self.rng.binomial(qty, p_fill))
                            if fill_qty > 0:
                                # respect available cash at fill time
                                max_affordable = int(self.cash // price)
                                fill_qty = min(fill_qty, max_affordable)
                                if fill_qty > 0:
                                    cost = float(fill_qty) * price
                                    self.cash -= cost
                                    # update position and cost basis
                                    if self.position_item is None:
                                        self.position_item = item_id
                                        self.position_units = 0.0
                                        self.position_cost_basis = 0.0
                                    if self.position_item == item_id:
                                        self.position_units += float(fill_qty)
                                        self.position_cost_basis += cost
                                        self.last_pos_price = mid
                                    qty -= fill_qty
                                    self.active_buy["qty_remaining"] = qty
                                    buy_filled = fill_qty

        if self.active_sell is not None:
            qty = int(self.active_sell.get("qty_remaining", 0))
            if qty > 0 and self.position_item is not None and self.position_units > 0:
                item_id = int(self.active_sell.get("item_id", -1))
                match = snap[snap["item_id"].astype(int) == item_id]
                if not match.empty:
                    mid = float(match.iloc[0]["mid"])
                    if mid > 0:
                        price = float(self.active_sell["price"])
                        edge = (mid - price) / mid
                        k = float(self.cfg.fill_slope)
                        p_fill = 1.0 / (1.0 + np.exp(-k * edge))
                        p_fill = float(np.clip(p_fill, 0.0, 1.0))
                        if p_fill > 0.0:
                            fill_qty = int(self.rng.binomial(qty, p_fill))
                            if fill_qty > 0:
                                # cannot sell more than current position
                                max_sellable = int(self.position_units)
                                fill_qty = min(fill_qty, max_sellable)
                                if fill_qty > 0:
                                    gross = float(fill_qty) * price
                                    proceeds_after_tax = self.rules.apply_sell_tax(gross)
                                    self.cash += proceeds_after_tax
                                    # update position units and cost basis (simple average-cost model)
                                    if self.position_units > 0:
                                        avg_cost = self.position_cost_basis / max(self.position_units, 1e-9)
                                        cost_out = avg_cost * float(fill_qty)
                                        self.position_cost_basis = max(0.0, self.position_cost_basis - cost_out)
                                        self.position_units = max(0.0, self.position_units - float(fill_qty))
                                    qty -= fill_qty
                                    self.active_sell["qty_remaining"] = qty
                                    sell_filled = fill_qty

        if self.active_buy is not None and int(self.active_buy.get("qty_remaining", 0)) <= 0:
            self.active_buy = None
        if self.active_sell is not None and int(self.active_sell.get("qty_remaining", 0)) <= 0:
            self.active_sell = None

        return buy_filled, sell_filled

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        cand_idx = int(action[0])
        act_type = int(action[1])  # 0=HOLD, 1=PLACE_BUY, 2=PLACE_SELL
        price_offset_idx = int(action[2])
        qty_idx = int(action[3])

        # Context-invalid actions → auto-HOLD, no blocked_reason (so they don't pollute "blocked" stats or penalty)
        blocked_reason: Optional[str] = None
        if act_type == 2 and (self.position_item is None or self.position_units <= 0):
            act_type = 0  # SELL when no position
        elif act_type == 1 and self.position_item is not None:
            act_type = 0  # BUY when already in position
        if act_type == 1 and self.active_buy is not None:
            act_type = 0  # PLACE_BUY when buy order already resting
        if act_type == 2 and self.active_sell is not None:
            act_type = 0  # PLACE_SELL when sell order already resting

        # v2.1: CANCEL — no order to cancel → treat as HOLD (no blocked_reason)
        if act_type == 3 and self.active_buy is None:
            act_type = 0
        if act_type == 4 and self.active_sell is None:
            act_type = 0

        snap = self._snapshot(self.t)

        cand_idx = max(0, min(cand_idx, len(snap) - 1))
        row = snap.iloc[cand_idx]
        item_id = int(row["item_id"]) if row["item_id"] != -1 else -1
        mid = float(row["mid"])
        spread_pct = float(row.get("spread_pct", self.cfg.min_spread_pct_floor))

        executed = "NONE"
        exec_price: Optional[float] = None
        realized_pnl: Optional[float] = None
        acted_item_id_for_info: int = item_id

        # map discrete indices to offsets / qty fractions
        price_offset = int(self._price_offset_grid[price_offset_idx % len(self._price_offset_grid)])
        qty_frac = float(self._qty_grid[qty_idx % len(self._qty_grid)])

        if act_type == 3:  # CANCEL_BUY: release order and return unfilled qty to 4h limit
            if self.active_buy is not None:
                item_id_b = int(self.active_buy.get("item_id", -1))
                qty_rem = int(self.active_buy.get("qty_remaining", 0))
                if item_id_b in self._item_to_bucket and qty_rem > 0:
                    bucket_id = self._item_to_bucket[item_id_b]
                    bought, start_ts = self._buy_window.get(bucket_id, (0, self.times[self.t]))
                    new_bought = max(0, int(bought) - qty_rem)
                    self._buy_window[bucket_id] = (new_bought, start_ts)
                self.active_buy = None
                executed = "CANCEL_BUY"

        elif act_type == 4:  # CANCEL_SELL: clear resting sell (position was never deducted on place)
            if self.active_sell is not None:
                self.active_sell = None
                executed = "CANCEL_SELL"

        elif act_type == 1:  # PLACE_BUY: post resting buy offer at limit price
            if item_id == -1 or mid <= 0 or self.cash <= 0 or qty_frac <= 0.0:
                if item_id == -1 or mid <= 0:
                    blocked_reason = blocked_reason or "buy_blocked_invalid_item_or_price"
                else:
                    blocked_reason = blocked_reason or "buy_blocked_no_cash"
            else:
                limit_price = mid * (1.0 + price_offset * self.cfg.price_offset_pct_step)
                if limit_price <= 0:
                    blocked_reason = blocked_reason or "buy_blocked_invalid_limit_price"
                else:
                    # buy limits (4h window): item must be in CSV; use bucket_id for connected limits
                    if item_id not in self._item_to_bucket:
                        blocked_reason = blocked_reason or "buy_blocked_no_limit_for_item"
                    else:
                        bucket_id = self._item_to_bucket[item_id]
                        limit_int = self._bucket_limit[bucket_id]
                        now_ts = self.times[self.t]
                        bought, start_ts = self._buy_window.get(bucket_id, (0, now_ts))
                        delta_sec = (pd.Timestamp(now_ts) - pd.Timestamp(start_ts)).total_seconds()
                        if delta_sec >= self.cfg.buy_limit_window_seconds:
                            bought, start_ts = 0, now_ts
                        bought_int = int(bought)
                        remaining = max(0, limit_int - bought_int)
                        if remaining <= 0:
                            blocked_reason = blocked_reason or "buy_blocked_limit_reached"
                        else:
                            # position sizing + integer qty (GE trades whole units)
                            spend_cap = self.cash * self.cfg.max_cash_fraction_per_trade
                            trade_budget = max(0.0, spend_cap) * qty_frac
                            units_wanted = int(trade_budget / limit_price)
                            units_order = min(units_wanted, remaining)
                            if units_order <= 0:
                                blocked_reason = blocked_reason or "buy_blocked_limit_reached"
                            else:
                                self.active_buy = {
                                    "item_id": item_id,
                                    "price": float(limit_price),
                                    "qty_remaining": int(units_order),
                                    "created_step": int(self.t),
                                }
                                # v2.0 simplification: consume limit at order placement (not per fill)
                                self._buy_window[bucket_id] = (bought_int + units_order, start_ts)
                                executed = "PLACE_BUY"
                                exec_price = float(limit_price)

        elif act_type == 2:  # PLACE_SELL: post resting sell offer at limit price (only if in position)
            if self.position_item is None or self.position_units <= 0:
                blocked_reason = blocked_reason or "sell_blocked_no_position"
            else:
                # use full slice so held item is always findable
                full_now = self._full_slice(self.t)
                match = full_now[full_now["item_id"].astype(int) == int(self.position_item)]
                if match.empty and self.last_pos_price is None:
                    blocked_reason = blocked_reason or "sell_blocked_no_price"
                else:
                    if not match.empty:
                        base_mid = float(match.iloc[0]["mid"])
                    else:
                        base_mid = float(self.last_pos_price or 0.0)
                    if base_mid <= 0:
                        blocked_reason = blocked_reason or "sell_blocked_invalid_price"
                    else:
                        limit_price = base_mid * (1.0 + price_offset * self.cfg.price_offset_pct_step)
                        if limit_price <= 0:
                            blocked_reason = blocked_reason or "sell_blocked_invalid_limit_price"
                        else:
                            max_units = int(self.position_units)
                            units_order = int(max_units * qty_frac)
                            units_order = max(1, min(units_order, max_units))
                            if units_order <= 0:
                                blocked_reason = blocked_reason or "sell_blocked_no_position"
                            else:
                                acted_item_id_for_info = int(self.position_item)
                                self.active_sell = {
                                    "item_id": int(self.position_item),
                                    "price": float(limit_price),
                                    "qty_remaining": int(units_order),
                                    "created_step": int(self.t),
                                }
                                executed = "PLACE_SELL"
                                exec_price = float(limit_price)

        # simulate probabilistic partial fills for resting orders (including ones just posted)
        buy_filled, sell_filled = self._simulate_fills(snap)
        if buy_filled > 0 and sell_filled == 0:
            executed = "BUY"
        elif sell_filled > 0 and buy_filled == 0:
            executed = "SELL"
        elif buy_filled > 0 and sell_filled > 0:
            executed = "BUY_SELL"

        # Advance time: each step uses next timestamp so prices can change (time-series, not static snapshot).
        self.t += 1
        self.step_index += 1
        done = (self.t >= self.t0 + self.cfg.episode_len)

        # Use same timestamp as logged step_ts so valuation and trace refer to the same moment (no off-by-one).
        worth_time_idx = self.t - 1
        worth_slice = self._full_slice(worth_time_idx)
        if done:
            self._force_liquidate(worth_slice)

        # Mark-to-market using full slice so position is always findable (no net_worth=0 spike).
        new_worth = self._net_worth(worth_slice)

        # log-return reward (stable, clipped)
        reward = equity_log_reward(self.prev_worth, new_worth, clip=1.0)
        self.prev_worth = new_worth

        # --- reward shaping (training-friendly) ---
        # Overridden invalid actions have act_type=0 so they get HOLD_PENALTY, not INVALID_ACTION_PENALTY
        HOLD_PENALTY = 0.0              # don't punish holding by default
        TRADE_PENALTY = 1e-5             # tiny cost for churn
        INVALID_ACTION_PENALTY = 1e-3   # only for non-overridden blocks (e.g. limit_reached, no_cash)
        if executed in ("BUY", "SELL"):
            reward -= TRADE_PENALTY
        elif act_type != 0 and executed == "NONE":
            reward -= INVALID_ACTION_PENALTY
        else:
            reward -= HOLD_PENALTY

        # Debug / eval: timestamp and acted-on item/price, position mark (for Episode 0 trace)
        step_ts = self.times[self.t - 1]
        pos_mid_val, pos_mid_used_fallback, pos_mid_match_count = self._get_position_mid_from_slice(worth_slice)
        pos_mid = pos_mid_val

        # acted_mid must match acted_item_id (BUY = candidate item, SELL = held item); use worth_slice for both
        acted_mid: Optional[float] = None
        if acted_item_id_for_info >= 0 and not worth_slice.empty:
            clean = worth_slice.dropna(subset=["item_id"])
            match = clean.loc[clean["item_id"].astype(int) == acted_item_id_for_info]
            if len(match) > 0:
                acted_mid = float(match.iloc[0]["mid"])

        worth_ts = self.times[worth_time_idx]

        # Suspicious valuation: only when we have a position and match/fallback suggest a real bug.
        valuation_debug = None
        in_position = self.position_item is not None and self.position_units > 0
        suspicious = in_position and (pos_mid_match_count != 1 or pos_mid_used_fallback)
        if suspicious and not worth_slice.empty:
            clean_ws = worth_slice.dropna(subset=["item_id"])
            rows_pos = clean_ws.loc[clean_ws["item_id"].astype(int) == int(self.position_item)] if self.position_item is not None else pd.DataFrame()
            rows_acted = clean_ws.loc[clean_ws["item_id"].astype(int) == acted_item_id_for_info] if acted_item_id_for_info >= 0 else pd.DataFrame()
            valuation_debug = {
                "step_ts": str(step_ts),
                "worth_ts": str(worth_ts),
                "position_item": self.position_item,
                "acted_item_id": acted_item_id_for_info,
                "acted_mid": acted_mid,
                "pos_mid": pos_mid,
                "pos_mid_match_count": pos_mid_match_count,
                "pos_mid_used_fallback": pos_mid_used_fallback,
                "matched_rows_position": rows_pos.head(3).to_dict("records") if len(rows_pos) > 0 else [],
                "matched_rows_acted": rows_acted.head(3).to_dict("records") if len(rows_acted) > 0 else [],
            }

        info = {"net_worth": new_worth, "cash": self.cash, "pos_item": self.position_item, "pos_units": self.position_units}
        info["action"] = act_type
        info["worth_ts"] = worth_ts
        info["valuation_debug"] = valuation_debug
        info["in_position"] = bool(self.position_item is not None)
        info["pos_item"] = int(self.position_item) if self.position_item is not None else -1
        info["pos_units"] = float(self.position_units)
        info["cash"] = float(self.cash)
        info["net_worth"] = float(new_worth)
        info["parsed_utc"] = step_ts
        info["acted_item_id"] = acted_item_id_for_info
        info["acted_mid"] = acted_mid
        info["pos_item_id"] = int(self.position_item) if self.position_item is not None else -1
        info["pos_mid"] = pos_mid  # mark price of position after step (None if no position)
        info["pos_mid_used_fallback"] = pos_mid_used_fallback
        info["pos_mid_match_count"] = pos_mid_match_count
        info["executed"] = executed
        info["buy_filled"] = int(buy_filled)
        info["sell_filled"] = int(sell_filled)
        info["has_active_buy"] = self.active_buy is not None
        info["has_active_sell"] = self.active_sell is not None
        if exec_price is not None:
            info["exec_price"] = exec_price
        if realized_pnl is not None:
            info["realized_pnl"] = realized_pnl
        if blocked_reason is not None:
            info["blocked_reason"] = blocked_reason
        obs = self._get_obs() if not done else self._get_obs()  # keep simple
        return obs, reward, done, False, info