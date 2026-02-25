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

    # action space: [candidate_index, action_type]
    # action_type: HOLD=0, BUY=1 (use candidate), SELL_CURRENT=2 (ignore candidate, sell held item if any)
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

        # Action: candidate index + action type
        self.action_space = spaces.MultiDiscrete([self.cfg.max_candidates, 3])

        # Observation: features per candidate + portfolio state
        # For each candidate: [mid_norm, logret, vol_5 (volatility), spread_pct]
        # Portfolio: [cash_norm, has_pos, pos_units_norm, pos_mid_norm]
        self.obs_dim_per_item = 4
        self.port_dim = 4
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
            # mark-to-market using current snapshot if present
            match = snap[snap["item_id"].astype(int) == int(self.position_item)]
            if not match.empty:
                pos_mid_norm = float(match.iloc[0]["mid"]) / max(med, 1e-9)

        port = np.array([cash_norm, has_pos, pos_units_norm, pos_mid_norm], dtype=np.float32)
        obs = np.concatenate([feats.astype(np.float32), port], axis=0)
        return obs

    def _net_worth(self, slice_df: pd.DataFrame) -> float:
        worth = self.cash
        if self.position_item is not None and self.position_units > 0:
            match = slice_df[slice_df["item_id"].astype(int) == int(self.position_item)]
            if not match.empty:
                price = float(match.iloc[0]["mid"])
                self.last_pos_price = price
                worth += price * self.position_units
            elif self.last_pos_price is not None:
                worth += self.last_pos_price * self.position_units
        return float(worth)

    def _force_liquidate(self, slice_df: pd.DataFrame) -> None:
        """Sell any open position at bid, applying sell tax. Uses full slice so item is findable."""
        if self.position_item is None or self.position_units <= 0:
            return
        match = slice_df[slice_df["item_id"].astype(int) == int(self.position_item)]
        bid: Optional[float] = None
        if not match.empty:
            mid = float(match.iloc[0]["mid"])
            sp = float(match.iloc[0].get("spread_pct", self.cfg.min_spread_pct_floor))
            bid, _ = self._bid_ask(mid, sp)
        elif self.last_pos_price is not None:
            bid, _ = self._bid_ask(self.last_pos_price, self.cfg.min_spread_pct_floor)
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

        self.cash = self.cfg.starting_cash
        self.position_item = None
        self.position_units = 0.0
        self.position_cost_basis = 0.0
        self.last_pos_price = None
        self._buy_window.clear()

        snap = self._snapshot(self.t)
        self.prev_worth = self._net_worth(snap)

        return self._get_obs(), {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        cand_idx = int(action[0])
        act_type = int(action[1])  # 0 hold, 1 buy, 2 sell

        # Override invalid actions to HOLD so the agent focuses on trading, not "don't be dumb"
        blocked_reason: Optional[str] = None
        if act_type == 2 and (self.position_item is None or self.position_units <= 0):
            blocked_reason = "sell_blocked_no_position"
            act_type = 0  # SELL with no position -> HOLD
        elif act_type == 1 and self.position_item is not None:
            blocked_reason = "buy_blocked_already_in_position"
            act_type = 0  # BUY when already in position -> HOLD

        snap = self._snapshot(self.t)

        # Candidate row is used only for BUY. For SELL_CURRENT we ignore it and sell the held item.
        cand_idx = max(0, min(cand_idx, len(snap) - 1))
        row = snap.iloc[cand_idx]
        item_id = int(row["item_id"]) if row["item_id"] != -1 else -1
        mid = float(row["mid"])
        spread_pct = float(row.get("spread_pct", self.cfg.min_spread_pct_floor))
        _, ask = self._bid_ask(mid, spread_pct)

        executed = "NONE"
        exec_price: Optional[float] = None
        realized_pnl: Optional[float] = None
        acted_item_id_for_info: int = item_id  # for BUY = candidate; for SELL = held item (set below)

        if act_type == 1:  # BUY selected candidate at ask
            if self.position_item is not None:
                blocked_reason = "buy_blocked_already_in_position"
            elif item_id == -1 or ask <= 0 or self.cash <= 0:
                if item_id == -1 or ask <= 0:
                    blocked_reason = "buy_blocked_invalid_item_or_price"
                else:
                    blocked_reason = "buy_blocked_no_cash"
            else:
                # buy limits (4h window): item must be in CSV; use bucket_id for connected limits
                if item_id not in self._item_to_bucket:
                    blocked_reason = "buy_blocked_no_limit_for_item"
                else:
                    bucket_id = self._item_to_bucket[item_id]
                    limit_int = self._bucket_limit[bucket_id]
                    # Window reset is by real elapsed time (parsed_utc), not step count
                    now_ts = self.times[self.t]
                    bought, start_ts = self._buy_window.get(bucket_id, (0, now_ts))
                    delta_sec = (pd.Timestamp(now_ts) - pd.Timestamp(start_ts)).total_seconds()
                    if delta_sec >= self.cfg.buy_limit_window_seconds:
                        bought, start_ts = 0, now_ts
                    bought_int = int(bought)
                    remaining = max(0, limit_int - bought_int)
                    if remaining <= 0:
                        blocked_reason = "buy_blocked_limit_reached"
                    else:
                        # position sizing + integer qty (GE trades whole units)
                        spend_cap = self.cash * self.cfg.max_cash_fraction_per_trade
                        units_wanted = int(spend_cap / ask)
                        units_bought = min(units_wanted, remaining)
                        if units_bought <= 0:
                            blocked_reason = "buy_blocked_limit_reached"
                        else:
                            cost = units_bought * ask
                            self.cash -= cost
                            self.position_units = float(units_bought)
                            self.position_item = item_id
                            self.last_pos_price = mid
                            self.position_cost_basis = cost
                            self._buy_window[bucket_id] = (bought_int + units_bought, start_ts)
                            executed = "BUY"
                            exec_price = ask

        elif act_type == 2:  # SELL_CURRENT: sell held position at bid; candidate_index is unused
            if self.position_item is None or self.position_units <= 0:
                blocked_reason = "sell_blocked_no_position"
            else:
                full_now = self._full_slice(self.t)
                match = full_now[full_now["item_id"].astype(int) == int(self.position_item)]
                if match.empty and self.last_pos_price is None:
                    blocked_reason = "sell_blocked_no_price"
                else:
                    if not match.empty:
                        m = float(match.iloc[0]["mid"])
                        sp = float(match.iloc[0].get("spread_pct", self.cfg.min_spread_pct_floor))
                        bid, _ = self._bid_ask(m, sp)
                    else:
                        bid, _ = self._bid_ask(self.last_pos_price or 0.0, self.cfg.min_spread_pct_floor)
                    if bid <= 0:
                        blocked_reason = "sell_blocked_invalid_price"
                    else:
                        acted_item_id_for_info = int(self.position_item)
                        proceeds = self.position_units * bid
                        proceeds_after_tax = self.rules.apply_sell_tax(proceeds)
                        self.cash += proceeds_after_tax
                        realized_pnl = proceeds_after_tax - self.position_cost_basis
                        self.position_item = None
                        self.position_units = 0.0
                        self.last_pos_price = None
                        self.position_cost_basis = 0.0
                        executed = "SELL"
                        exec_price = bid

        # Advance time: each step uses next timestamp so prices can change (time-series, not static snapshot).
        self.t += 1
        done = (self.t >= self.t0 + self.cfg.episode_len)

        worth_time_idx = self.t - 1 if done else self.t
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
        pos_mid = None
        if self.position_item is not None and not worth_slice.empty:
            match = worth_slice[worth_slice["item_id"].astype(int) == int(self.position_item)]
            if not match.empty:
                pos_mid = float(match.iloc[0]["mid"])
            elif self.last_pos_price is not None:
                pos_mid = self.last_pos_price

        info = {"net_worth": new_worth, "cash": self.cash, "pos_item": self.position_item, "pos_units": self.position_units}
        info["action"] = act_type
        info["in_position"] = bool(self.position_item is not None)
        info["pos_item"] = int(self.position_item) if self.position_item is not None else -1
        info["pos_units"] = float(self.position_units)
        info["cash"] = float(self.cash)
        info["net_worth"] = float(new_worth)
        info["parsed_utc"] = step_ts
        info["acted_item_id"] = acted_item_id_for_info
        info["acted_mid"] = mid
        info["pos_item_id"] = int(self.position_item) if self.position_item is not None else -1
        info["pos_mid"] = pos_mid  # mark price of position after step (None if no position)
        info["executed"] = executed
        if exec_price is not None:
            info["exec_price"] = exec_price
        if realized_pnl is not None:
            info["realized_pnl"] = realized_pnl
        if blocked_reason is not None:
            info["blocked_reason"] = blocked_reason
        obs = self._get_obs() if not done else self._get_obs()  # keep simple
        return obs, reward, done, False, info