from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

from src.geRules import GERules


@dataclass
class EnvConfig:
    k_candidates: int = 25 # number of items the agent can choose every step
    max_spread_pct: float = 0.05 # liquidity filter
    episode_len: int = 200 # steps per episode
    starting_cash: float = 1.0 # normalized bankroll
    use_log_mid: bool = True # stabilizing scale of prices
    
class GERLTraderEnv(gym.Env):
    # Offline RL trading env which uses the GE snapshots
    
    # Action = (candidate_index, action_type)
    # action_type: 0 = HOLD, 1 = BUY, 2 = SELL
    
    # Constraints:
    # - single position at a time / buy has no GE tax / sell implies GE tax through GERules / execution at snapshot mid (proxy)
    
    metadata = {"render_modes": []}
    
    def __init__(self, ts_csv: str | Path, cfg: EnvConfig = EnvConfig(), rules: GERules | None=None):
        super().__init__()
        self.cfg = cfg
        self.rules = rules or GERules()
        
        df = pd.read_csv(ts_csv, parse_dates=["parsed_utc"])
        df = df.sort_values(["parsed_utc", "item_id"]).reset_index(drop=True)
        
        needed = ["parsed_utc", "item_id", "mid", "log_return", "spread_pct"]
        df = df.dropna(subset=needed)
        
        self.df = df
        self.times = sorted(df["parsed_utc"].unique())
        
        # portfolio state
        self.cash = self.cfg.starting_cash
        self.pos_item: int | None = None
        self.pos_units: float = 0.0
        self.held_last_price: float | None = None
        
        # observation: K candidates * 3 features + 2 portfolio features
        # item features: [log(mid) or mid, log_return, spread_pct]
        obs_dim = self.cfg.k_candidates * 3 + 2
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.MultiDiscrete([self.cfg.k_candidates, 3])
        
        self.t0 = 0
        self.t = 0
        
    def _snapshot(self, time_idx: int) -> pd.DataFrame:
        tstamp = self.times[time_idx]
        snap = self.df[self.df["parsed_utc"] == tstamp].copy()
        
        # liquidity filter
        snap = snap[snap["spread_pct"] <= self.cfg.max_spread_pct].copy()
        
        # rank candidates: prefer movement with slow spread
        snap["score"] = (snap["log_return"].abs().fillna(0.0)) / (snap["spread_pct"] + 1e-6)
        snap = snap.sort_values("score", ascending=False)
        return snap.head(self.cfg.k_candidates)
    
    # added a full snapshopt at time t
    def _full_snapshot(self, time_idx: int) -> pd.DataFrame:
        tstamp = self.times[time_idx]
        return self.df[self.df["parsed_utc"] == tstamp]
    
    def _mark_to_market(self, time_idx: int) -> float:
        if self.pos_item is None:
            return self.cash

        full = self._full_snapshot(time_idx)
        row = full[full["item_id"] == self.pos_item]
        if len(row) > 0:
            self.held_last_price = float(row.iloc[0]["mid"])

        if self.held_last_price is None:
            return self.cash

        return self.cash + self.pos_units * self.held_last_price    
    
    def _get_obs(self) -> np.ndarray:
        snap = self._snapshot(self.t)
        
        feats: list[float] = []
        for _, r in snap.iterrows():
            mid = float(r["mid"])
            mid_feat = np.log(max(mid, 1e-9)) if self.cfg.use_log_mid else mid
            feats.extend([mid_feat, float(r["log_return"]), float(r["spread_pct"])])
            
        while len(feats) < self.cfg.k_candidates * 3:
            feats.extend([0.0, 0.0, 0.0])
            
        holding_flag = 0.0 if self.pos_item is None else 1.0
        feats.extend([float(self.cash), holding_flag])
        
        return np.asarray(feats, dtype=np.float32)
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        options = options or {}
        start_idx = options.get("start_idx", None)

        max_start = len(self.times) - (self.cfg.episode_len + 1)
        if max_start < 0:
            raise ValueError(
                f"Not enough timestamps ({len(self.times)}) for episode_len={self.cfg.episode_len}"
            )

        if start_idx is None:
            self.t0 = int(self.np_random.integers(0, max_start + 1))
        else:
            start_idx = int(start_idx)
            if not (0 <= start_idx <= max_start):
                raise ValueError(f"start_idx must be in [0, {max_start}], got {start_idx}")
            self.t0 = start_idx

        self.t = self.t0

        self.cash = self.cfg.starting_cash
        self.pos_item = None
        self.pos_units = 0.0
        self.held_last_price = None

        return self._get_obs(), {}    

    def step(self, action):
        cand_idx,act = int(action[0]), int(action[1])
        
        snap = self._snapshot(self.t)
        prev_worth = self._mark_to_market(self.t)        
        
        chosen_item = None
        chosen_price = None
        if cand_idx < len(snap):
            chosen_item = int(snap.iloc[cand_idx]["item_id"])
            chosen_price = float(snap.iloc[cand_idx]["mid"])
            
        # executing
        if chosen_item is not None and chosen_price is not None:
            # approximate bid/ask from mid + spread
            # spread_pct ~= (high-low)/mid, so half-spread ~= spread_pct/2
            half = float(snap.iloc[cand_idx]["spread_pct"]) * 0.5
            ask = chosen_price * (1.0 + half)   # buy worse
            bid = chosen_price * (1.0 - half)   # sell worse

            # BUY: no GE tax (but you pay the spread implicitly via ask)
            if act == 1 and self.pos_item is None:
                if ask > 0:
                    self.pos_units = self.cash / ask
                    self.cash = 0.0
                    self.pos_item = chosen_item
                    self.held_last_price = ask

            # SELL: apply spread (bid) + 2% GE tax
            # elif act == 2 and self.pos_item is not None:
            #     # only allow selling the held item; use its bid from the *current* snapshot if present
            #     # if the held item isn't in the candidate list, fall back to full snapshot mid and apply its spread if available
            #     if chosen_item == self.pos_item:
            #         gross = self.pos_units * bid
            #         self.cash = self.rules.sell_net_proceeds(gross)
            #         self.pos_units = 0.0
            #         self.pos_item = None
            #         self.held_last_price = None
            #     else:
            #         # fallback: look up held item at this time
            #         full = self._full_snapshot(self.t)
            #         row = full[full["item_id"] == self.pos_item]
            #         if len(row) > 0:
            #             mid = float(row.iloc[0]["mid"])
            #             sp = float(row.iloc[0]["spread_pct"])
            #             bid2 = mid * (1.0 - 0.5 * sp)
            #             gross = self.pos_units * bid2
            #             self.cash = self.rules.sell_net_proceeds(gross)
            #             self.pos_units = 0.0
            #             self.pos_item = None
            #             self.held_last_price = None    
            
            ## Updated to below so that I remove the "candidate-set trap" where PPO can buyu something, but later can't sell because it wasn't in the top-K candidates at the sell timestep.
            # SELL: always sell the currently held item (ignore cand_idx)
            elif act == 2 and self.pos_item is not None and self.pos_units > 0:
                # Look up held item on the FULL snapshot for this timestep
                full = self._full_snapshot(self.t)
                row = full[full["item_id"] == self.pos_item]

                if len(row) > 0:
                    mid = float(row.iloc[0]["mid"])
                    sp = float(row.iloc[0]["spread_pct"])
                    half = 0.5 * sp
                    bid = mid * (1.0 - half)  # worse execution on sell

                    gross = self.pos_units * bid
                    self.cash = float(self.rules.sell_net_proceeds(gross))  # applies 2% tax (floored)
                else:
                    # If the held item can't be found (should be rare), do nothing
                    # (Alternatively: you can force liquidate at last price)
                    pass

                # Clear position
                self.pos_units = 0.0
                self.pos_item = None
                self.held_last_price = None 
                                    
        # advance time
        self.t += 1

        # clamp time first (prevents index errors in _snapshot/_get_obs)
        if self.t >= len(self.times):
            self.t = len(self.times) - 1

        # terminate if we reached episode horizon OR dataset end
        terminated = (self.t >= self.t0 + self.cfg.episode_len) or (self.t >= len(self.times) - 1)

        new_worth = self._mark_to_market(self.t)
        reward = float(new_worth - prev_worth) / float(self.cfg.starting_cash)
        obs = self._get_obs()
        info = {"net_worth": new_worth, "t": self.t}
        
        return obs, reward, terminated, False, info