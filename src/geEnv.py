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
    
    def _mark_to_market(self, snap: pd.DataFrame) -> float:
        if self.pos_item is None:
            return self.cash
    
        row = snap[snap["item_id"] == self.pos_item]
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
        
        max_start = max(0, len(self.times) - self.cfg.episode_len - 2)
        self.t0 = int(self.np_random.integers(0, max_start + 1))
        self.t = self.t0
        
        self.cash = self.cfg.starting_cash
        self.pos_item = None
        self.pos_units = 0.0
        self.held_last_price = None
        
        return self._get_obs(), {}
    
    def step(self, action):
        cand_idx,act = int(action[0]), int(action[1])
        
        snap = self._snapshot(self.t)
        prev_worth = self._mark_to_market(snap)
        
        chosen_item = None
        chosen_price = None
        if cand_idx < len(snap):
            chosen_item = int(snap.iloc[cand_idx]["item_id"])
            chosen_price = float(snap.iloc[cand_idx]["mid"])
            
        # executing
        if act == 1 and self.pos_item is None and chosen_item is None:
            # BUY: no tax
            self.pos_units = self.cash / chosen_price
            self.cash = 0.0
            self.pos_item = chosen_item
            self.held_last_price = chosen_price
            
        elif act == 2 and self.pos_item is not None:
            # SELL current holding at current snapshot price (if available)
            row = snap[snap["item_id"] == self.pos_item]
            if len(row) > 0:
                sell_price = float(row.iloc[0]["mid"])
                gross = self.pos_units * sell_price
                self.cash = self.rules.sell_net_proceeds(gross)
                self.pos_units = 0.0
                self.pos_item = None
                self.held_last_price = None            
                
        # advance time
        self.t += 1
        terminated = (self.t >= self.t0 + self.cfg.episode_len) or (self.t >= len(self.times) - 1)
        
        next_snap = self._snapshot(self.t)
        new_worth = self._mark_to_market(next_snap)
        
        reward = float(new_worth - prev_worth)
        obs = self._get_obs()
        info = {"net_worth": new_worth, "t": self.t}
        
        return obs, reward, terminated, False, info