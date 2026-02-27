# Post-training verification: 3 learning checks + baselines.
# Run on out-of-sample data when possible (--eval-ts path different from training).
# Usage: python -m scripts.verifyLearning [--eval-ts path] [--n-episodes N]

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.geEnv import GEEnv, GEEnvConfig

STARTING_CASH = 10_000_000.0
EPISODE_LEN = 75
MIN_TIMESTAMPS = EPISODE_LEN + 1  # env needs this many unique parsed_utc
MAX_CANDIDATES = 25
OBS_DIM_PER_ITEM = 4  # mid_norm, logret, vol_5, spread_pct
DEFAULT_TS = "data/timeseries/ge_item_timeseries.csv"


def _count_timestamps(csv_path: Path) -> int:
    df = pd.read_csv(csv_path, parse_dates=["parsed_utc"], usecols=["parsed_utc"])
    return df["parsed_utc"].nunique()


def make_env(ts_path: str):
    cfg = GEEnvConfig(
        ts_path=ts_path,
        episode_len=EPISODE_LEN,
        starting_cash=STARTING_CASH,
    )
    return GEEnv(cfg)


def run_policy(venv, model, n_episodes: int):
    """Run trained policy; return final_equities, trade_counts, blocked_reasons (all episodes)."""
    final_equities = []
    trade_counts = []
    all_blocked = []
    for _ in range(n_episodes):
        obs = venv.reset()
        done_arr = np.array([False])
        trades = 0
        last_info = None
        while not done_arr.any():
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done_arr, info = venv.step(action)
            info0 = info[0]
            last_info = info0
            ex = info0.get("executed", "NONE")
            if ex in ("BUY", "SELL"):
                trades += 1
            if info0.get("blocked_reason"):
                all_blocked.append(info0["blocked_reason"])
        if last_info and "net_worth" in last_info:
            final_equities.append(last_info["net_worth"] / STARTING_CASH)
        trade_counts.append(trades)
    return final_equities, trade_counts, all_blocked


def run_baseline_hold(venv, n_episodes: int):
    """Always HOLD (action type 0, neutral price/qty indices)."""
    final_equities = []
    for _ in range(n_episodes):
        obs = venv.reset()
        done_arr = np.array([False])
        last_info = None
        while not done_arr.any():
            # HOLD: candidate 0, act_type=0, center price_offset_idx, max qty_idx (unused)
            action = np.array([[0, 0, 3, 2]])
            obs, _, done_arr, info = venv.step(action)
            last_info = info[0]
        if last_info and "net_worth" in last_info:
            final_equities.append(last_info["net_worth"] / STARTING_CASH)
    return final_equities


def run_baseline_random(venv, n_episodes: int, seed: int = 42):
    """Random valid actions: BUY only when flat, SELL only when in position."""
    rng = np.random.default_rng(seed)
    final_equities = []
    for _ in range(n_episodes):
        obs = venv.reset()
        done_arr = np.array([False])
        last_info = None
        while not done_arr.any():
            in_position = (last_info or {}).get("in_position", False) if last_info else False
            if in_position:
                action_type = rng.choice([0, 2])  # HOLD or PLACE_SELL (no PLACE_BUY when in position)
            else:
                action_type = rng.choice([0, 1])  # HOLD or PLACE_BUY
            cand = rng.integers(0, MAX_CANDIDATES)
            price_offset_idx = rng.integers(0, 7)
            qty_idx = rng.integers(0, 3)
            action = np.array([[cand, action_type, int(price_offset_idx), int(qty_idx)]])
            obs, _, done_arr, info = venv.step(action)
            last_info = info[0]
        if last_info and "net_worth" in last_info:
            final_equities.append(last_info["net_worth"] / STARTING_CASH)
    return final_equities


def run_baseline_heuristic(venv, n_episodes: int, momentum: bool = True):
    """Simple heuristic: momentum = buy when logret>0, sell when logret<0. Uses candidate 0's logret (obs index 1)."""
    final_equities = []
    for _ in range(n_episodes):
        obs = venv.reset()
        done_arr = np.array([False])
        last_info = None
        while not done_arr.any():
            in_position = (last_info or {}).get("in_position", False) if last_info else False
            logret_0 = obs[0][1] if obs.size > 1 else 0.0  # candidate 0 logret
            if momentum:
                buy_signal = logret_0 > 0
                sell_signal = logret_0 < 0
            else:
                buy_signal = logret_0 < 0
                sell_signal = logret_0 > 0
            # Use center price offset (idx=3 -> 0%) and full qty (idx=2 -> 1.0)
            if in_position and sell_signal:
                action = np.array([[0, 2, 3, 2]])  # PLACE_SELL current
            elif not in_position and buy_signal:
                action = np.array([[0, 1, 3, 2]])  # PLACE_BUY candidate 0
            else:
                action = np.array([[0, 0, 3, 2]])  # HOLD
            obs, _, done_arr, info = venv.step(action)
            last_info = info[0]
        if last_info and "net_worth" in last_info:
            final_equities.append(last_info["net_worth"] / STARTING_CASH)
    return final_equities


def main():
    parser = argparse.ArgumentParser(description="Verify learning: 3 checks + baselines (use --eval-ts for OOS).")
    parser.add_argument("--eval-ts", type=str, default="data/timeseries/ge_item_timeseries.csv",
                        help="Timeseries path for eval (use different from training for out-of-sample).")
    parser.add_argument("--n-episodes", type=int, default=50, help="Episodes per run.")
    parser.add_argument("--model", type=str, default="docs/assets/ppo_ge_trader/model.zip", help="PPO model path.")
    parser.add_argument("--vecnorm", type=str, default="docs/assets/ppo_ge_trader/vecnormalize.pkl", help="VecNormalize path.")
    args = parser.parse_args()

    ts_path = args.eval_ts
    n = args.n_episodes
    model_path = Path(args.model)
    vecnorm_path = Path(args.vecnorm)

    if not Path(ts_path).exists():
        print(f"Eval timeseries not found: {ts_path}")
        return
    if not model_path.exists():
        print(f"Model not found: {model_path}. Train first.")
        return
    if not vecnorm_path.exists():
        print(f"VecNormalize not found: {vecnorm_path}")
        return

    # Env needs at least episode_len+1 timestamps; fall back to full timeseries if --eval-ts has too few
    n_ts = _count_timestamps(Path(ts_path))
    if n_ts < MIN_TIMESTAMPS:
        fallback = Path(DEFAULT_TS)
        if fallback.exists() and fallback.resolve() != Path(ts_path).resolve():
            print(f"'{ts_path}' has only {n_ts} timestamps (need {MIN_TIMESTAMPS}). Using full timeseries: {DEFAULT_TS}")
            ts_path = DEFAULT_TS
        else:
            print(f"Error: '{ts_path}' has only {n_ts} timestamps. Need >= {MIN_TIMESTAMPS} (episode_len+1).")
            print("Use the full timeseries or the *eval* split (ge_item_timeseries_eval.csv), not the train split.")
            return
    elif ts_path == DEFAULT_TS:
        print("Note: Using default eval-ts. For out-of-sample, use --eval-ts data/timeseries/ge_item_timeseries_eval.csv\n")

    venv = DummyVecEnv([lambda: make_env(ts_path)])
    venv = VecNormalize.load(str(vecnorm_path), venv)
    venv.training = False
    venv.norm_reward = False

    model = PPO.load(str(model_path))

    print("=" * 60)
    print("1) POLICY (trained PPO)")
    print("=" * 60)
    eq, trades, blocked = run_policy(venv, model, n)
    eq_arr = np.array(eq)
    trade_arr = np.array(trades)
    print(f"   Final equity: mean={eq_arr.mean():.4f} std={eq_arr.std():.4f} min={eq_arr.min():.4f} max={eq_arr.max():.4f}")
    print(f"   Executed trades/ep: mean={trade_arr.mean():.2f} min={trade_arr.min()} max={trade_arr.max()}")
    if blocked:
        counts = Counter(blocked)
        print(f"   Blocked reasons (total): {dict(counts)}")
    else:
        print("   Blocked reasons: none")

    print()
    print("2) BASELINES (same env, same episodes)")
    print("=" * 60)

    hold_eq = run_baseline_hold(venv, n)
    print(f"   Always HOLD:     mean equity = {np.mean(hold_eq):.4f}")

    rand_eq = run_baseline_random(venv, n)
    print(f"   Random valid:    mean equity = {np.mean(rand_eq):.4f}")

    heur_eq = run_baseline_heuristic(venv, n, momentum=True)
    print(f"   Heuristic (mom): mean equity = {np.mean(heur_eq):.4f}")

    print()
    print("3) LEARNING CHECKS")
    print("=" * 60)
    policy_mean = eq_arr.mean()
    hold_mean = np.mean(hold_eq)
    rand_mean = np.mean(rand_eq)
    heur_mean = np.mean(heur_eq)

    check1 = 0 < trade_arr.mean() < n * EPISODE_LEN * 0.5  # not all HOLD, not constant churn
    print(f"   [1] Trade rate sensible (not all HOLD, not churn): {trade_arr.mean():.1f} executed/ep -> {'OK' if check1 else 'CHECK'}")
    print(f"   [2] Blocked actions: log over training to see if frequency goes down (here: {len(blocked)} total blocks in {n} ep)")
    beats_hold = policy_mean > hold_mean
    beats_rand = policy_mean > rand_mean
    beats_heur = policy_mean > heur_mean
    print(f"   [3] Policy beats HOLD:    {policy_mean:.4f} > {hold_mean:.4f} -> {'YES' if beats_hold else 'NO'}")
    print(f"       Policy beats Random:  {policy_mean:.4f} > {rand_mean:.4f} -> {'YES' if beats_rand else 'NO'}")
    print(f"       Policy beats Heuristic: {policy_mean:.4f} > {heur_mean:.4f} -> {'YES' if beats_heur else 'NO'}")
    print()
    print("For out-of-sample: run with --eval-ts <path_to_different_date_range.csv>")
    print("To track blocked_reason over training: add a callback that logs blocked counts per eval.")


if __name__ == "__main__":
    main()
