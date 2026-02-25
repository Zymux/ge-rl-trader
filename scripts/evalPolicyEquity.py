# Evaluate policy by final equity (not just reward).
# Requires a time-series CSV: multiple timestamps and varying mid prices so PnL can occur.

import numpy as np
import pandas as pd
from collections import Counter
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from src.geEnv import GEEnv, GEEnvConfig

TS_PATH = "data/timeseries/ge_item_timeseries.csv"
STARTING_CASH = 10_000_000.0
EPISODE_LEN = 75


def make_env():
    cfg = GEEnvConfig(
        ts_path=TS_PATH,
        episode_len=EPISODE_LEN,
        starting_cash=STARTING_CASH,
    )
    return GEEnv(cfg)


def check_timeseries(ts_path: str) -> None:
    """Ensure the timeseries has multiple timestamps and varying prices (required for equity to move)."""
    path = Path(ts_path)
    if not path.exists():
        raise FileNotFoundError(f"Timeseries not found: {ts_path}. Evaluation needs real time-series data.")
    df = pd.read_csv(path, parse_dates=["parsed_utc"], nrows=50000)
    df = df.dropna(subset=["parsed_utc", "item_id", "mid"])
    times = df["parsed_utc"].unique()
    n_ts = len(times)
    if n_ts < 2:
        raise ValueError(
            f"Timeseries has only {n_ts} timestamp(s). Need multiple timestamps so prices can change over time."
        )
    # Check that mid varies across time (e.g. same item at different times has different mid)
    by_time = df.groupby("parsed_utc")["mid"].agg(["min", "max", "mean"])
    if by_time["max"].std() < 1e-9 and by_time["min"].std() < 1e-9:
        raise ValueError(
            "Timeseries mid prices do not vary across timestamps. Equity cannot change with static prices."
        )
    print(f"[OK] Timeseries: {n_ts} timestamps, mid varies across time.")


def main():
    check_timeseries(TS_PATH)

    model = PPO.load("docs/assets/ppo_ge_trader/model.zip")
    venv = DummyVecEnv([make_env])
    venv = VecNormalize.load("docs/assets/ppo_ge_trader/vecnormalize.pkl", venv)
    venv.training = False
    venv.norm_reward = False

    n_episodes = 50
    final_equities = []
    trade_counts = []

    TRACE_STEPS = 20  # first N steps of Episode 0 to print

    for ep in range(n_episodes):
        obs = venv.reset()
        done_arr = np.array([False])
        trades = 0
        last_info = None
        step_equities = []
        # Episode 0 debug: per-step trace and stats
        step_trace = []
        inventory_values = []
        traded_item_mids = []
        # Episode 0: count requested vs executed and blocked reasons
        requested_non_hold = 0
        executed_buys = 0
        executed_sells = 0
        blocked_reasons = []

        while not done_arr.any():
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done_arr, info = venv.step(action)
            info0 = info[0]
            last_info = info0
            # Count only executed trades (BUY/SELL that actually ran)
            ex = info0.get("executed", "NONE")
            if ex == "BUY":
                executed_buys += 1
                trades += 1
            elif ex == "SELL":
                executed_sells += 1
                trades += 1
            if info0.get("action", 0) != 0:
                requested_non_hold += 1
            if info0.get("blocked_reason"):
                blocked_reasons.append(info0["blocked_reason"])
            if "net_worth" in info0:
                step_equities.append(info0["net_worth"])
            # Collect Episode 0 trace (first TRACE_STEPS only) and stats
            if ep == 0:
                inv_qty = info0.get("pos_units", 0) or 0
                inventory_values.append(inv_qty)
                if info0.get("action", 0) != 0 and "acted_item_id" in info0 and info0.get("acted_item_id") != -1:
                    traded_item_mids.append((info0["acted_item_id"], info0.get("acted_mid")))
                if len(step_trace) < TRACE_STEPS:
                    step_trace.append({
                        "step": len(step_trace),
                        "parsed_utc": info0.get("parsed_utc"),
                        "action": info0.get("action"),
                        "acted_item_id": info0.get("acted_item_id"),
                        "acted_mid": info0.get("acted_mid"),
                        "pos_item_id": info0.get("pos_item_id", info0.get("pos_item", -1)),
                        "pos_mid": info0.get("pos_mid"),
                        "exec_price": info0.get("exec_price"),
                        "realized_pnl": info0.get("realized_pnl"),
                        "cash": info0.get("cash"),
                        "pos_units": info0.get("pos_units") or 0,
                        "net_worth": info0.get("net_worth"),
                    })

        if last_info is not None and "net_worth" in last_info:
            final_equities.append(last_info["net_worth"] / STARTING_CASH)
        trade_counts.append(trades)

        # Episode 0: per-step trace (first TRACE_STEPS) and summary stats
        if ep == 0:
            if step_equities:
                step_arr = np.array(step_equities)
                print(
                    f"[Episode 0] Net worth steps: min={step_arr.min():.0f} max={step_arr.max():.0f} "
                    f"mean={step_arr.mean():.0f} (starting={STARTING_CASH:.0f})"
                )
                if step_arr.max() - step_arr.min() < 1.0:
                    print(
                        "  WARNING: Net worth did not change during the episode. "
                        "Check that env uses advancing timestamps and varying mid prices."
                    )
            # Trace table (first ~20 steps)
            if step_trace:
                print("\n--- Episode 0 trace (first %d steps) ---" % TRACE_STEPS)
                print(
                    f"{'step':>4}  {'parsed_utc':<22}  {'act':>3}  {'pos_item_id':>11}  {'pos_mid':>10}  "
                    f"{'exec_price':>10}  {'realized_pnl':>12}  {'cash':>12}  {'pos_units':>10}  {'net_worth':>12}"
                )
                print("-" * 120)
                for row in step_trace:
                    ts = str(row["parsed_utc"])[:22] if row.get("parsed_utc") is not None else "—"
                    pos_mid = row.get("pos_mid")
                    pos_mid_s = f"{pos_mid:.2f}" if pos_mid is not None else "—"
                    exec_pr = row.get("exec_price")
                    exec_s = f"{exec_pr:.2f}" if exec_pr is not None else "—"
                    rpnl = row.get("realized_pnl")
                    rpnl_s = f"{rpnl:,.0f}" if rpnl is not None else "—"
                    print(
                        f"{row['step']:>4}  {ts:<22}  {row.get('action', -1):>3}  "
                        f"{row.get('pos_item_id', -1):>11}  {pos_mid_s:>10}  {exec_s:>10}  {rpnl_s:>12}  "
                        f"{row.get('cash', 0):>12,.0f}  {row.get('pos_units', 0):>10.2f}  {row.get('net_worth', 0):>12,.0f}"
                    )
            # Summary: fraction of steps with inventory != 0; mid range for traded items
            inv_arr = np.array(inventory_values)
            inventory_nonzero_frac = (inv_arr != 0).mean() if len(inv_arr) else 0.0
            print(f"\n[Episode 0] inventory_nonzero_steps (frac): {inventory_nonzero_frac:.3f}  (1.0 = hold every step)")
            if traded_item_mids:
                item_ids = set(i for i, _ in traded_item_mids)
                mids_per_item = {}
                for iid, mid in traded_item_mids:
                    if iid not in mids_per_item:
                        mids_per_item[iid] = []
                    mids_per_item[iid].append(mid)
                print("[Episode 0] episode_mid_range for traded item(s):")
                for iid in sorted(item_ids):
                    m = mids_per_item[iid]
                    print(f"  item_id={iid}  mid min={min(m):.2f} max={max(m):.2f} (n={len(m)})")
            else:
                print("[Episode 0] episode_mid_range: no traded items (action was always HOLD or acted on padding)")
            # Requested vs executed trades; why BUY/SELL was blocked
            print(
                f"[Episode 0] requested (non-HOLD) actions: {requested_non_hold}  "
                f"executed BUY: {executed_buys}  executed SELL: {executed_sells}  "
                f"(trades = executed only: {executed_buys + executed_sells})"
            )
            if blocked_reasons:
                counts = Counter(blocked_reasons)
                print("[Episode 0] blocked_reason counts:", dict(counts))
            print()

    print(f"Episodes: {n_episodes}")
    print(f"Final equity mean: {np.mean(final_equities):.4f}  std: {np.std(final_equities):.4f}")
    print(f"Final equity min/max: {np.min(final_equities):.4f} / {np.max(final_equities):.4f}")
    print(f"Executed trades per ep mean: {np.mean(trade_counts):.2f}  min/max: {np.min(trade_counts)} / {np.max(trade_counts)}")

if __name__ == "__main__":
    main()