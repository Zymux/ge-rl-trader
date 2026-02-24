# forced test to confirm env changes net worth

from src.geEnv import GERLTraderEnv, EnvConfig

cfg = EnvConfig(k_candidates=25, episode_len=10, starting_cash=10_000_000)
env = GERLTraderEnv("data/timeseries/ge_item_timeseries.csv", cfg)

# Use deterministic start if your geEnv supports it:
obs, info = env.reset(options={"start_idx": 0})

snap0 = env._snapshot(env.t)
item0 = int(snap0.iloc[0]["item_id"])
buy_price = float(snap0.iloc[0]["mid"])
print("BUY candidate0 item:", item0, "price:", buy_price)

# BUY
obs, reward, done, trunc, info = env.step([0, 1])
print("After BUY -> cash:", env.cash, "pos_item:", env.pos_item, "pos_units:", env.pos_units)
print("Net worth after BUY:", info.get("net_worth"))

# HOLD a few steps (must be < episode_len)
for _ in range(3):
    obs, reward, done, trunc, info = env.step([0, 0])
    if done:
        print("Episode ended during HOLD; cannot SELL.")
        break

if not done:
    # SELL (note: this will only work if your env's SELL semantics sell the held item,
    # or if candidate 0 still corresponds to the held item at this timestep)
    obs, reward, done, trunc, info = env.step([0, 2])
    print("After SELL -> cash:", env.cash, "pos_item:", env.pos_item, "pos_units:", env.pos_units)
    print("Final net worth:", info.get("net_worth"))