from src.geEnv import GERLTraderEnv, EnvConfig

env = GERLTraderEnv("data/timeseries/ge_item_timeseries.csv", EnvConfig(k_candidates=25, episode_len=50))
obs, info = env.reset()

total = 0.0
for _ in range(50):
    action = env.action_space.sample()
    obs, reward, done, trunc, info = env.step(action)
    total += reward
    if done:
        break

print("Random policy total reward:", total)
print("Final net worth:", info.get("net_worth"))