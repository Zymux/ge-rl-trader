# scripts/evalPPO.py
from __future__ import annotations

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

from src.geEnv import GEEnv, GEEnvConfig


def make_env():
    cfg = GEEnvConfig(episode_len=75, max_candidates=25, max_spread_pct=0.05, fee_sell_tax=0.02)
    env = GEEnv(cfg)
    return Monitor(env)


def main():
    env = DummyVecEnv([make_env])
    env = VecNormalize.load("docs/assets/ppo_ge_trader/vecnormalize.pkl", env)
    env.training = False
    env.norm_reward = False

    model = PPO.load("docs/assets/ppo_ge_trader/model", env=env)

    obs = env.reset()
    total = 0.0

    for _ in range(75):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        total += float(reward[0])
        if done[0]:
            break

    # Monitor puts episode info in info dict at end
    print("Total eval return:", total)


if __name__ == "__main__":
    main()