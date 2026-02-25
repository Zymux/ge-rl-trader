# scripts/trainPPO.py
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback

from src.geEnv import GEEnv, GEEnvConfig


ASSETS_DIR = Path("docs") / "assets"
TS_DIR = Path("data") / "timeseries"
TRAIN_TS = TS_DIR / "ge_item_timeseries_train.csv"
DEFAULT_TS = TS_DIR / "ge_item_timeseries.csv"
EPISODE_LEN = 75
MIN_TIMESTAMPS = EPISODE_LEN + 1
ASSETS_DIR.mkdir(parents=True, exist_ok=True)


def _count_timestamps(csv_path: Path) -> int:
    """Number of unique parsed_utc in CSV (env needs >= episode_len + 1)."""
    df = pd.read_csv(csv_path, parse_dates=["parsed_utc"], usecols=["parsed_utc"])
    return df["parsed_utc"].nunique()


def make_env():
    # Use train split if it exists and has enough timestamps, else full timeseries
    if TRAIN_TS.exists():
        n = _count_timestamps(TRAIN_TS)
        if n < MIN_TIMESTAMPS:
            print(f"Train split has {n} timestamps (need {MIN_TIMESTAMPS}); using full timeseries.")
            ts_path = str(DEFAULT_TS)
        else:
            ts_path = str(TRAIN_TS)
    else:
        ts_path = str(DEFAULT_TS)
    cfg = GEEnvConfig(
        ts_path=ts_path,
        episode_len=EPISODE_LEN,
        max_candidates=25,
        max_spread_pct=0.05,
        fee_sell_tax=0.02,
        starting_cash=10_000_000.0,
    )
    env = GEEnv(cfg)
    env = Monitor(env)
    return env


def main():
    # vectorized envs
    train_env = DummyVecEnv([make_env])
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    eval_env = DummyVecEnv([make_env])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    eval_env.training = False

    model_dir = ASSETS_DIR / "ppo_ge_trader"
    best_dir = ASSETS_DIR / "ppo_best"
    model_dir.mkdir(parents=True, exist_ok=True)
    best_dir.mkdir(parents=True, exist_ok=True)

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(best_dir),
        log_path=str(best_dir),
        eval_freq=5_000,
        n_eval_episodes=10,
        deterministic=True,
        render=False,
    )

    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        n_steps=1024,
        batch_size=256,
        learning_rate=3e-4,
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
    )

    model.learn(total_timesteps=300_000, callback=eval_cb)

    # Save model + VecNormalize stats
    model.save(str(model_dir / "model"))
    train_env.save(str(model_dir / "vecnormalize.pkl"))

    print(f"Saved model: {model_dir / 'model.zip'}")
    print(f"Saved VecNormalize: {model_dir / 'vecnormalize.pkl'}")

    # Plot episode rewards from monitor (raw env reward is log-return; vecnormalize changes it during training)
    # EvalCallback logs are usually clearer; we’ll still do a simple plot from eval logs if present.
    eval_file = best_dir / "evaluations.npz"
    out_plot = ASSETS_DIR / "ppo_training_returns.png"

    if eval_file.exists():
        data = np.load(eval_file)
        timesteps = data["timesteps"]
        results = data["results"]  # shape (n_evals, n_eval_episodes)
        mean = results.mean(axis=1)

        plt.figure()
        plt.plot(timesteps, mean)
        plt.xlabel("Timesteps")
        plt.ylabel("Mean eval episode return")
        plt.title("PPO Evaluation Returns")
        plt.tight_layout()
        plt.savefig(out_plot, dpi=200)
        plt.close()
        print(f"Saved plot: {out_plot}")
    else:
        print("No eval plot generated (evaluations.npz not found).")


if __name__ == "__main__":
    main()