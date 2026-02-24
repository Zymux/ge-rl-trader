from pathlib import Path
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from src.geEnv import GERLTraderEnv, EnvConfig

ASSETS_DIR = Path("docs") / "assets"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

def make_env():
    env = GERLTraderEnv(
        "data/timeseries/ge_item_timeseries.csv",
        EnvConfig(k_candidates=25, episode_len=75, starting_cash=10_000_000),
    )
    return Monitor(env)

def main():
    vec_env = DummyVecEnv([make_env])
    
    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        n_steps=1024,
        batch_size=256,
        gamma=0.99,
        learning_rate=3e-4,
        ent_coef=0.01,
    )
    
    model.learn(total_timesteps=100_000)
    
    model_path = ASSETS_DIR / "ppo_ge_trader"
    model.save(model_path)
    print("Saved model:", model_path)
    
    returns = vec_env.envs[0].get_episode_rewards()
    if len(returns) > 0:
        plt.figure()
        plt.plot(returns)
        plt.xlabel("Episode")
        plt.ylabel("Episode return (Δ net worth, gp)")
        plt.title("PPO Training Returns")
        plt.tight_layout()
        out = ASSETS_DIR / "ppo_training_returns.png"
        plt.savefig(out, dpi=200)
        plt.close()
        print("Saved plot:", out)
    else:
        print("No episode returns logged yet (try more timesteps).")


if __name__ == "__main__":
    main()