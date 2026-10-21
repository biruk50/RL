"""
PPO comparison on LunarLander-v3:
  1) Baseline: Stable-Baselines3 default MlpPolicy architecture.
  2) Separate 75/25 branch model: 75% of branch hidden units go to the
     observation branch, 25% to the observation-difference branch.

The custom policy is sized to approximately match the baseline's trainable
parameter count. Exact counts are printed at startup.

"""

from pathlib import Path
import random

import gymnasium as gym
import numpy as np
import torch
from torch import nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from gymnasium import spaces


ENV_ID = "LunarLander-v3"
SEED = 96
TOTAL_TIMESTEPS = 800_000
N_ENVS = 4
EVAL_FREQ = 10_000
N_EVAL_EPISODES = 10
PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = PROJECT_DIR / "lunarlander_ppo_75_25"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ObservationDifferenceWrapper(gym.ObservationWrapper):
    """Return concatenated [observation, observation - previous observation]."""

    def __init__(self, env):
        super().__init__(env)
        original = env.observation_space
        self.obs_dim = int(np.prod(original.shape))
        low_obs = np.asarray(original.low, dtype=np.float32).reshape(-1)
        high_obs = np.asarray(original.high, dtype=np.float32).reshape(-1)
        low = np.concatenate([low_obs, np.full(self.obs_dim, -np.inf, np.float32)])
        high = np.concatenate([high_obs, np.full(self.obs_dim, np.inf, np.float32)])
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.previous_obs = None

    def observation(self, observation):
        obs = np.asarray(observation, dtype=np.float32).reshape(-1)
        delta = np.zeros_like(obs) if self.previous_obs is None else obs - self.previous_obs
        self.previous_obs = obs.copy()
        return np.concatenate([obs, delta]).astype(np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.previous_obs = None
        return self.observation(obs), info


class Separate75_25Extractor(BaseFeaturesExtractor):
    """
    75/25 feature split for LunarLander's 8-dimensional observation:
      main branch: 8 -> 42 -> 42
      difference branch: 8 -> 14 -> 14
    Output: 56 features total, concatenated [main(42), difference(14)].
    """

    def __init__(self, observation_space: spaces.Box):
        # ObservationDifferenceWrapper doubles the original 8-D observation.
        super().__init__(observation_space, features_dim=56)
        input_dim = observation_space.shape[0] // 2
        self.main_branch = nn.Sequential(
            nn.Linear(input_dim, 42),
            nn.Tanh(),
            nn.Linear(42, 42),
            nn.Tanh(),
        )
        self.diff_branch = nn.Sequential(
            nn.Linear(input_dim, 14),
            nn.Tanh(),
            nn.Linear(14, 14),
            nn.Tanh(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        n = observations.shape[1] // 2
        current = observations[:, :n]
        difference = observations[:, n:]
        return torch.cat(
            [self.main_branch(current), self.diff_branch(difference)], dim=1
        )


def count_trainable_parameters(model) -> int:
    return sum(p.numel() for p in model.policy.parameters() if p.requires_grad)


def make_env(with_difference: bool, seed: int):
    def _init():
        env = gym.make(ENV_ID)
        if with_difference:
            env = ObservationDifferenceWrapper(env)
        env = Monitor(env)
        env.reset(seed=seed)
        return env
    return _init


def build_model(env, separate: bool, seed: int, tensorboard_log: str):
    if not separate:
        # Untouched Stable-Baselines3 default MlpPolicy architecture.
        return PPO(
            "MlpPolicy",
            env,
            seed=seed,
            tensorboard_log=tensorboard_log,
            device="auto",
        )

    policy_kwargs = dict(
        features_extractor_class=Separate75_25Extractor,
        # SB3 applies these head layers separately to policy and value paths.
        # 42 main features + 14 difference features gives an exact 75/25 split.
        # Width 61 keeps total trainable parameters almost equal to the baseline.
        net_arch=dict(pi=[61], vf=[61]),
        activation_fn=nn.Tanh,
    )
    return PPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        seed=seed,
        tensorboard_log=tensorboard_log,
        device="auto",
    )


def train_one(name: str, separate: bool):
    run_dir = OUTPUT_ROOT / name
    best_dir = run_dir / "best"
    logs_dir = run_dir / "eval_logs"
    tensorboard_dir = run_dir / "tensorboard"
    for directory in (run_dir, best_dir, logs_dir, tensorboard_dir):
        directory.mkdir(parents=True, exist_ok=True)

    train_env = make_vec_env(
        make_env(separate, SEED),
        n_envs=N_ENVS,
        seed=SEED,
        vec_env_cls=DummyVecEnv,
    )
    eval_env = DummyVecEnv([make_env(separate, SEED + 10_000)])

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(best_dir),
        log_path=str(logs_dir),
        eval_freq=max(EVAL_FREQ // N_ENVS, 1),
        n_eval_episodes=N_EVAL_EPISODES,
        deterministic=True,
        render=False,
    )

    model = build_model(
        train_env,
        separate,
        SEED,
        str(tensorboard_dir),
    )

    print(f"\n{'=' * 72}")
    print(f"Training: {name}")
    print(f"Trainable policy parameters: {count_trainable_parameters(model):,}")
    print(f"{'=' * 72}\n")

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=eval_callback,
        progress_bar=True,
        tb_log_name=name,
    )
    model.save(str(run_dir / "final_model"))
    print(f"Final model saved to: {run_dir / 'final_model.zip'}")
    print(f"Best model saved under: {best_dir / 'best_model.zip'}")

    train_env.close()
    eval_env.close()


def main():
    seed_everything(SEED)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    # Baseline default architecture has 8 inputs, 4 actions, and default
    # [64, 64] policy/value MLPs. Print its actual count rather than assuming.
    print(f"Environment: {ENV_ID}")
    print(f"Training steps per variant: {TOTAL_TIMESTEPS:,}")
    print(f"Parallel environments: {N_ENVS}")
    print("Variants use the same PPO defaults and training budget.")

    train_one("baseline", separate=False)
    train_one("separate_75_25", separate=True)
    print("\nTraining complete. Best and final checkpoints were saved for both.")


if __name__ == "__main__":
    main()
