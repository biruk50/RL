
from pathlib import Path
import random

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecMonitor,
)
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn


ENV_ID = "Walker2d-v5"
SEED = 42

CONTINUE_TIMESTEPS = 200_000
N_ENVS = 4

EVAL_FREQ = 10_000
N_EVAL_EPISODES = 10

PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "walker2d_ppo_parameter_matched"
BASE_DIR = OUTPUT_DIR / "baseline"
SEPARATE_DIR = OUTPUT_DIR / "separate_branches"

DEVICE = "cpu"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
BASE_DIR.mkdir(parents=True, exist_ok=True)
SEPARATE_DIR.mkdir(parents=True, exist_ok=True)

torch.set_num_threads(1)


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ============================================================
# Observation wrapper
# Output: [current observation, observation difference]
# ============================================================

class ObservationDifferenceWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

        original_space = env.observation_space
        assert isinstance(original_space, spaces.Box)

        self.obs_dim = int(np.prod(original_space.shape))

        low = np.concatenate([original_space.low, original_space.low])
        high = np.concatenate([original_space.high, original_space.high])

        self.observation_space = spaces.Box(
            low=low.astype(np.float32),
            high=high.astype(np.float32),
            dtype=np.float32,
        )

        self.previous_obs = None

    def observation(self, observation):
        observation = np.asarray(observation, dtype=np.float32).reshape(-1)

        if self.previous_obs is None:
            difference = np.zeros_like(observation)
        else:
            difference = observation - self.previous_obs

        self.previous_obs = observation.copy()

        return np.concatenate([observation, difference]).astype(np.float32)

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self.previous_obs = None
        return self.observation(observation), info


# ============================================================
# Separate feature extractor
#
# State branch:      17 -> 42 -> 42 features
# Difference branch: 17 -> 14 -> 14 features
# Combined:          42 + 14 = 56 features
#
# Feature split: 75% state / 25% difference
# ============================================================

class SeparateStateDifferenceExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box):
        obs_dim = observation_space.shape[0] // 2

        super().__init__(observation_space, features_dim=56)

        self.state_branch = nn.Sequential(
            nn.Linear(obs_dim, 42),
            nn.Tanh(),
            nn.Linear(42, 42),
            nn.Tanh(),
        )

        self.difference_branch = nn.Sequential(
            nn.Linear(obs_dim, 14),
            nn.Tanh(),
            nn.Linear(14, 14),
            nn.Tanh(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        obs_dim = observations.shape[1] // 2

        state = observations[:, :obs_dim]
        difference = observations[:, obs_dim:]

        state_features = self.state_branch(state)
        difference_features = self.difference_branch(difference)

        return torch.cat(
            [state_features, difference_features],
            dim=1,
        )


# ============================================================
# Environment factories
# ============================================================

def make_train_env(seed: int, use_difference: bool):
    def _init():
        env = gym.make(ENV_ID)

        if use_difference:
            env = ObservationDifferenceWrapper(env)

        env.reset(seed=seed)
        return env

    return _init


def make_eval_env(seed: int, use_difference: bool):
    def _init():
        env = gym.make(ENV_ID)

        if use_difference:
            env = ObservationDifferenceWrapper(env)

        env.reset(seed=seed)
        return env

    return _init


# ============================================================
# Parameter counting
# ============================================================

def count_parameters(model):
    return sum(
        parameter.numel()
        for parameter in model.policy.parameters()
        if parameter.requires_grad
    )


# ============================================================
# Train one model
# ============================================================

def train_model(name: str, output_dir: Path, use_difference: bool):
    print(f"\n{'=' * 65}")
    print(f"Training: {name}")
    print(f"{'=' * 65}")

    seed_everything(SEED)

    # Use subprocess environments for parallel training.
    train_env = SubprocVecEnv(
        [
            make_train_env(SEED + i, use_difference)
            for i in range(N_ENVS)
        ]
    )
    train_env = VecMonitor(train_env)

    eval_env = VecMonitor(
        DummyVecEnv(
            [make_eval_env(SEED + 10_000, use_difference)]
        )
    )

    model_path = output_dir / "final_model.zip"
    model = PPO.load(str(model_path), env=train_env, device=DEVICE)
    parameter_count = count_parameters(model)

    print(f"Resuming from: {model_path}")
    print(f"Current timesteps: {model.num_timesteps:,}")
    print(f"Additional timesteps: {CONTINUE_TIMESTEPS:,}")
    print(f"{name} trainable parameters: {parameter_count:,}")

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(output_dir / "best_continued"),
        log_path=str(output_dir / "evaluations_continued"),
        eval_freq=max(EVAL_FREQ // N_ENVS, 1),
        n_eval_episodes=N_EVAL_EPISODES,
        deterministic=True,
        render=False,
    )

    try:
        model.learn(
            total_timesteps=CONTINUE_TIMESTEPS,
            callback=eval_callback,
            tb_log_name=name,
            reset_num_timesteps=False,
            progress_bar=True,
        )

        model.save(str(output_dir / "final_model"))

        print(f"\nFinal model saved to: {output_dir / 'final_model.zip'}")
        print(f"Best continuation model directory: {output_dir / 'best_continued'}")

    finally:
        train_env.close()
        eval_env.close()

    return parameter_count


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    baseline_params = train_model(
        name="baseline_ppo",
        output_dir=BASE_DIR,
        use_difference=False,
    )

    separate_params = train_model(
        name="separate_state_difference",
        output_dir=SEPARATE_DIR,
        use_difference=True,
    )

    print("\n" + "=" * 65)
    print("PARAMETER COUNT SUMMARY")
    print("=" * 65)
    print(f"Baseline PPO:      {baseline_params:,}")
    print(f"Separate branches: {separate_params:,}")
    print(
        "Parameter difference: "
        f"{abs(baseline_params - separate_params):,}"
    )

    print("\nTraining and evaluation finished.")
