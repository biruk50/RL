"""
BipedalWalker-v3: default PPO baseline vs. parameter-matched separate branches.

Baseline:
  - Stable-Baselines3 PPO MlpPolicy defaults (normally [64, 64] policy/value MLPs).

Separate:
  - Current observation and frame-to-frame observation difference enter independent branches.
  - Main branch is approximately 75% of feature-extractor branch parameters.
  - Difference branch is approximately 25%.
  - Total parameters are kept close to the default baseline.
"""

import random
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv


# ------------------------------ Experiment settings ---------------------------

ENV_ID = "BipedalWalker-v3"
SEED = 67
TOTAL_TIMESTEPS = 850_000
N_ENVS = 4
EVAL_FREQ = 10_000
N_EVAL_EPISODES = 5
FINAL_EVAL_EPISODES = 10
PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "bipedalwalker_default_vs_separate"
DEVICE = "auto"

# PPO hyperparameters are identical across the two methods.
LEARNING_RATE = 3e-4
N_STEPS = 2048
BATCH_SIZE = 256
N_EPOCHS = 10

# Chosen to keep the full separate model close to default PPO's ~11,849 params.
# For BipedalWalker-v3 (24 observations, 4 actions), the branch extractor has:
# main branch width 18, difference branch width 8, output features 26.
MAIN_BRANCH_WIDTH = 18
DIFF_BRANCH_WIDTH = 8
SEPARATE_HEAD_WIDTH = 60

torch.set_num_threads(1)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class AddObservationDifference(gym.ObservationWrapper):
    """Emit [observation_t, observation_t - observation_(t-1)]."""

    def __init__(self, env):
        super().__init__(env)
        self.obs_dim = int(np.prod(env.observation_space.shape))
        low = np.concatenate([
            env.observation_space.low.reshape(-1),
            np.full(self.obs_dim, -np.inf, dtype=np.float32),
        ]).astype(np.float32)
        high = np.concatenate([
            env.observation_space.high.reshape(-1),
            np.full(self.obs_dim, np.inf, dtype=np.float32),
        ]).astype(np.float32)
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


def branch_parameter_count(input_dim: int, width: int) -> int:
    # Two Linear layers: input_dim -> width -> width, including biases.
    return input_dim * width + width + width * width + width


class SeparateStateDifferenceExtractor(BaseFeaturesExtractor):
    """Separate MLP branches for current state and frame-to-frame difference."""

    def __init__(
        self,
        observation_space: spaces.Box,
        main_width: int = MAIN_BRANCH_WIDTH,
        diff_width: int = DIFF_BRANCH_WIDTH,
    ):
        total_dim = int(np.prod(observation_space.shape))
        if total_dim % 2:
            raise ValueError(f"Expected [state, difference] with even dimension, got {total_dim}")
        self.state_dim = total_dim // 2
        super().__init__(observation_space, features_dim=main_width + diff_width)

        self.state_branch = torch.nn.Sequential(
            torch.nn.Linear(self.state_dim, main_width),
            torch.nn.Tanh(),
            torch.nn.Linear(main_width, main_width),
            torch.nn.Tanh(),
        )
        self.diff_branch = torch.nn.Sequential(
            torch.nn.Linear(self.state_dim, diff_width),
            torch.nn.Tanh(),
            torch.nn.Linear(diff_width, diff_width),
            torch.nn.Tanh(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        state = observations[..., :self.state_dim]
        difference = observations[..., self.state_dim:]
        return torch.cat(
            [self.state_branch(state), self.diff_branch(difference)],
            dim=-1,
        )


def count_trainable_parameters(model) -> int:
    return sum(p.numel() for p in model.policy.parameters() if p.requires_grad)


def make_env(seed: int, use_difference: bool):
    def _init():
        env = gym.make(ENV_ID)
        if use_difference:
            env = AddObservationDifference(env)
        env = Monitor(env)
        env.reset(seed=seed)
        env.action_space.seed(seed)
        return env
    return _init


def build_model(method: str):
    use_difference = method == "separate"
    train_env = DummyVecEnv([
        make_env(SEED + i, use_difference) for i in range(N_ENVS)
    ])

    if method == "baseline":
        # Intentionally omit policy_kwargs to use SB3's standard default MlpPolicy.
        model = PPO(
            "MlpPolicy",
            train_env,
            learning_rate=LEARNING_RATE,
            n_steps=N_STEPS,
            batch_size=BATCH_SIZE,
            n_epochs=N_EPOCHS,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.0,
            vf_coef=0.5,
            max_grad_norm=0.5,
            seed=SEED,
            device=DEVICE,
            tensorboard_log=str(OUTPUT_DIR / "tensorboard"),
        )
    else:
        policy_kwargs = dict(
            features_extractor_class=SeparateStateDifferenceExtractor,
            features_extractor_kwargs=dict(
                main_width=MAIN_BRANCH_WIDTH,
                diff_width=DIFF_BRANCH_WIDTH,
            ),
            net_arch=dict(
                pi=[SEPARATE_HEAD_WIDTH],
                vf=[SEPARATE_HEAD_WIDTH],
            ),
            activation_fn=torch.nn.Tanh,
        )
        model = PPO(
            "MlpPolicy",
            train_env,
            policy_kwargs=policy_kwargs,
            learning_rate=LEARNING_RATE,
            n_steps=N_STEPS,
            batch_size=BATCH_SIZE,
            n_epochs=N_EPOCHS,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.0,
            vf_coef=0.5,
            max_grad_norm=0.5,
            seed=SEED,
            device=DEVICE,
            tensorboard_log=str(OUTPUT_DIR / "tensorboard"),
        )
    return model, train_env


def evaluate_model(model, use_difference: bool, seed: int, episodes: int):
    eval_env = DummyVecEnv([make_env(seed, use_difference)])
    returns = []
    try:
        for episode in range(episodes):
            obs = eval_env.reset()
            done = np.array([False])
            total_return = 0.0
            while not bool(done[0]):
                action, _ = model.predict(obs, deterministic=True)
                obs, rewards, dones, _ = eval_env.step(action)
                total_return += float(rewards[0])
                done = dones
            returns.append(total_return)
    finally:
        eval_env.close()
    return np.asarray(returns, dtype=np.float64)


def run_experiment(method: str):
    use_difference = method == "separate"
    out_dir = OUTPUT_DIR / method
    out_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(SEED)
    model, train_env = build_model(method)
    actual_params = count_trainable_parameters(model)

    print(f"\n[{method}] total trainable parameters: {actual_params:,}")
    if method == "separate":
        obs_dim = int(np.prod(gym.make(ENV_ID).observation_space.shape))
        main_params = branch_parameter_count(obs_dim, MAIN_BRANCH_WIDTH)
        diff_params = branch_parameter_count(obs_dim, DIFF_BRANCH_WIDTH)
        branch_total = main_params + diff_params
        print(f"  Main branch:       {main_params:,} ({100 * main_params / branch_total:.1f}% of branch params)")
        print(f"  Difference branch: {diff_params:,} ({100 * diff_params / branch_total:.1f}% of branch params)")
        print(f"  Branch widths: main={MAIN_BRANCH_WIDTH}, difference={DIFF_BRANCH_WIDTH}")
        print(f"  Downstream policy/value head width: {SEPARATE_HEAD_WIDTH}")

    eval_env = DummyVecEnv([make_env(SEED + 10_000, use_difference)])
    callback = EvalCallback(
        eval_env,
        best_model_save_path=str(out_dir / "best"),
        log_path=str(out_dir / "eval"),
        eval_freq=max(EVAL_FREQ // N_ENVS, 1),
        n_eval_episodes=N_EVAL_EPISODES,
        deterministic=True,
        render=False,
    )

    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=callback,
            tb_log_name=f"{ENV_ID}_{method}",
            progress_bar=True,
        )
        model.save(str(out_dir / "final_model"))
        returns = evaluate_model(
            model,
            use_difference=use_difference,
            seed=SEED + 20_000,
            episodes=FINAL_EVAL_EPISODES,
        )
        np.savetxt(out_dir / "final_10_episode_returns.csv", returns, delimiter=",")
        print(f"[{method}] final return mean: {returns.mean():.2f}")
        print(f"[{method}] final return std:  {returns.std():.2f}")
        return actual_params, returns
    finally:
        train_env.close()
        eval_env.close()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Show the expected default count before training; actual counts below are authoritative.
    probe = gym.make(ENV_ID)
    obs_dim = int(np.prod(probe.observation_space.shape))
    action_dim = int(np.prod(probe.action_space.shape))
    probe.close()

    # Default SB3 continuous PPO with [64, 64] policy/value MLPs:
    # policy = (obs*64+64) + (64*64+64) + (64*action+action) = 6,020
    # value  = (obs*64+64) + (64*64+64) + (64*1+1) = 5,825
    # log_std = action_dim = 4; total = 11,849 for obs_dim=24, action_dim=4.
    expected_default = (
        (obs_dim * 64 + 64) + (64 * 64 + 64) + (64 * action_dim + action_dim)
        + (obs_dim * 64 + 64) + (64 * 64 + 64) + (64 * 1 + 1)
        + action_dim
    )

    main_branch_params = branch_parameter_count(obs_dim, MAIN_BRANCH_WIDTH)
    diff_branch_params = branch_parameter_count(obs_dim, DIFF_BRANCH_WIDTH)
    print("=" * 76)
    print(f"Environment: {ENV_ID} | observation dim={obs_dim} | action dim={action_dim}")
    print(f"Expected untouched SB3 default PPO parameters: {expected_default:,}")
    print(
        "Separate branch allocation: "
        f"main={main_branch_params:,} ({100 * main_branch_params / (main_branch_params + diff_branch_params):.1f}%), "
        f"difference={diff_branch_params:,} ({100 * diff_branch_params / (main_branch_params + diff_branch_params):.1f}%)"
    )
    print("Training both methods sequentially with the same seed and PPO settings.")
    print("=" * 76)

    # Both runs are enabled. Baseline intentionally runs first as the reference.
    baseline_params, baseline_returns = run_experiment("baseline")
    separate_params, separate_returns = run_experiment("separate")

    print("\n" + "=" * 76)
    print("FINAL COMPARISON")
    print(f"Baseline total trainable parameters: {baseline_params:,}")
    print(f"Separate total trainable parameters: {separate_params:,}")
    print(f"Absolute parameter difference: {abs(baseline_params - separate_params):,}")
    print(f"Baseline final return:  {baseline_returns.mean():.2f} ± {baseline_returns.std():.2f}")
    print(f"Separate final return:  {separate_returns.mean():.2f} ± {separate_returns.std():.2f}")
    print(f"Mean-return difference (separate - baseline): {separate_returns.mean() - baseline_returns.mean():.2f}")
    print("=" * 76)


if __name__ == "__main__":
    main()
