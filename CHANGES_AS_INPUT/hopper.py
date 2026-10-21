
import os
import random
import numpy as np

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecMonitor,
)
import torch


ENV_ID = "Hopper-v5"

SEED = 67
N_ENVS = 4
TOTAL_TIMESTEPS = 800_000

EVAL_FREQ = 10_000
N_EVAL_EPISODES = 3
FINAL_EVAL_EPISODES = 10

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(PROJECT_DIR, "hopper_ppo_tensorboard_log")
MODEL_DIR = os.path.join(PROJECT_DIR, "hopper_ppo_models")

DEVICE = "cpu"


# ============================================================
# REPRODUCIBILITY AND CPU SETTINGS
# ============================================================

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

torch.set_num_threads(1)


# ============================================================
# OBSERVATION DIFFERENCE WRAPPER
# Augmented observation = [current observation,
#                          current observation - previous observation]
# ============================================================

class AddRawObservationDifference(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

        self.previous_observation = None

        original_space = env.observation_space

        if not isinstance(original_space, spaces.Box):
            raise TypeError(
                "This wrapper requires a Box observation space."
            )

        low = np.concatenate(
            [original_space.low, np.full_like(original_space.low, -np.inf)]
        )
        high = np.concatenate(
            [original_space.high, np.full_like(original_space.high, np.inf)]
        )

        self.observation_space = spaces.Box(
            low=low,
            high=high,
            dtype=original_space.dtype,
        )

    def observation(self, observation):
        observation = np.asarray(observation, dtype=np.float32)

        if self.previous_observation is None:
            difference = np.zeros_like(observation)
        else:
            difference = observation - self.previous_observation

        self.previous_observation = observation.copy()

        return np.concatenate(
            [observation, difference]
        ).astype(np.float32)

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self.previous_observation = None

        return self.observation(observation), info

    def step(self, action):
        observation, reward, terminated, truncated, info = (
            self.env.step(action)
        )

        return (
            self.observation(observation),
            reward,
            terminated,
            truncated,
            info,
        )


# ============================================================
# ENVIRONMENT FACTORIES
# ============================================================

def make_train_env(rank, use_difference):
    def _init():
        env = gym.make(ENV_ID)

        if use_difference:
            env = AddRawObservationDifference(env)

        env = Monitor(env)
        env.reset(seed=SEED + rank)

        return env

    return _init


def make_eval_env(use_difference):
    def _init():
        env = gym.make(ENV_ID)

        if use_difference:
            env = AddRawObservationDifference(env)

        env = Monitor(env)
        return env

    return _init


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(model, env, episodes=10):
    episode_rewards = []
    episode_lengths = []

    for episode in range(episodes):
        env.seed(SEED + 10_000 + episode)
        observation = env.reset()

        done = False
        total_reward = 0.0
        length = 0

        while not done:
            action, _ = model.predict(
                observation,
                deterministic=True,
            )

            observation, step_rewards, dones, _ = env.step(action)

            total_reward += float(step_rewards[0])
            length += 1
            done = bool(dones[0])

        episode_rewards.append(total_reward)
        episode_lengths.append(length)

    print(
        f"Evaluation over {episodes} episodes: "
        f"mean reward = {np.mean(episode_rewards):.2f} "
        f"+/- {np.std(episode_rewards):.2f}, "
        f"mean length = {np.mean(episode_lengths):.1f}"
    )

    return episode_rewards, episode_lengths


# ============================================================
# TRAINING
# ============================================================

def train_experiment(name, use_difference):
    print("\n" + "=" * 70)
    print(f"Starting experiment: {name}")
    print(f"Observation difference enabled: {use_difference}")
    print("=" * 70)

    experiment_log_dir = os.path.join(LOG_DIR, name)
    experiment_model_dir = os.path.join(MODEL_DIR, name)

    os.makedirs(experiment_log_dir, exist_ok=True)
    os.makedirs(experiment_model_dir, exist_ok=True)

    # Parallel training environments
    train_env = SubprocVecEnv(
        [
            make_train_env(rank, use_difference)
            for rank in range(N_ENVS)
        ],
        start_method="spawn",
    )

    train_env = VecMonitor(train_env)

    # Evaluation environment
    eval_env = DummyVecEnv(
        [make_eval_env(use_difference)]
    )
    eval_env = VecMonitor(eval_env)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=experiment_model_dir,
        log_path=experiment_log_dir,
        eval_freq=max(EVAL_FREQ // N_ENVS, 1),
        n_eval_episodes=N_EVAL_EPISODES,
        deterministic=True,
        render=False,
        verbose=1,
    )

    policy_kwargs = dict(
        net_arch=dict(
            pi=[256, 256],
            vf=[256, 256],
        )
    )

    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=4e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        tensorboard_log=LOG_DIR,
        seed=SEED,
        device=DEVICE,
    )

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=eval_callback,
        tb_log_name=name,
        progress_bar=True,
    )

    final_model_path = os.path.join(
        experiment_model_dir,
        f"{name}_final",
    )

    model.save(final_model_path)
    print(f"Final model saved to: {final_model_path}")
    print(
        f"Best model, when evaluation improves, is saved under: "
        f"{experiment_model_dir}"
    )

    print(f"\nFinal evaluation: {name}")
    evaluate_model(
        model,
        eval_env,
        episodes=FINAL_EVAL_EPISODES,
    )

    train_env.close()
    eval_env.close()

    del model
    del train_env
    del eval_env

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================
# MAIN
# ============================================================

def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Verify that the required MuJoCo environment is available.
    test_env = gym.make(ENV_ID)
    observation, info = test_env.reset(seed=SEED)

    print(f"Environment: {ENV_ID}")
    print(f"Base observation shape: {observation.shape}")
    print(f"Action shape: {test_env.action_space.shape}")
    print(f"Training device: {DEVICE}")
    print(f"Parallel environments: {N_ENVS}")
    print(f"Timesteps per experiment: {TOTAL_TIMESTEPS:,}")

    test_env.close()

    experiments = [
        ("hopper_base", False),
        ("hopper_base_plus_raw_diff", True),
    ]

    for name, use_difference in experiments:
        train_experiment(name, use_difference)

    print("\nAll Hopper PPO experiments completed.")
    print(f"TensorBoard logs: {LOG_DIR}")
    print(f"Saved models: {MODEL_DIR}")


if __name__ == "__main__":
    main()
