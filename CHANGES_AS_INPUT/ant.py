import os
# Limit CPU math-library threads to avoid oversubscription on Windows.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import time
from pathlib import Path

import numpy as np
import gymnasium as gym
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    EvalCallback,
)
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecMonitor,
)


# ============================================================
# CONFIGURATION
# ============================================================

ENV_ID = "Ant-v5"

N_ENVS = 4
TOTAL_TIMESTEPS = 600_000

SEED = 42
EVAL_SEED = 10_000

EVAL_FREQ = 8_000
N_EVAL_EPISODES = 3
FINAL_EVAL_EPISODES = 10

LEARNING_RATE = 4e-4
N_STEPS = 2048
BATCH_SIZE = 256
N_EPOCHS = 10

GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_RANGE = 0.2
ENT_COEF = 0.0
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5

DEVICE = "cpu"

PROJECT_DIR = Path(__file__).resolve().parent
LOG_DIR = PROJECT_DIR / "ant_ppo_tensorboard_log"
MODEL_DIR = PROJECT_DIR / "ant_ppo_models"

LOG_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# OBSERVATION WRAPPER: APPEND RAW FIRST DIFFERENCES
# ============================================================

class AddRawObservationDifference(gym.ObservationWrapper):
    """
    Output:
        [current observation, current observation - previous observation]

    Differences are not normalized or clipped.
    The first difference after reset is zero.
    """

    def __init__(self, env):
        super().__init__(env)

        base_space = env.observation_space

        if not isinstance(base_space, gym.spaces.Box):
            raise TypeError("Expected a Box observation space.")

        self.previous_observation = None
        self.base_dim = int(np.prod(base_space.shape))

        base_low = np.asarray(
            base_space.low, dtype=np.float32
        ).reshape(-1)

        base_high = np.asarray(
            base_space.high, dtype=np.float32
        ).reshape(-1)

        # Difference features can exceed the original observation bounds.
        low = np.concatenate([
            base_low,
            np.full(self.base_dim, -np.inf, dtype=np.float32),
        ])

        high = np.concatenate([
            base_high,
            np.full(self.base_dim, np.inf, dtype=np.float32),
        ])

        self.observation_space = gym.spaces.Box(
            low=low,
            high=high,
            dtype=np.float32,
        )

    def observation(self, observation):
        current = np.asarray(
            observation, dtype=np.float32
        ).reshape(-1)

        if self.previous_observation is None:
            difference = np.zeros_like(
                current, dtype=np.float32
            )
        else:
            difference = current - self.previous_observation

        self.previous_observation = current.copy()

        return np.concatenate([
            current,
            difference,
        ]).astype(np.float32, copy=False)

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)

        self.previous_observation = None

        return self.observation(observation), info


# ============================================================
# ENVIRONMENT FACTORIES
# ============================================================

def make_env_fn(add_difference, seed, rank):
    """Create a picklable environment factory for Windows."""

    def _init():
        env = gym.make(ENV_ID)

        if add_difference:
            env = AddRawObservationDifference(env)

        env.reset(seed=seed + rank)

        return env

    return _init


def make_training_env(add_difference, seed):
    env_fns = [
        make_env_fn(add_difference, seed, rank)
        for rank in range(N_ENVS)
    ]

    # Windows-safe multiprocessing.
    env = SubprocVecEnv(
        env_fns,
        start_method="spawn",
    )

    return VecMonitor(env)


def make_eval_env(add_difference, seed):
    def _init():
        env = gym.make(ENV_ID)

        if add_difference:
            env = AddRawObservationDifference(env)

        env.reset(seed=seed)

        return env

    # Use the same outer wrapper type as the training environment.
    env = DummyVecEnv([_init])

    return VecMonitor(env)


# ============================================================
# TRAINING SPEED CALLBACK
# ============================================================

class SpeedCallback(BaseCallback):

    def __init__(self, report_every=25_000):
        super().__init__()
        self.report_every = report_every
        self.last_time = None
        self.last_steps = 0

    def _on_training_start(self):
        self.last_time = time.perf_counter()
        self.last_steps = self.num_timesteps

    def _on_step(self):
        progressed = self.num_timesteps - self.last_steps

        if progressed >= self.report_every:
            now = time.perf_counter()
            elapsed = max(now - self.last_time, 1e-9)

            speed = progressed / elapsed

            print(
                f"[{self.num_timesteps:,} steps] "
                f"Speed: {speed:,.1f} transitions/sec"
            )

            self.last_time = now
            self.last_steps = self.num_timesteps

        return True


# ============================================================
# RUN ONE EXPERIMENT
# ============================================================

def run_experiment(name, add_difference):

    print("\n" + "=" * 70)
    print(f"Experiment: {name}")
    print(f"Difference features enabled: {add_difference}")
    print(f"Parallel environments: {N_ENVS}")
    print(f"Training steps: {TOTAL_TIMESTEPS:,}")
    print(f"Device: {DEVICE}")
    print("=" * 70)

    train_env = make_training_env(
        add_difference=add_difference,
        seed=SEED,
    )

    eval_env = make_eval_env(
        add_difference=add_difference,
        seed=EVAL_SEED,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(
            MODEL_DIR / f"{name}_best"
        ),
        log_path=str(
            LOG_DIR / f"{name}_eval"
        ),
        eval_freq=max(EVAL_FREQ // N_ENVS, 1),
        n_eval_episodes=N_EVAL_EPISODES,
        deterministic=True,
        render=False,
        verbose=1,
    )

    model = PPO(
        policy="MlpPolicy",
        env=train_env,

        learning_rate=LEARNING_RATE,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,

        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        clip_range=CLIP_RANGE,

        ent_coef=ENT_COEF,
        vf_coef=VF_COEF,
        max_grad_norm=MAX_GRAD_NORM,

        policy_kwargs={
            "net_arch": {
                "pi": [256, 256],
                "vf": [256, 256],
            }
        },

        tensorboard_log=str(LOG_DIR),
        seed=SEED,
        device=DEVICE,
    )

    start_time = time.perf_counter()

    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=CallbackList([
                SpeedCallback(),
                eval_callback,
            ]),
            log_interval=10,
            tb_log_name=name,
            progress_bar=False,
        )

        # Save the final model.
        final_path = MODEL_DIR / f"{name}_final"

        model.save(str(final_path))

        # Final evaluation using the same evaluation environment.
        mean_reward, std_reward = evaluate_policy(
            model,
            eval_env,
            n_eval_episodes=FINAL_EVAL_EPISODES,
            deterministic=True,
            render=False,
        )

        elapsed_minutes = (
            time.perf_counter() - start_time
        ) / 60.0

        print(f"\nResults for {name}")
        print(
            f"Mean evaluation reward: "
            f"{mean_reward:.2f} +/- {std_reward:.2f}"
        )
        print(f"Elapsed time: {elapsed_minutes:.1f} minutes")
        print(f"Final model saved: {final_path}.zip")

    finally:
        train_env.close()
        eval_env.close()

def main():

    torch.set_num_threads(1)

    print("Device:", DEVICE)
    print("CPU threads for PyTorch:", torch.get_num_threads())
    print("Parallel workers:", N_ENVS)

    # Check that Ant and its MuJoCo dependencies are installed.
    test_env = gym.make(ENV_ID)
    observation, _ = test_env.reset(seed=SEED)
    print("Base observation shape:", observation.shape)
    test_env.close()

    # Experiment 1: original Ant observation.
    run_experiment(
        name="ppo_ant_standard",
        add_difference=False,
    )

    # Experiment 2: original observation + raw first difference.
    run_experiment(
        name="ppo_ant_with_raw_difference",
        add_difference=True,
    )

    print("\nBoth experiments completed.")
    print("Models directory:", MODEL_DIR.resolve())
    print("TensorBoard directory:", LOG_DIR.resolve())
    print(
        "View TensorBoard using: "
        f"tensorboard --logdir {LOG_DIR}"
    )


if __name__ == "__main__":
    main()