
import os

# Limit CPU thread oversubscription on Windows.
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
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecMonitor,
)


# ============================================================
# CONFIGURATION
# ============================================================

ENV_ID = "Swimmer-v5"

N_ENVS = 4
TOTAL_TIMESTEPS = 1_000_000

SEED = 42
EVAL_SEED = 10_000

EVAL_FREQ = 20_000
N_EVAL_EPISODES = 3
FINAL_EVAL_EPISODES = 10

LEARNING_RATE = 3e-4
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
LOG_DIR = PROJECT_DIR / "swimmer_ppo_tensorboard_log"
MODEL_DIR = PROJECT_DIR / "swimmer_ppo_models"

LOG_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# OBSERVATION WRAPPER
# ============================================================

class AddRawObservationDifference(gym.ObservationWrapper):
    """
    Append current observation minus previous observation.

    Differences are raw, unnormalized, and unclipped.
    First difference after reset is zero.
    """

    def __init__(self, env):
        super().__init__(env)

        space = env.observation_space

        if not isinstance(space, gym.spaces.Box):
            raise TypeError("Expected a Box observation space.")

        self.previous_observation = None
        self.base_dim = int(np.prod(space.shape))

        base_low = np.asarray(
            space.low, dtype=np.float32
        ).reshape(-1)

        base_high = np.asarray(
            space.high, dtype=np.float32
        ).reshape(-1)

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

    def _init():
        env = gym.make(ENV_ID)
        env = Monitor(env)

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

    # Spawn is compatible with Windows multiprocessing.
    env = SubprocVecEnv(
        env_fns,
        start_method="spawn",
    )

    return VecMonitor(env)


def make_eval_env(add_difference, seed):

    def _init():
        env = gym.make(ENV_ID)
        env = Monitor(env)

        if add_difference:
            env = AddRawObservationDifference(env)

        env.reset(seed=seed)
        return env

    # Match the outer vector wrapper type used for training.
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

            print(
                f"[{self.num_timesteps:,} steps] "
                f"{progressed / elapsed:,.1f} transitions/sec"
            )

            self.last_time = now
            self.last_steps = self.num_timesteps

        return True


# ============================================================
# RUN EXPERIMENT
# ============================================================

def run_experiment(name, add_difference):

    print("\n" + "=" * 70)
    print(f"Experiment: {name}")
    print(f"Raw difference features: {add_difference}")
    print(f"Parallel environments: {N_ENVS}")
    print(f"Timesteps: {TOTAL_TIMESTEPS:,}")
    print(f"Device: {DEVICE}")
    print("=" * 70)

    train_env = make_training_env(
        add_difference,
        SEED,
    )

    eval_env = make_eval_env(
        add_difference,
        EVAL_SEED,
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

    started = time.perf_counter()

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

        final_path = MODEL_DIR / f"{name}_final"
        model.save(str(final_path))

        mean_reward, std_reward = evaluate_policy(
            model,
            eval_env,
            n_eval_episodes=FINAL_EVAL_EPISODES,
            deterministic=True,
            render=False,
        )

        elapsed_minutes = (
            time.perf_counter() - started
        ) / 60.0

        print(f"\nResults for {name}")
        print(
            f"Mean evaluation reward: "
            f"{mean_reward:.2f} +/- {std_reward:.2f}"
        )
        print(f"Elapsed time: {elapsed_minutes:.1f} minutes")
        print(f"Saved final model: {final_path}.zip")

    finally:
        train_env.close()
        eval_env.close()


# ============================================================
# MAIN
# ============================================================

def main():

    torch.set_num_threads(1)
    print("Device:", DEVICE)
    print("PyTorch CPU threads:", torch.get_num_threads())
    print("Parallel workers:", N_ENVS)

    # Verify that Swimmer and its MuJoCo dependencies are available.
    test_env = gym.make(ENV_ID)
    observation, _ = test_env.reset(seed=SEED)
    print("Base observation shape:", observation.shape)
    test_env.close()

    # Variant 1: standard observations.
    run_experiment(
        name="ppo_swimmer_standard",
        add_difference=False,
    )

    # Variant 2: standard observations + raw first differences.
    run_experiment(
        name="ppo_swimmer_with_raw_difference",
        add_difference=True,
    )

    print("\nBoth Swimmer PPO experiments completed.")
    print("Models:", MODEL_DIR.resolve())
    print("TensorBoard logs:", LOG_DIR.resolve())
    print(
        "View logs with: "
        f"tensorboard --logdir {LOG_DIR}"
    )


if __name__ == "__main__":
    # Required for SubprocVecEnv multiprocessing on Windows.
    main()
