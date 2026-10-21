"""
Record the trained BipedalWalker-v3 baseline and separate-branch PPO models.

Expected model files (created by the training script):
  CHANGES_AS_INPUT/bipedalwalker_default_vs_separate/baseline/final_model.zip
  CHANGES_AS_INPUT/bipedalwalker_default_vs_separate/separate/final_model.zip

Creates two MP4 videos, each exactly 1 minute long at 50 FPS:
  CHANGES_AS_INPUT/bipedalwalker_recordings/baseline_1.0min.mp4
  CHANGES_AS_INPUT/bipedalwalker_recordings/separate_1.0min.mp4

Install dependencies if needed:
  pip install "stable-baselines3[extra]" "gymnasium[box2d]" imageio imageio-ffmpeg

Run from CHANGES_AS_INPUT:
  python bipedal_record.py
"""

from pathlib import Path
import time

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
from stable_baselines3 import PPO
from gymnasium import spaces


ENV_ID = "BipedalWalker-v3"
PROJECT_DIR = Path(__file__).resolve().parent
MODEL_ROOT = PROJECT_DIR / "bipedalwalker_default_vs_separate"
OUTPUT_DIR = PROJECT_DIR / "bipedalwalker_recordings"

VIDEO_SECONDS = 60
FPS = 50
TOTAL_FRAMES = VIDEO_SECONDS * FPS
SEED = 2026


class AddObservationDifference(gym.ObservationWrapper):
    """Emit [current observation, current observation - previous observation]."""

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


def make_recording_env(use_difference: bool):
    env = gym.make(ENV_ID, render_mode="rgb_array")
    if use_difference:
        env = AddObservationDifference(env)
    return env


def record_model(method: str):
    model_path = MODEL_ROOT / method / "final_model.zip"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Could not find trained model: {model_path}\n"
            "Run the training script first, and make sure it saved final_model.zip."
        )

    use_difference = method == "separate"
    output_path = OUTPUT_DIR / f"{method}_1.0min.mp4"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading {method} model from: {model_path}")
    model = PPO.load(str(model_path), device="auto")
    env = make_recording_env(use_difference)

    frame_count = 0
    episode_count = 0
    episode_return = 0.0
    episode_returns = []
    start_time = time.time()

    try:
        obs, _ = env.reset(seed=SEED)
        episode_count = 1

        # Stream frames directly to disk to avoid keeping 9,000 frames in RAM.
        with imageio.get_writer(
            str(output_path),
            fps=FPS,
            codec="libx264",
            quality=8,
            macro_block_size=None,
        ) as writer:
            while frame_count < TOTAL_FRAMES:
                frame = env.render()
                writer.append_data(frame)
                frame_count += 1

                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                episode_return += float(reward)

                if terminated or truncated:
                    episode_returns.append(episode_return)
                    obs, _ = env.reset()
                    episode_count += 1
                    episode_return = 0.0

                if frame_count % (FPS * 15) == 0:
                    elapsed = time.time() - start_time
                    print(
                        f"  {method}: {frame_count}/{TOTAL_FRAMES} frames "
                        f"({frame_count / FPS:.0f}/{VIDEO_SECONDS} video seconds); "
                        f"elapsed wall time {elapsed:.1f}s"
                    )

    finally:
        env.close()

    if episode_return != 0.0:
        episode_returns.append(episode_return)

    print(f"Saved: {output_path.resolve()}")
    print(f"Video duration: {TOTAL_FRAMES / FPS:.1f} seconds at {FPS} FPS")
    print(f"Completed episodes: {len(episode_returns)}")
    if episode_returns:
        print(f"Mean completed-episode return: {np.mean(episode_returns):.2f}")
        print(f"Episode returns: {[round(x, 2) for x in episode_returns]}")
    print(f"Recording wall time: {time.time() - start_time:.1f}s")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    record_model("baseline")
    record_model("separate")
    print("\nBoth recordings are finished.")


if __name__ == "__main__":
    main()
