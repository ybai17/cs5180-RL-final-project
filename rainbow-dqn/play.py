"""
Watch a trained DQN agent play Airstriker-Genesis.

Usage:
    python play.py --checkpoint runs/<run_name>/checkpoints/ckpt_final.pt
    python play.py --checkpoint runs/<run_name>/checkpoints/ckpt_100000.pt --episodes 5
    python play.py --checkpoint runs/<run_name>/checkpoints/ckpt_final.pt --delay 0.02
"""

import argparse
import time
import numpy as np
import torch

from config import CONFIG
from env_wrappers import create_airstriker_env
from agent import DQNAgent


def play(checkpoint: str, n_episodes: int = 3, delay: float = 0.01):
    device = torch.device("cpu")  # no need for GPU during playback

    # --- Create env with rendering enabled and raw rewards ---
    env = create_airstriker_env(
        game=CONFIG["game"],
        clip_rewards=False,
        render_mode="human",
    )

    obs_shape = env.observation_space.shape
    n_actions = env.action_space.n

    # --- Load agent from checkpoint ---
    agent = DQNAgent(
        n_actions=n_actions,
        in_channels=obs_shape[0],
        config=CONFIG,
        device=device,
    )
    agent.load(checkpoint)
    agent.online_net.eval()
    print(f"Loaded checkpoint: {checkpoint}")

    # --- Play ---
    for ep in range(1, n_episodes + 1):
        obs, _ = env.reset()
        ep_reward = 0.0
        steps = 0
        done = False

        while not done:
            action = agent.select_action(obs, step=0, eval_mode=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            steps += 1
            done = terminated or truncated
            time.sleep(delay)

        print(f"Episode {ep}  |  Steps: {steps:>5d}  |  Reward: {ep_reward:.1f}")

    env.close()
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Watch a trained agent play")
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to a .pt checkpoint file",
    )
    parser.add_argument(
        "--episodes", type=int, default=3,
        help="Number of episodes to play",
    )
    parser.add_argument(
        "--delay", type=float, default=0.01,
        help="Seconds to pause between frames (adjust playback speed)",
    )
    args = parser.parse_args()
    play(checkpoint=args.checkpoint, n_episodes=args.episodes, delay=args.delay)