"""
File containing function for performing a basic test of the environment using a random agent.
"""

import argparse
import time
import numpy as np
from env_wrappers import create_airstriker_env
 
def run_random_episodes(n_episodes: int = 3, render: bool = False):
    """
    function for running N episodes
    Use "python test.py --render" when running if you want to watch
    """
    render_mode = "human" if render else None
    env = create_airstriker_env(render_mode=render_mode)
 
    print("=" * 60)
    print("ENVIRONMENT INFO")
    print("=" * 60)
    print(f"Game: Airstriker-Genesis-v0")
    print(f"Observation space: {env.observation_space}")
    print(f"Obs dtype: {env.observation_space.dtype}")
    print(f"Action space: {env.action_space}  (n={env.action_space.n})")
    print("=" * 60)
 
    all_rewards = []
    
    # main loop
    for ep in range(1, n_episodes + 1):
        obs, info = env.reset()
        assert obs.shape == env.observation_space.shape, (
            f"Shape mismatch: got {obs.shape}, expected {env.observation_space.shape}"
        )
        assert obs.dtype == np.uint8, f"Dtype mismatch: got {obs.dtype}"
 
        ep_reward = 0.0
        steps = 0
        done = False
 
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            steps += 1
            done = terminated or truncated
 
            if render:
                time.sleep(0.01)  # slow down we you can watch
 
        all_rewards.append(ep_reward)
        print(
            f"  Episode {ep:>3d}  |  steps: {steps:>5d}  |  "
            f"reward: {ep_reward:>8.1f}  |  final obs range: [{obs.min()}, {obs.max()}]"
        )
 
    env.close()
 
    print("-" * 60)
    print(
        f"  Mean reward over {n_episodes} episodes: "
        f"{np.mean(all_rewards):.1f} +/- {np.std(all_rewards):.1f}"
    )
    print("=" * 60)
    print("Initial check test PASSED — environment is working.")
 
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the retro environment wrappers")
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes")
    parser.add_argument("--render", action="store_true", help="Render game window")
    args = parser.parse_args()
 
    run_random_episodes(n_episodes=args.episodes, render=args.render)