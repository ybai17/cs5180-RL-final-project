"""
DQN Training Loop.

Handles:
  - Environment interaction and buffer filling
  - Training at the configured frequency
  - Target network syncing
  - Epsilon schedule
  - TensorBoard logging
  - Periodic evaluation episodes
  - Checkpoint saving

Usage:
    python train.py
    python train.py --device cuda
    tensorboard --logdir runs/
"""

import argparse
import os
import time
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from config import CONFIG
from env_wrappers import create_airstriker_env
from replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from n_step import NStepBuffer
from agent import DQNAgent


def evaluate(agent: DQNAgent, num_episodes: int, step: int) -> float:
    """
    Run evaluation episodes with a low fixed epsilon

    Returns:
    -------
    mean_reward : float
    """
    env = create_airstriker_env(game=CONFIG["game"], clip_rewards=False)  # raw rewards for eval
    rewards = []
    for _ in range(num_episodes):
        obs, _ = env.reset()
        ep_reward = 0.0
        done = False
        while not done:
            action = agent.select_action(obs, step, eval_mode=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            done = terminated or truncated
        rewards.append(ep_reward)
    env.close()
    return float(np.mean(rewards))


def train(device_str: str = "cpu"):

    # setup stuff
    device = torch.device(device_str)
    print(f"Device: {device}")

    env = create_airstriker_env(game=CONFIG["game"])
    obs_shape = env.observation_space.shape    # (4, 84, 84)
    n_actions = env.action_space.n

    agent = DQNAgent(
        n_actions=n_actions,
        in_channels=obs_shape[0],
        config=CONFIG,
        device=device,
    )

    buffer = ReplayBuffer(capacity=CONFIG["buffer_size"], obs_shape=obs_shape)

    use_per = CONFIG.get("per", False)
    if use_per:
        buffer = PrioritizedReplayBuffer(
            capacity=CONFIG["buffer_size"],
            obs_shape=obs_shape,
            alpha=CONFIG["per_alpha"],
        )
        print("Using Prioritized Experience Replay")

    # the n-step return buffer
    n_step = CONFIG.get("n_step", 1)
    n_step_buf = None
    if n_step > 1:
        n_step_buf = NStepBuffer(n=n_step, gamma=CONFIG["gamma"], buffer=buffer)
        print(f"Using {n_step}-step returns")

    # fo9r logging
    run_name = f"dqn_{CONFIG['game']}_{int(time.time())}"
    log_dir = os.path.join("runs", run_name)
    writer = SummaryWriter(log_dir)
    ckpt_dir = os.path.join(log_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    print(f"Logging to {log_dir}")

    # the seed
    np.random.seed(CONFIG["seed"])
    torch.manual_seed(CONFIG["seed"])

    # keep track of the data during training
    obs, _ = env.reset()
    ep_reward = 0.0
    ep_steps = 0
    ep_count = 0
    losses = []

    total_steps = CONFIG["total_steps"]
    start_time = time.time()

    for step in range(1, total_steps + 1):
        # select and execute the action
        action = agent.select_action(obs, step)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # store the transition
        if n_step_buf is not None:
            n_step_buf.push(obs, action, reward, next_obs, done)
        else:
            buffer.push(obs, action, reward, next_obs, done)

        obs = next_obs
        ep_reward += reward
        ep_steps += 1

        # handle when the episode ends
        if done:
            ep_count += 1
            writer.add_scalar("train/episode_reward", ep_reward, step)
            writer.add_scalar("train/episode_length", ep_steps, step)
            writer.add_scalar("train/epsilon", agent._epsilon(step), step)
            writer.add_scalar("train/episodes", ep_count, step)

            if ep_count % 10 == 0:
                elapsed = time.time() - start_time
                fps = step / elapsed
                print(
                    f"Step {step}/{total_steps} | "
                    f"Ep {ep_count} | "
                    f"Reward: {ep_reward} | "
                    f"Eps: {agent._epsilon(step)} | "
                    f"Buf: {len(buffer)} | "
                    f"FPS: {fps}"
                )

            obs, _ = env.reset()
            ep_reward = 0.0
            ep_steps = 0

        # train
        if step >= CONFIG["train_start"] and step % CONFIG["train_freq"] == 0:
            if use_per:
                # anneal the beta linearly from beta_start to beta_end
                beta_frac = min(1.0, step / CONFIG["per_beta_anneal_steps"])
                beta = CONFIG["per_beta_start"] + beta_frac * (CONFIG["per_beta_end"] - CONFIG["per_beta_start"])
                batch = buffer.sample(CONFIG["batch_size"], device, beta=beta)
            else:
                batch = buffer.sample(CONFIG["batch_size"], device)

            loss, td_errors = agent.update(batch)
            losses.append(loss)

            # update priorities in the sum tree
            if use_per:
                buffer.update_priorities(batch["indices"], td_errors, epsilon=CONFIG["per_epsilon"])

        # sync to the target netwrork
        if step % CONFIG["target_update_freq"] == 0:
            agent.sync_target()

        # log the training metrics
        if step % CONFIG["log_freq"] == 0 and losses:
            mean_loss = np.mean(losses)
            writer.add_scalar("train/loss", mean_loss, step)
            losses.clear()

        # evaluate
        # stable-retro only allows one emulator per process, so we must
        # close the training env before evaluate() creates its own.
        if step % CONFIG["eval_freq"] == 0:
            env.close()
            mean_eval = evaluate(agent, CONFIG["eval_episodes"], step)
            writer.add_scalar("eval/mean_reward", mean_eval, step)
            print(f"  [EVAL] Step {step}  |  Mean reward: {mean_eval:.1f}")
            # Reopen training env and start a fresh episode
            env = create_airstriker_env(game=CONFIG["game"])
            obs, _ = env.reset()
            ep_reward = 0.0
            ep_steps = 0

        # save the checkpoint
        if step % CONFIG["save_freq"] == 0:
            path = os.path.join(ckpt_dir, f"ckpt_{step}.pt")
            agent.save(path)
            print(f"  [SAVE] {path}")

    # cleanup
    agent.save(os.path.join(ckpt_dir, "ckpt_final.pt"))
    env.close()
    writer.close()
    print("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DQN on Airstriker-Genesis")
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to train on (cpu / cuda / mps)"
    )
    args = parser.parse_args()
    train(device_str=args.device)

# to train:
# python train.py                # CPU
# python train.py --device cuda  # GPU
# tensorboard --logdir runs/     # monitor in browser using local tensorboard server

# watch the final trained agent
# python play.py --checkpoint runs/<run_name>/checkpoints/ckpt_final.pt

# compare an earlier checkpoint vs a late one (or the last one)
# python play.py --checkpoint runs/<run_name>/checkpoints/ckpt_100000.pt
# python play.py --checkpoint runs/<run_name>/checkpoints/ckpt_final.pt

# You can also slow it down to make it easier to watch 
# python play.py --checkpoint runs/<run_name>/checkpoints/ckpt_final.pt --delay 0.03

# watch more episodes
# python play.py --checkpoint runs/<run_name>/checkpoints/ckpt_final.pt --episodes 10