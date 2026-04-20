"""
DQN Agent.

Owns the online network, target network, optimizer, and provides:
  - epsilon-greedy action selection
  - TD loss computation
  - target network synchronization

Later phases modify:
  - Phase 3: Double DQN target computation  (self.compute_loss)
  - Phase 5: Dueling architecture            (model.py)
  - Phase 7: Distributional loss             (self.compute_loss)
  - Phase 8: Noisy nets / remove epsilon     (self.select_action)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from model import QNetwork


class DQNAgent:
    def __init__(self, n_actions: int, in_channels: int, config: dict, device: torch.device):
        self.n_actions = n_actions
        self.config = config
        self.device = device
        self.gamma = config["gamma"]

        # --- Networks ---
        self.online_net = QNetwork(in_channels, n_actions).to(device)
        self.target_net = QNetwork(in_channels, n_actions).to(device)
        self.sync_target()                       # copy weights
        self.target_net.eval()                    # target never trains

        # --- Optimizer ---
        self.optimizer = optim.Adam(
            self.online_net.parameters(),
            lr=config["learning_rate"],
        )

        # --- Epsilon schedule (linear decay) ---
        self.eps_start = config["eps_start"]
        self.eps_end = config["eps_end"]
        self.eps_decay_steps = config["eps_decay_steps"]

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------
    def select_action(self, obs: np.ndarray, step: int, eval_mode: bool = False) -> int:
        """
        Epsilon-greedy action selection.

        Parameters
        ----------
        obs  : np.ndarray (C, H, W) uint8 — a single observation.
        step : int — current training step (for epsilon schedule).
        eval_mode : bool — if True, use a fixed low epsilon (0.01).

        Returns
        -------
        action : int
        """
        epsilon = self._epsilon(step) if not eval_mode else 0.01

        if np.random.random() < epsilon:
            return np.random.randint(self.n_actions)

        with torch.no_grad():
            # (1, C, H, W) float32 in [0, 1]
            obs_t = (
                torch.as_tensor(obs, dtype=torch.float32, device=self.device)
                .unsqueeze(0)
                .div_(255.0)
            )
            q_values = self.online_net(obs_t)      # (1, n_actions)
            return q_values.argmax(dim=1).item()

    def _epsilon(self, step: int) -> float:
        """Linear epsilon decay from eps_start to eps_end."""
        fraction = min(1.0, step / self.eps_decay_steps)
        return self.eps_start + fraction * (self.eps_end - self.eps_start)

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------
    def compute_loss(self, batch: dict) -> torch.Tensor:
        """
        Compute the standard DQN (or Double DQN) TD loss.

        Parameters
        ----------
        batch : dict from ReplayBuffer.sample()

        Returns
        -------
        loss : scalar tensor (mean Huber loss over the batch)
        """
        obs      = batch["obs"]        # (B, C, H, W)
        actions  = batch["actions"]    # (B,)
        rewards  = batch["rewards"]    # (B,)
        next_obs = batch["next_obs"]   # (B, C, H, W)
        dones    = batch["dones"]      # (B,)

        # Q(s, a) for the actions that were taken
        q_values = self.online_net(obs)                          # (B, A)
        q_sa = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)  # (B,)

        # --- Target computation ---
        with torch.no_grad():
            if self.config.get("double_dqn", False):
                # Double DQN: online net selects, target net evaluates
                next_actions = self.online_net(next_obs).argmax(dim=1, keepdim=True)
                next_q = self.target_net(next_obs).gather(1, next_actions).squeeze(1)
            else:
                # Vanilla DQN: target net does both
                next_q = self.target_net(next_obs).max(dim=1)[0]

            target = rewards + self.gamma * next_q * (1.0 - dones)

        # Huber loss (smooth L1) — less sensitive to outliers than MSE
        loss = nn.functional.smooth_l1_loss(q_sa, target)
        return loss

    def update(self, batch: dict) -> float:
        """
        Run one gradient step.

        Returns
        -------
        loss_value : float (for logging)
        """
        loss = self.compute_loss(batch)
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping — stabilizes early training
        nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=10.0)
        self.optimizer.step()
        return loss.item()

    # ------------------------------------------------------------------
    # Target network
    # ------------------------------------------------------------------
    def sync_target(self):
        """Hard-copy online network weights to target network."""
        self.target_net.load_state_dict(self.online_net.state_dict())

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------
    def save(self, path: str):
        torch.save({
            "online_net": self.online_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.online_net.load_state_dict(ckpt["online_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])