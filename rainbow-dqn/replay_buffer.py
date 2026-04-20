"""
Replay Buffer for DQN.

Stores transitions (s, a, r, s', done) in pre-allocated numpy arrays
for memory efficiency.  Observations are stored as uint8 and converted
to float32 tensors on sampling.

Phase 4 will add a PrioritizedReplayBuffer subclass alongside this.
"""

import numpy as np
import torch


class ReplayBuffer:
    """
    Circular fixed-size replay buffer with uniform sampling.

    Parameters
    ----------
    capacity : int
        Maximum number of transitions stored.
    obs_shape : tuple
        Shape of a single observation, e.g. (4, 84, 84).
    """

    def __init__(self, capacity: int, obs_shape: tuple):
        self.capacity = capacity
        self.idx = 0          # next write position
        self.size = 0         # current number of stored transitions

        # Pre-allocate storage — observations as uint8 to save RAM
        self.obs      = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.next_obs = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.actions  = np.zeros(capacity, dtype=np.int64)
        self.rewards  = np.zeros(capacity, dtype=np.float32)
        self.dones    = np.zeros(capacity, dtype=np.float32)  # 1.0 if terminal

    def push(self, obs, action, reward, next_obs, done):
        """
        Store a single transition.

        Parameters
        ----------
        obs      : np.ndarray  (obs_shape,) uint8
        action   : int
        reward   : float
        next_obs : np.ndarray  (obs_shape,) uint8
        done     : bool
        """
        i = self.idx
        self.obs[i]      = obs
        self.actions[i]  = action
        self.rewards[i]  = reward
        self.next_obs[i] = next_obs
        self.dones[i]    = float(done)

        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device):
        """
        Sample a random batch and return tensors on `device`.

        Returns
        -------
        dict with keys:
            obs        (batch, C, H, W)  float32 in [0, 1]
            actions    (batch,)           int64
            rewards    (batch,)           float32
            next_obs   (batch, C, H, W)  float32 in [0, 1]
            dones      (batch,)           float32
        """
        indices = np.random.randint(0, self.size, size=batch_size)

        obs      = torch.as_tensor(self.obs[indices],      dtype=torch.float32, device=device).div_(255.0)
        next_obs = torch.as_tensor(self.next_obs[indices],  dtype=torch.float32, device=device).div_(255.0)
        actions  = torch.as_tensor(self.actions[indices],   dtype=torch.int64,   device=device)
        rewards  = torch.as_tensor(self.rewards[indices],   dtype=torch.float32, device=device)
        dones    = torch.as_tensor(self.dones[indices],     dtype=torch.float32, device=device)

        return {
            "obs": obs,
            "actions": actions,
            "rewards": rewards,
            "next_obs": next_obs,
            "dones": dones,
        }

    def __len__(self):
        return self.size