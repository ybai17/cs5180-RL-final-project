"""
Replay Buffer for DQN.

Stores transitions (s, a, r, s', done) in pre-allocated numpy arrays
for memory efficiency. Observations are stored as uint8 and converted
to float32 tensors on sampling.

Also contains a PrioritizedReplayBuffer class
"""

import numpy as np
import torch


class ReplayBuffer:
    """
    Circular fixed-size replay buffer with uniform sampling

    Parameters:
    ----------
    capacity : int
        Maximum number of transitions stored.
    obs_shape : tuple
        Shape of a single observation, e.g. (4, 84, 84).
    """

    def __init__(self, capacity: int, obs_shape: tuple):
        self.capacity = capacity
        self.index = 0          # next write position
        self.size = 0         # current number of stored transitions

        # Pre-allocate storage — observations as uint8 to save RAM
        self.obs      = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.next_obs = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.actions  = np.zeros(capacity, dtype=np.int64)
        self.rewards  = np.zeros(capacity, dtype=np.float32)
        self.dones    = np.zeros(capacity, dtype=np.float32)  # 1.0 if terminal

    def push(self, obs, action, reward, next_obs, done):
        """
        Store a single transition

        Parameters:
        ----------
        obs      : np.ndarray  (obs_shape,) uint8
        action   : int
        reward   : float
        next_obs : np.ndarray  (obs_shape,) uint8
        done     : bool
        """
        i = self.index
        self.obs[i]      = obs
        self.actions[i]  = action
        self.rewards[i]  = reward
        self.next_obs[i] = next_obs
        self.dones[i]    = float(done)

        self.index = (self.index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device):
        """
        Sample a random batch and return tensors on `device`

        Returns:
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

from sum_tree import SumTree

class PrioritizedReplayBuffer:
    """
    Replay buffer with proportional prioritization based on work by Schaul et al., 2016

    Sampling probability for transition i:
        P(i) = p_i^alpha / sum_k p_k^alpha

    where p_i = |TD_error_i| + epsilon

    Importance-sampling weights correct the bias introduced by
    non-uniform sampling:
        w_i = (N * P(i))^(-beta)   (normalized by max weight in batch)

    Parameters:
    ----------
    capacity  : int    Max stored transitions.
    obs_shape : tuple  Observation shape, e.g. (4, 84, 84).
    alpha     : float  Prioritization exponent (0 = uniform, 1 = full).
    """

    def __init__(self, capacity: int, obs_shape: tuple, alpha: float = 0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.index = 0
        self.size = 0
        self.max_priority = 1.0   # used for new transitions

        # data storage is same as ReplayBuffer
        self.obs      = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.next_obs = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.actions  = np.zeros(capacity, dtype=np.int64)
        self.rewards  = np.zeros(capacity, dtype=np.float32)
        self.dones    = np.zeros(capacity, dtype=np.float32)

        self.tree = SumTree(capacity)

    def push(self, obs, action, reward, next_obs, done):
        """
        Store a transition with max priority so it gets sampled at least once
        """
        i = self.index
        self.obs[i]      = obs
        self.actions[i]  = action
        self.rewards[i]  = reward
        self.next_obs[i] = next_obs
        self.dones[i]    = float(done)

        # new transitions get max priority
        self.tree.update(i, self.max_priority ** self.alpha)

        self.index = (self.index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device, beta: float = 0.4):
        """
        Sample a prioritized batch from the buffer

        Parameters:
        ----------
        batch_size : int
        device     : torch.device
        beta       : float   IS exponent; annealed from beta_start to 1.0.

        Returns:
        -------
        dict with keys:
            obs, actions, rewards, next_obs, dones  — same as ReplayBuffer
            weights   (batch,) float32  — importance-sampling weights
            indices   (batch,) int      — buffer indices (needed for priority update)
        """
        indices = np.empty(batch_size, dtype=np.int64)
        priorities = np.empty(batch_size, dtype=np.float64)

        # divide the total priority range into equal segments, then
        # sample one transition from each segment (stratified sampling)
        total = self.tree.total()
        segment = total / batch_size

        for i in range(batch_size):
            lo = segment * i
            hi = segment * (i + 1)
            value = np.random.uniform(lo, hi)
            index = self.tree.sample(value)
            indices[i] = index
            priorities[i] = self.tree.get(index)

        # the weights for importance sampling
        # P(i) = priority_i / total
        probs = priorities / total
        # w_i = (N * P(i))^(-beta), then normalize by max
        weights = (self.size * probs) ** (-beta)
        weights /= weights.max()

        # the tensors
        obs      = torch.as_tensor(self.obs[indices],      dtype=torch.float32, device=device).div_(255.0)
        next_obs = torch.as_tensor(self.next_obs[indices],  dtype=torch.float32, device=device).div_(255.0)
        actions  = torch.as_tensor(self.actions[indices],   dtype=torch.int64,   device=device)
        rewards  = torch.as_tensor(self.rewards[indices],   dtype=torch.float32, device=device)
        dones    = torch.as_tensor(self.dones[indices],     dtype=torch.float32, device=device)
        weights  = torch.as_tensor(weights,                 dtype=torch.float32, device=device)

        return {
            "obs": obs,
            "actions": actions,
            "rewards": rewards,
            "next_obs": next_obs,
            "dones": dones,
            "weights": weights,
            "indices": indices,
        }

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray, epsilon: float = 1e-6):
        """
        Update priorities for sampled transitions

        Parameters:
        ----------
        indices   : (batch,) int     — indices returned by sample().
        td_errors : (batch,) float   — absolute TD errors from the loss computation.
        epsilon   : float            — small constant to prevent zero priority.
        """
        for index, td in zip(indices, td_errors):
            priority = (abs(td) + epsilon) ** self.alpha
            self.tree.update(int(index), priority)
            self.max_priority = max(self.max_priority, abs(td) + epsilon)

    def __len__(self):
        return self.size