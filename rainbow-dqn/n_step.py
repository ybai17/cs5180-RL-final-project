"""
This file contains the code for an queue that will store a set of N returns in a buffer, before they get eventually popped out
and used for updates.
"""

from collections import deque
 
 
class NStepBuffer:
    """
    Parameters
    ----------
    n      : int    Number of steps to accumulate (e.g. 3).
    gamma  : float  Discount factor.
    buffer : ReplayBuffer or PrioritizedReplayBuffer to push into.
    """
 
    def __init__(self, n: int, gamma: float, buffer):
        self.n = n
        self.gamma = gamma
        self.buffer = buffer
        self._deque = deque(maxlen=n)
 
    def push(self, obs, action, reward, next_obs, done):
        """
        Add a single-step transition.  When the deque is full (n items)
        or an episode ends, flush the accumulated n-step transition(s)
        to the main buffer.
        """
        self._deque.append((obs, action, reward, next_obs, done))
 
        # If episode ended, flush all pending transitions
        if done:
            self._flush_all()
        # If deque is full, flush the oldest transition as an n-step
        elif len(self._deque) == self.n:
            self._flush_one()
 
    def _flush_one(self):
        """
        Compute the n-step return from the full deque and push the
        oldest transition with the accumulated reward and the newest
        next_obs.
        """
        obs, action, _, _, _ = self._deque[0]
 
        # Accumulate discounted reward across the deque
        n_step_reward = 0.0
        for i, (_, _, r, _, _) in enumerate(self._deque):
            n_step_reward += (self.gamma ** i) * r
 
        # The "next state" is the next_obs from the most recent transition
        _, _, _, next_obs, done = self._deque[-1]
 
        self.buffer.push(obs, action, n_step_reward, next_obs, done)
 
    def _flush_all(self):
        """
        Flush all remaining transitions at episode end.
        For the k-th entry (0-indexed) in the deque, we compute a
        (len - k)-step return using the remaining transitions.
        """
        length = len(self._deque)
        for start in range(length):
            obs, action, _, _, _ = self._deque[start]
 
            n_step_reward = 0.0
            for i in range(start, length):
                n_step_reward += (self.gamma ** (i - start)) * self._deque[i][2]
 
            # next_obs and done from the last transition in the deque
            _, _, _, next_obs, done = self._deque[-1]
 
            self.buffer.push(obs, action, n_step_reward, next_obs, done)
 
        self._deque.clear()