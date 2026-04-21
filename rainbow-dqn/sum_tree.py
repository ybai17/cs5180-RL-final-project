"""
Sum Tree for Prioritized Experience Replay

A binary tree where each leaf holds a priority value and each internal
node holds the sum of its children.  This gives us:
  - O(log n) priority update
  - O(log n) proportional sampling
  - O(1)     total priority lookup

Tree layout (array-based, 1-indexed internally):
         [root = sum of all]
           /            \
      [left sum]     [right sum]
       /    \          /    \
     p1     p2       p3     p4    <- leaves (priorities)

Array indices: internal nodes at 1..capacity-1, leaves at capacity..2*capacity-1
We use 0-based arrays sized 2*capacity, with index 0 unused
"""

import numpy as np

class SumTree:
    """
    Fixed-capacity sum tree backed by a flat numpy array

    Parameters:
    ----------
    capacity : int
        Number of leaf nodes (must equal the replay buffer capacity)
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        # tree array: indices [0..2*capacity-1].  Index 0 is unused
        # internal nodes: 1 .. capacity-1
        # leaves:         capacity .. 2*capacity-1
        self.tree = np.zeros(2 * capacity, dtype=np.float64)

    def total(self) -> float:
        """Sum of all priorities (root node)."""
        return self.tree[1]

    def update(self, leaf_index: int, priority: float):
        """
        Set the priority of a leaf and propagate the change upwards

        Parameters:
        ----------
        leaf_index : int
            Data index in [0, capacity).  Internally mapped to tree index
        priority : float
            New priority value (must be > 0)
        """
        tree_index = leaf_index + self.capacity
        delta = priority - self.tree[tree_index]
        self.tree[tree_index] = priority
        # Walk up to root
        tree_index //= 2
        while tree_index >= 1:
            self.tree[tree_index] += delta
            tree_index //= 2

    def sample(self, value: float) -> int:
        """
        Sample a leaf proportional to its priority

        Parameters:
        ----------
        value : float
            A uniform random number in [0, total)

        Returns:
        -------
        leaf_idx : int
            Data index in [0, capacity)
        """
        index = 1  # start at root
        while index < self.capacity:
            left = 2 * index
            right = left + 1
            if value <= self.tree[left]:
                index = left
            else:
                value -= self.tree[left]
                index = right
        return index - self.capacity  # convert tree index to data index

    def get(self, leaf_index: int) -> float:
        """Get the priority of a leaf by data index."""
        return self.tree[leaf_index + self.capacity]