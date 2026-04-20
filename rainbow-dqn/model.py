"""
DQN architecture defined in this file

Input:  (batch, frame_stack, 84, 84)  float32 in [0, 1]
Output: (batch, n_actions)             Q-values per action

"""

import torch
import torch.nn as nn

# This class defines the DQN that will be used for Q-value estimations. Uses CNNs
class QNetwork(nn.Module):
    """
    Conv2d(frame_stack, 32, 8, stride=4) -> ReLU
    Conv2d(32, 64, 4, stride=2)          -> ReLU
    Conv2d(64, 64, 3, stride=1)          -> ReLU
    Flatten -> Linear(3136, 512) -> ReLU -> Linear(512, n_actions)
    """

    def __init__(self, in_channels: int, n_actions: int):
        super().__init__()
        self.in_channels = in_channels
        self.n_actions = n_actions

        # CNN feature extraction
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        # get flattened size after convolutions on an 84x84 input
        # 84 -> conv1(8,4) -> 20 -> conv2(4,2) -> 9 -> conv3(3,1) -> 7
        # 64 * 7 * 7 = 3136
        self.feature_size = 64 * 7 * 7

        # fully connected head
        self.head = nn.Sequential(
            nn.Linear(self.feature_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (batch, in_channels, 84, 84) float32, expected in [0, 1].

        Returns
        -------
        q_values : (batch, n_actions)
        """
        x = self.features(x)
        x = x.view(x.size(0), -1)  # flatten
        return self.head(x)