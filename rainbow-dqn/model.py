"""
DQN architecture defined in this file

Non-distributional mode:
    Input:  (batch, frame_stack, 84, 84)  float32 in [0, 1]
    Output: (batch, n_actions)             Q-values

Distributional mode:
    Input:  (batch, frame_stack, 84, 84)  float32 in [0, 1]
    Output: (batch, n_actions, n_atoms)    probability distributions over returns

"""

import torch
import torch.nn as nn

class QNetwork(nn.Module):
    """
    This class defines the DQN that will be used for Q-value estimations. Uses CNNs

    Conv2d(frame_stack, 32, 8, stride=4) -> ReLU
    Conv2d(32, 64, 4, stride=2)          -> ReLU
    Conv2d(64, 64, 3, stride=1)          -> ReLU
    Flatten -> Linear(3136, 512) -> ReLU -> Linear(512, n_actions)
    """

    def __init__(self, in_channels: int, num_actions: int, dueling: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.n_actions = num_actions
        self.dueling = dueling

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

        # check if dueling. If so, split the heads
        if dueling:
            self.value_stream = nn.Sequential(
                nn.Linear(self.feature_size, 512),
                nn.ReLU(),
                nn.Linear(512, 1),
            )
            
            self.advantage_stream = nn.Sequential(
                nn.Linear(self.feature_size, 512),
                nn.ReLU(),
                nn.Linear(512, num_actions)
            )

        else:
            # fully connected head
            self.head = nn.Sequential(
                nn.Linear(self.feature_size, 512),
                nn.ReLU(),
                nn.Linear(512, num_actions),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Fowards the input through the network 
        
        Parameters:
        ----------
        x : (batch, in_channels, 84, 84) float32, expected in [0, 1].

        Returns:
        -------
        q_values : (batch, n_actions)
        """
        x = self.features(x)
        x = x.view(x.size(0), -1)  # flatten

        if self.dueling:
            value = self.value_stream(x)
            advantage = self.advantage_stream(x)

            q = value + advantage - advantage.mean(dim=1, keepdim=True)
            return q

        return self.head(x)