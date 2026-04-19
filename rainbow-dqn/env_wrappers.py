'''
This file contains wrappers for the environment to essentially reformat and handle state info and other data the stable-retro game.
'''

import gymnasium as gym
import numpy as np
import stable_retro as retro
import cv2

# This class will simplify the action space available to the agent into a discrete space that is mapped to the button inputs on a Genesis controller.
# Normally there are many different possible action inputs that make the state-action space much more complex
class AirStrikerActionWrapper(gym.ActionWrapper):
    # each sub-list gives the button indices to press at the same time: (index 0 = B/shoot, 4 = up, etc.)
    DEFAULT_COMBOS = [
        [],          # 0  NOOP
        [0],         # 1  B  (shoot)
        [4],         # 2  UP
        [5],         # 3  DOWN
        [6],         # 4  LEFT
        [7],         # 5  RIGHT
        [4, 0],      # 6  UP    + SHOOT
        [5, 0],      # 7  DOWN  + SHOOT
        [6, 0],      # 8  LEFT  + SHOOT
        [7, 0],      # 9  RIGHT + SHOOT
    ]
 
    def __init__(self, env, combos=None):
        super().__init__(env)
        self.combos = combos if combos is not None else self.DEFAULT_COMBOS
        self.n_buttons = env.action_space.n  # MultiBinary size
        self.action_space = gym.spaces.Discrete(len(self.combos))
 
    def action(self, act_idx: int):
        """Convert a discrete action index to a MultiBinary array."""
        multi = np.zeros(self.n_buttons, dtype=np.int8)
        for btn in self.combos[act_idx]:
            multi[btn] = 1
        return multi

# This class will clip the rewards to within a range of [-1, 1] instead of having potentially very large values
class ClipReward(gym.RewardWrapper):
    
    def __init__(self, env):
        super().__init__(env)

    def reward(self, reward):
        # np.sign returns -1 for negative, 0 for zero, 1 for positive
        return np.sign(reward)

# A wrapper for the observations of frames obtained from the game that applies grayscale
# to remove high dimensionality of using colors in the frame images
class GrayscaleWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape[:2]  # (H, W)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(*obs_shape, 1), dtype=np.uint8
        )
    
    # return the observation in grayscale with format: grayscale uint8 (H, W, 1)
    def observation(self, obs):
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        return gray[:, :, np.newaxis]

# wrapper to resize the game screen dimensions (normally 224 x 320) to a smaller square (e.g. 84 x 84)
class ResizeWrapper(gym.ObservationWrapper):
    def __init__(self, env, size=84):
        super().__init__(env)
        self.size = size
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(size, size, 1), dtype=np.uint8
        )
 
    def observation(self, obs):
        # obs is (H, W, 1) — squeeze channel for cv2, then add it back
        resized = cv2.resize(
            obs[:, :, 0], (self.size, self.size), interpolation=cv2.INTER_AREA
        )
        return resized[:, :, np.newaxis]

# class that handles stacking frames together to capture velocity of player ship and enemy objects over some bit of time
class FrameStackWrapper(gym.Wrapper):
    def __init__(self, env, num_frames=4):
        super().__init__(env)
        self.k = num_frames
        self._frames = np.zeros(
            (num_frames, *env.observation_space.shape[:2]), dtype=np.uint8
        )
        h, w = env.observation_space.shape[:2]
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(num_frames, h, w), dtype=np.uint8
        )
    
    # resets 
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        frame = obs[:, :, 0]  # (H, W)
        for i in range(self.k):
            self._frames[i] = frame
        return self._frames.copy(), info
 
    # advance the frame by one
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        frame = obs[:, :, 0]
        # Shift frames left and insert the new one at the end
        self._frames[:-1] = self._frames[1:]
        self._frames[-1] = frame
        return self._frames.copy(), reward, terminated, truncated, info

# create the environment for the agent to play the game in, with wrappers applied
def create_airstriker_env(game="Airstriker-Genesis", state=retro.State.DEFAULT, frame_stack=4, resize=84, clip_rewards=True,render_mode: str | None = None,
):
    """
    Create and wrap a stable-retro environment.
 
    Returns an env whose:
      - observation space is (frame_stack, resize, resize) uint8
      - action space is Discrete(10)  (see DiscretizeActionWrapper)
      - rewards are clipped to [-1, 1]
 
    Parameters
    ----------
    game : str
        ROM name recognized by stable-retro.
    state : str
        Initial save-state to load.
    frame_stack : int
        Number of consecutive frames to stack.
    resize : int
        Spatial size to resize frames to (square).
    clip_rewards : bool
        Whether to clip rewards to {-1, 0, +1}.
    render_mode : str or None
        Pass "human" to open a window, None for headless.
    """
    env = retro.make(game=game, state=state, render_mode=render_mode)
    env = AirStrikerActionWrapper(env)
    env = GrayscaleWrapper(env)
    env = ResizeWrapper(env, size=resize)
    if clip_rewards:
        env = ClipReward(env)
    env = FrameStackWrapper(env, k=frame_stack)
    return env