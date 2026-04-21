'''
This file contains wrappers for the environment to essentially reformat and handle state info and other data the stable-retro game.
'''

import gymnasium as gym
import numpy as np
import stable_retro as retro
import cv2

class AirStrikerActionWrapper(gym.ActionWrapper):
    """This class will simplify the action space available to the agent into a discrete space that is mapped to the button inputs on a Genesis controller.
       Normally there are many different possible action inputs that make the state-action space much more complex."""
    
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
        """Takes in an already created env object as parameter, as well as an optional action space"""
        super().__init__(env)
        self.combos = combos if combos is not None else self.DEFAULT_COMBOS
        self.n_buttons = env.action_space.n  # number of total possible acitons
        self.action_space = gym.spaces.Discrete(len(self.combos))
 
    def action(self, act_index: int):
        """Convert a discrete action index to the action array accepted by the env (e.g. [1, 0, 0, 1, 0, ..., 0])"""
        multi = np.zeros(self.n_buttons, dtype=np.int8)
        for btn in self.combos[act_index]:
            multi[btn] = 1
        return multi

class ClipReward(gym.RewardWrapper):
    """This class will clip the rewards to within a range of [-1, 1] instead of having potentially very large values"""
    def __init__(self, env):
        super().__init__(env)

    def reward(self, reward):
        # np.sign returns -1 for negative, 0 for zero, 1 for positive
        return np.sign(reward)

class GrayscaleWrapper(gym.ObservationWrapper):
    """ A wrapper for the observations of frames obtained from the game that applies grayscale
        # to remove high dimensionality of using colors in the frame images"""
    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape[:2]  # (H, W)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(*obs_shape, 1), dtype=np.uint8
        )
    
    def observation(self, obs):
        """Return the given observation in grayscale with format: grayscale uint8 (H, W, 1)"""
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        return gray[:, :, np.newaxis]

class ResizeWrapper(gym.ObservationWrapper):
    """Wrapper to resize the game screen dimensions (normally 224 x 320) to a smaller square (e.g. 84 x 84)"""
    def __init__(self, env, size=84):
        super().__init__(env)
        self.size = size
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(size, size, 1), dtype=np.uint8
        )
 
    def observation(self, obs):
        """obs is (H, W, 1) — squeeze channel for cv2, then add it back"""
        resized = cv2.resize(
            obs[:, :, 0], (self.size, self.size), interpolation=cv2.INTER_AREA
        )
        return resized[:, :, np.newaxis]

class FrameStackWrapper(gym.Wrapper):
    """Class that handles stacking frames together to capture velocity of player ship and enemy objects over some bit of time"""
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
    
    def reset(self, **kwargs):
        """Resets the framestack"""
        obs, info = self.env.reset(**kwargs)
        frame = obs[:, :, 0]  # (H, W)
        for i in range(self.k):
            self._frames[i] = frame
        return self._frames.copy(), info
 
    def step(self, action):
        """advance the frame by one. Returns the new stack of frames after the step + reward, terminated, truc, info"""
        obs, reward, terminated, truncated, info = self.env.step(action)
        frame = obs[:, :, 0]

        # shift frames left and insert the new one at the end
        self._frames[:-1] = self._frames[1:]
        self._frames[-1] = frame
        return self._frames.copy(), reward, terminated, truncated, info

# create the environment for the agent to play the game in, with wrappers applied
def create_airstriker_env(game="Airstriker-Genesis-v0", state=retro.State.DEFAULT, frame_stack=4, resize=84, clip_rewards=True,render_mode: str | None = None,
):
    """
    Creates and wraps a stable-retro environment
 
    Returns an env with:
      - observation space = (frame_stack, resize, resize) uint8 <- FrameStackWrapper() + ResizeWrapper()
      - action space is Discrete(10) <- AirStrikerActionWrapper()
      - rewards are clipped to [-1, 1] <- ClipReward()
 
    Parameters
    ----------
    game : str
        ROM name supported by stable-retor (i.e. "Airstriker-Genesis-v0")
    state : str
        Initial save-state to load
    frame_stack : int
        Number of consecutive frames to stack (using FrameStackWrapper)
    resize : int
        Spatial size to resize frames to (square) (using ResizeWrapper)
    clip_rewards : bool
        Whether or not to clip rewards to {-1, 0, +1} (using ClipReward)
    render_mode : str or None
        Set to "human" to open a UI window, None otherwise
    """
    env = retro.make(game=game, state=state, render_mode=render_mode)
    env = AirStrikerActionWrapper(env)
    env = GrayscaleWrapper(env)
    env = ResizeWrapper(env, size=resize)
    if clip_rewards:
        env = ClipReward(env)
    env = FrameStackWrapper(env, num_frames=frame_stack)
    return env