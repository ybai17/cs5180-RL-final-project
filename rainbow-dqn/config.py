'''
This file contains all the hyperparameters and other parameters used to run the program, contained in dictionaries
'''

CONFIG = {

    # contains the parameters for the overall environment
    "game": "Airstriker-Genesis-v0", # comes pre-installed with stable-retro
    "frame_stack": 4,                # number of frames to stack together to capture velocity information during gameplay
    "frame_size": 84,                # dimension to which the Genesis game screen (normally 224 x 320) will be reduced, as the length of a square (e.g. 84 x 84)
    "clip_rewards": True,            # set to True to use reward clipping within a range of [-1, 1] instead of larger ones
    "seed": 42,

    # contains hyperparameters and parameters used for training
    "total_steps": 1_000_000,         # total number of time steps to train on
    "learning_rate": 1e-4,      # learning rate when performing gradient update
    "batch_size": 32,           # how many states to sample from replay buffer at a time for training
    "buffer_size": 100_000,         # size of replay buffer from which we sample in order to train
    "gamma": 0.99,              # discount rate
    "train_start": 1e5,         # fill buffer before training begins
    "train_freq": 4,            # env steps between gradient updates
    "target_update_freq": 8000, # steps between target-net syncs

    # contains parameters for epsilon value for epsilon-greedy policy in early learning/testing. CONFLICTS WITH NOISY NETS
    "eps_start": 1.0,           # starting value of epsilon, which will decay over time
    "eps_end": 0.01,            # value of epsilon to eventually end at, after decaying
    "eps_decay_steps": 100000, # number of steps for epsilon to decay over

    # contains parameters for controlling how logs are printed
    "log_freq": 1000,   # steps between TensorBoard writes
    "eval_freq": 5e4,   # steps between evaluation runs
    "eval_episodes": 5,
    "save_freq": 1e5,   # steps between checkpoint saves

    # contains flags dictating whether or not we are using certain additional rainbow DQN features
    "double_dqn": False,     # a boolean to set whether or not we are using 2 DQN's. True = use double DQN, False = don't (for early testing)
    "per": False,            # a boolean to set whether or not we are using the prioritized experience replay. True = use PER, False = don't
    "dueling": False,        # a boolean to set whether or not we are using dueling networks.
    "distributional": False, # a boolean to set whether or not we are using distributional future rewards
    "noisy_nets": False,     # a boolean to set whether or not we are using noisy nets for exploration. CONFLICTS WITH EPSILON-GREEDY
}