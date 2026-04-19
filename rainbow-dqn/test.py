# This file is for testing 

import stable_retro as retro

def main():
    env = retro.make(game="Airstriker-Genesis-v0")
    env.reset()
    while True:
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        env.render()
        if terminated or truncated:
            env.reset()
    env.close()

if __name__ == "__main__":
    main()