import gym

env = gym.make("MsPacman-v0")
state = env.reset()
env.render()