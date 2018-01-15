import gym

env = gym.make("Taxi-v2")
PRINT = False
state = env.reset()
env.render()
counter = 0
G = 0
reward = None

while reward != 20:
    state, reward, done, info = env.step(env.action_space.sample())
    if(PRINT):
        env.render()
    counter += 1
    G += reward

env.render()
print("Solved in {} Steps with a total reward of {}".format(counter,G))
