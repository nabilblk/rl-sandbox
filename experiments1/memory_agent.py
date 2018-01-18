import numpy as np
import gym

env = gym.make("Taxi-v2")
Q = np.zeros([env.observation_space.n, env.action_space.n])
G = 0
alpha = 0.618
TESTING = True

# Training
for episode in range(1,1001):
    done = False
    G, reward, counter = 0,0,0
    state = env.reset()
    while done != True:
        counter += 1
        action = np.argmax(Q[state]) #1
        state2, reward, done, info = env.step(action) #2
        Q[state,action] += alpha * (reward + np.max(Q[state2]) - Q[state,action]) #3
        G += reward
        state = state2

    if episode % 50 == 0:
        print('Episode {} Solved in {} Total Reward: {}'.format(episode,counter,G))

# Testing
if(TESTING):
    state = env.reset()
    done = None

    while done != True:
        # We simply take the action with the highest Q Value
        action = np.argmax(Q[state])
        state, reward, done, info = env.step(action)
        env.render()