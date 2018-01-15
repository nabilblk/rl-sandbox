import gym
import pandas as pd
import numpy as np
import xgboost as xgb
import random

FEATURES = ['observation_1','observation_2', 'observation_3', 'observation_4', 'action']
FRAME_COLUMNS = ['index'] + FEATURES + ['future_reward']
XGB_PARAMS = {
              'eta': 0.05,
              'max_depth': 5,
              'silent': 1,
              'gamma':5,
              'lambda': 10
            }
MAX_ITERATIONS = 2000
MAX_TIME_FRAMES = 500
MAX_EPISODES = 200
FINAL_FRAME = 200


class Cache:
    
    def __init__(self):
        self.cache = []
        self.index = 0
   
    def cache_data(self, observation, action, time_frame):
        cache_data = np.append(observation,[action,time_frame])
        indexed_cache_data = np.append(self.index, cache_data)
        self.cache.append(indexed_cache_data)
        self.index += 1
    
    def get_frame(self):
        df_cache = pd.DataFrame(columns=FRAME_COLUMNS, data=self.cache)
        
        # Normalize reward
        future_reward = df_cache['future_reward'].values
        max_future_reward = np.max(future_reward)
        df_cache['future_reward'] = max_future_reward - future_reward
        
        return df_cache
    
    
class Memory:
    
    def __init__(self):
        self.df_data = pd.DataFrame(columns=FRAME_COLUMNS)
    
    def add_cache(self, cache):
        self.df_data = pd.concat([self.df_data, cache.get_frame()])
        

class Brain:
    
    def __init__(self, memory):
        self.regressor = None
        self.memory = memory
        
    def train(self, ):
        
        msk = np.random.rand(len(self.memory.df_data)) < 0.90 # 10% of data is used for early stopping
        train, validation = self.memory.df_data[msk], self.memory.df_data[~msk]
        
        train_xgdmat =  xgb.DMatrix(train[FEATURES], label=train['future_reward'])
        validation_xgdmat =  xgb.DMatrix(validation[FEATURES], label=validation['future_reward'])
        watchlist = [(validation_xgdmat, 'test')]
        self.regressor = xgb.train(XGB_PARAMS, train_xgdmat, MAX_ITERATIONS, watchlist, verbose_eval=False)
        
    def is_exploration(self, episode):
        return episode < 5 or (episode < 15 and episode % 2 == 0)
        
    def decide_action(self, observations, episode):
        if self.is_exploration(episode):
            return random.randint(0, 1)
        else:
            x_0 = np.append(observations, [0]).reshape(1,5)
            x_1 = np.append(observations, [1]).reshape(1,5)
            future_reward_0 = self.regressor.predict(xgb.DMatrix(x_0, feature_names=FEATURES))[0]
            future_reward_1 = self.regressor.predict(xgb.DMatrix(x_1, feature_names=FEATURES))[0]
            return 0 if future_reward_0 > future_reward_1 else 1

env = gym.make('CartPole-v0')

memory = Memory()
brain=Brain(memory)

for episode in range(MAX_EPISODES):
    
    observation = env.reset()
    cache = Cache()
    
    done = False
    
    for time_frame in range(MAX_TIME_FRAMES):
        action = brain.decide_action(observation, episode)
        cache.cache_data(observation, action, time_frame)
        observation, _, done, _ = env.step(action)
        if done:
            print("Episode %d finished after %d timesteps" % (episode, time_frame+1))
            break
     
    memory.add_cache(cache)
    brain.train(episode)