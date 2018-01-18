# -*- coding: utf-8 -*-
import logging
import gym

def main():
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    env = gym.make("Taxi-v2")
    env.reset()
    print('Total number of possible states : {}'.format(env.observation_space.n))
    env.render()
    # the yellow square represents the taxi
    # the (“|”) represents a wall
    # the blue letter represents the pick-up location
    # the purple letter is the drop-off location
    print('The actions available to the agent : {}'.format(env.action_space.n))

    # Se positionner sur l"état 114
    env.env.s = 114
    env.render()
    # Faire une pas
    state, reward, done, info = env.step(1)
    print('State: {}, reward: {}, done: {} , info: {}'.format(state, reward, done, info))
    env.render()

    # Faire un 2EME pas
    state, reward, done, info = env.step(1)
    print('State: {}, reward: {}, done: {} , info: {}'.format(state, reward, done, info))
    env.render()

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
