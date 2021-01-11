
# This script implements agent class and Q-learning updating step

import numpy as np
import random
from config import configFile
import logging

logger = logging.getLogger('root')
LENGTH = configFile['LENGTH']   # board width/height


class Agent:
    def __init__(self, eps = 0.1, gamma = 0.9, alpha = 0.1):
        self.sym = None            # agent symbol to play game
        self.eps = eps             # probability of choosing random action instead of the greedy one
        self.gamma = gamma         # discount rate
        self.alpha = alpha         # learning rate
        self.Q = None              # state-action value function
    
    def take_greedy_action(self, env, obs):
            obs = obs[0]
            Q_obs = self.Q[obs]

            # masking not available actions
            excludedActionsIdx = [idx for idx, action in env.dict_idx_actions.items() if action not in env.action_space.available_actions]
            m = np.zeros(Q_obs.size, dtype = bool)
            m[excludedActionsIdx] = True
            Q_obs = np.ma.array(Q_obs, mask = m)

            # greedy action
            idx_action = np.argmax(Q_obs)                   
            action = env.dict_idx_actions[idx_action]

            return action

    def take_action(self, env, obs):
        if random.uniform(0, 1) <= self.eps:
            action = env.action_space.sample()              # Exploration
        else:
            action = self.take_greedy_action(env, obs)      # Exploitation

        return action

    def updateQ(self, env, obs1, obs2, action, reward):
        """
        Implementation of Q-learning updating step
        
        OUTPUT: np.array, a random state
        """
        logger.debug('Updating Q(s,a), obs1: {0}, action: {1}, obs2: {2}, reward: {3}'.format(obs1[0], action, obs2[0], reward))

        # action index
        idx_action = [k for k, cell in env.dict_idx_actions.items() if cell == action][0]
        logger.debug('Old Value Q(obs1, a): {0}'.format(self.Q[obs1[0], idx_action]))

        # Q-learning updating step
        if not env.is_game_over:
            self.Q[obs1[0], idx_action] += self.alpha * (reward + self.gamma * max(self.Q[obs2[0]]) - self.Q[obs1[0], idx_action])
        else:
            self.Q[obs1[0], idx_action] += self.alpha * (reward - self.Q[obs1[0], idx_action])
            
        logger.debug('New Value Q(obs1, a): {0}'.format(self.Q[obs1[0], idx_action]))