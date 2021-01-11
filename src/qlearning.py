
# This script implements the agent reinforcement learning by Q-learning algorithm (off-policy TD control)
#    + agent plays against a random agent (eps = 1)
#    + first player is switched at the beginning of any new episode
#    + eps and alpha agent parameters decrease every NUMBER_EPISODE_TO_DECREASE episodes

import os
import numpy as np
import pandas as pd
from collections import Counter
from datetime import datetime
from tqdm import tqdm
import csv
from environment import TicTacToeEnv
from agent import *
from config import configFile
from utils import log

# init logs 
logger = log.setup_custom_logger('root')
logger.setLevel('INFO')

if not os.path.exists('log'):
    os.makedirs('log')

logCsv = csv.writer(open('log/log_agent_training.csv', 'w'))
logCsv.writerow(['DATETIME', 'EPISODE', 'STEP', 'IS_GAME_OVER', 'PLAYER', 'ALPHA', 'EPS', 'OBS1', 'OBS2', 'ACTION', 'REWARD', 'Q_VALUE_OBS1'])

# init agents' parameters
SYMBOL_AGENT_1 = configFile['Agent_1_symbol']
SYMBOL_AGENT_2 = configFile['Agent_2_symbol']
NUM_EPISODE = 1000
ALPHA = 0.1
EPS = 0.2
GAMMA = 0.9
NUMBER_EPISODE_TO_DECREASE = 1000
DECREASE_RATE_ALPHA = 0.01
DECREASE_RATE_EPS = 0.02


if __name__ == '__main__':
    # Envorinment
    env = TicTacToeEnv()
    
    # Agents
    agent = Agent(eps = EPS, alpha = ALPHA, gamma = GAMMA)
    agentRandom = Agent(eps = 1)

    agent.sym = SYMBOL_AGENT_1
    agentRandom.sym = SYMBOL_AGENT_2

    dim_obs_space = env.observation_space.high[0]   # 19683 observations
    dim_acts_space = env.action_space.n             # 9 actions
    agent.Q = np.full((dim_obs_space, dim_acts_space), 0.)
    
    # track winners
    winners = []

    # Agent training
    for episode in tqdm(range(NUM_EPISODE)):
        logger.debug('Starting episode %s' % episode)
        step = 1  # first step
        
        # switch first player
        env.start_symbol = agent.sym if episode % 2 == 0 else agentRandom.sym
        obsCurrent = env.reset()
        logCsv.writerow([datetime.now().strftime("%d/%m/%Y %H:%M:%S"), episode+1, step, env.done, env.start_symbol, agent.alpha, agent.eps, 0, obsCurrent[0], None, 0, None])

        while True:
            # 1- switch player
            env.symbol = agentRandom.sym if env.symbol == agent.sym else agent.sym

            # 2- choose an action by agent
            if env.symbol == agent.sym:
                actionCurrent = agent.take_action(env, obsCurrent)
            else:
                actionCurrent = agentRandom.take_action(env, obsCurrent)

            # updating available actions    
            env.action_space.updateAvailableActions([actionCurrent])

            # 3- one step of the agents
            obsNext, reward, done, info = env.step(actionCurrent)

            if env.symbol == agent.sym:
                agent.updateQ(env, obsCurrent, obsNext, actionCurrent, reward)
                logCsv.writerow([datetime.now().strftime("%d/%m/%Y %H:%M:%S"), episode+1, step+1, env.done, SYMBOL_AGENT_1, agent.alpha, agent.eps, obsCurrent[0], obsNext[0], str(actionCurrent), reward, agent.Q[obsCurrent[0]]])
            else:
                logCsv.writerow([datetime.now().strftime("%d/%m/%Y %H:%M:%S"), episode+1, step+1, env.done, SYMBOL_AGENT_2, agent.alpha, agent.eps, obsCurrent[0], obsNext[0], str(actionCurrent), None, None])
                           
            # game is over
            if done:
                # if agent loses, last opponent action is the one it should choose in order to prevent the lose
                if env.winner != SYMBOL_AGENT_1 and env.winner is not None:
                    rewardLoss = env._reward(SYMBOL_AGENT_1)
                    agent.updateQ(env, obsPrevious, obsCurrent, actionCurrent, -1 * rewardLoss)
                    logCsv.writerow([datetime.now().strftime("%d/%m/%Y %H:%M:%S"), episode+1, step+1, env.done, SYMBOL_AGENT_1, agent.alpha, agent.eps, obsPrevious[0], obsCurrent[0], str(actionCurrent), -1 * rewardLoss, agent.Q[obsPrevious[0]]])
                winners.append(env.winner)
                logger.debug("Winner: {0}".format(env.winner))
                logger.debug("Episode finished after {} timesteps".format(step))
                break
            
            # next step
            step += 1
            obsPrevious = obsCurrent
            actionPrevious = actionCurrent
            obsCurrent = obsNext

        # decreasing learning rate e eps
        if (episode+1) % NUMBER_EPISODE_TO_DECREASE == 0:
            agent.alpha *= (1-DECREASE_RATE_ALPHA)
            agent.eps *= (1-DECREASE_RATE_EPS)

    env.close()
    
    # Agent wins results
    winners = {winner: round(count*100/NUM_EPISODE, 2) for winner, count in Counter(winners).items()}
    logger.info('Winners count: {0}'.format(winners))


    # Save state-action value function
    logger.info('Saving Q function...')
    
    if not os.path.exists('static'):
        os.makedirs('static')

    np.save('static/Q.npy', agent.Q)
    logger.info('Training successfully finished')
