
# This script implements the user interface based on tkinter API where it is possible to play against the trained agent
#     + the game is reseted by clicking the bottom below
#     + when the game is resetted the initial player is random

import time
import random
import tkinter as tk
import numpy as np
import pandas as pd
from environment import TicTacToeEnv
from agent import *
from config import configFile
from utils import log, utils_ui

# Init
logger = log.setup_custom_logger('root')
logger.setLevel('INFO')

SYMBOL_AGENT = configFile['Agent_1_symbol']
SYMBOL_HUMAN = configFile['Agent_2_symbol']
SYMBOL_AGENT_VALUE = configFile['Agent_1_symbol_value']
SYMBOL_HUMAN_VALUE = configFile['Agent_2_symbol_value']

agent = Agent()
agent.sym = SYMBOL_AGENT
agent.Q = np.load('static/Q.npy')

def refresh():
    # reset envorinment
    logger.info("Resetting envorinment")
    
    # refresh the board
    for i in range(3):
        for j in range(3):
            board[i][j]['text'] = ''

    obs = env.reset()
    env.symbol = random.choice([SYMBOL_AGENT, SYMBOL_HUMAN])
    logger.debug("First player: {0}".format(env.symbol))

    if env.symbol == SYMBOL_HUMAN:
        env.board = np.zeros((LENGTH, LENGTH))
    else:
        boardInit = utils_ui.obsToBoard(obs[0])
        valuesBoard = [x[0] for x in boardInit.reshape(9, 1)]
        idxAction = np.argmax([x == SYMBOL_AGENT_VALUE for x in valuesBoard])
        i, j = env.dict_idx_actions[idxAction]
        logger.debug("AGENT step ({0}, {1})".format(i,j))
        board[i][j].config(text = SYMBOL_AGENT)

def main_gameflow(env, r, c):
    isGameOver = env.is_game_over()
    logger.debug("Game over: {0}".format(isGameOver))
    logger.debug("Winner: %s" % env.winner)

    if board[r][c]['text'] == '' and not isGameOver:
        # HUMAN
        logger.debug("HUMAN step ({0}, {1})".format(r,c))
        board[r][c].config(text = SYMBOL_HUMAN)
        
        # updating envorinment
        env.board[r,c] = SYMBOL_HUMAN_VALUE

        # updating available actions    
        env.action_space.updateAvailableActions([(r,c)])
        logger.debug("Available actions: {0}".format(env.action_space.available_actions))
        
        isGameOver = env.is_game_over()
        logger.debug("Winner: %s" % env.winner)
        obs = env.get_obs()
        utils_ui.renderQOnBoard(env, obs[0], agent.Q)       

        if not isGameOver and env.winner is None:
            env.symbol = SYMBOL_HUMAN if env.symbol == SYMBOL_AGENT else SYMBOL_AGENT
            label.config(text = ("It's agent's turn"))
        elif env.winner == SYMBOL_HUMAN:
            label.config(text = ("You win!"))
        elif env.winner == SYMBOL_AGENT:
            label.config(text = ("Agent wins"))
        else:
            label.config(text = "Draw!")
        
        # AGENT
        if not isGameOver:
            obs = env.get_obs()
            i, j = agent.take_greedy_action(env, obs)
            logger.debug("AGENT step ({0}, {1})".format(i,j))
            board[i][j].config(text = SYMBOL_AGENT)
            env.board[i,j] = SYMBOL_AGENT_VALUE
            env.action_space.updateAvailableActions([(i,j)])
            logger.debug("Available actions: {0}".format(env.action_space.available_actions))
            
            isGameOver = env.is_game_over()
            logger.debug("Game over: {0}".format(isGameOver))
            logger.debug("Winner: %s" % env.winner)

            if not isGameOver and env.winner is None:
                env.symbol = SYMBOL_HUMAN if env.symbol == SYMBOL_AGENT else SYMBOL_AGENT
                label.config(text = ("It's your turn"))
            elif env.winner == SYMBOL_HUMAN:
                label.config(text = ("You win!"))
            elif env.winner == SYMBOL_AGENT:
                label.config(text = ("Agent wins"))
            else:
                label.config(text = "Draw!")


if __name__ == '__main__':
    window = tk.Tk()
    window.tk.call('tk', 'scaling', 1.0)
    window.title('Tic Tac Toe Game')

    env = TicTacToeEnv()
    env.symbol = SYMBOL_AGENT
    board = [[0,0,0],[0,0,0],[0,0,0]]
    logger.debug("Available actions: {0}".format(env.action_space.available_actions))

    for i in range(3):
        for j in range(3):
            board[i][j] = tk.Button(text = '', font = ('normal', 60, 'normal'), width = 5, height = 3, command = lambda env = env, r = i, c = j: main_gameflow(env, r, c))
            board[i][j].grid(row = i, column = j)

    label = tk.Label(text = "It's your turn", font = ('normal',22,'bold'))
    label.grid(row = 3, column = 1)
    button = tk.Button(text = 'restart', font = ('Courier',18,'normal'), fg = 'red', command = refresh)
    button.grid(row = 4,column = 1)
    window.mainloop()