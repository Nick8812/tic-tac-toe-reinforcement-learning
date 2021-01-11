
# This script implements a custom tic-tac-toe envorinment based on OpenAI Gym API

import numpy as np
import random
import itertools
import gym
from gym import spaces
from config import configFile

# Configuration data
LENGTH = configFile['LENGTH']  # board width/height


# This class represents a tic-tac-toe board
class TicTacToeEnv(gym.Env):
    """
    TicTacToe envorinment OpenAI Gym API based
    """

    def __init__(self):
        # Inherit OpenAI Gym API envorinment init methods
        super(TicTacToeEnv, self).__init__()

        # Initialize empty board
        self.board = np.zeros((LENGTH, LENGTH))

        # Define agents values
        self.symbol = None
        self.start_symbol = configFile['Agent_1_symbol']
        self.x = configFile['Agent_1_symbol_value']  # represents x on the board, agent 1
        self.o = configFile['Agent_2_symbol_value']  # represents o on the board, agent 2

        # Define action and observation spaces
        self.action_space = self.DynamicTuple(3)
        self.dict_idx_actions = dict(zip(range(LENGTH * LENGTH),
                                         list(itertools.product(range(LENGTH), range(LENGTH)))))

        self.observation_space = spaces.Box(low = 0, high = 3 ** (LENGTH * LENGTH),
                                            shape = (1,), dtype = np.int16)

        # Data to track if game is over
        self.done = None
        self.winner = None

    def reset(self):
        """
        Reset envorinment at the beginning of episode and return a random initial state
        
        OUTPUT: np.array, a random state
        """
        # Initialize the board
        self.action_space = self.DynamicTuple(3)
        self.board = np.zeros((LENGTH, LENGTH))
        self.done = False
        self.symbol = self.start_symbol
        self.winner = None

        # Choose a random cell
        i, j = self.action_space.sample()
        self.board[i, j] = self.x if self.symbol == 'x' else self.o
        self.action_space.available_actions = self._get_available_actions()

        return self.get_obs()

    def step(self, action):
        """
        Run one timestep of the environment dynamics, taking an action
        
        OUTPUT: state, reward, done, info
        """
        # An action is an available cell where inserting the agent's symbol (e.g. 'x' or 'o')
        done = self.is_game_over()

        if done:
            return self._get_obs(), self._reward(self.symbol), True, dict()
        else:
            i, j = action
            symbol_value = self.x if self.symbol == configFile['Agent_1_symbol'] else self.o
            self.board[i, j] = symbol_value
            reward = self._reward(self.symbol)
            self.done = self.is_game_over()
            info = dict()

        return self.get_obs(), reward, self.done, info

    def render(self, mode = 'console'):
        '''
        Visualizing the current board
        '''
        if mode != 'console':
            raise NotImplementedError()

        for i in range(LENGTH):
            print("-------------")
            for j in range(LENGTH):
                print("  ", end="")
                if self.board[i, j] == self.x:
                    print("x ", end="")
                elif self.board[i, j] == self.o:
                    print("o ", end="")
                else:
                    print("  ", end="")
            print("")
        print("-------------")

    def close(self):
        '''
        Closing the envorinment
        '''
        pass

    def _reward(self, sym):
        """
        The agent with the input symbol obtains the reward if it wins (+1) o loses (-1)
        """
        # No reward until game is over
        if not self.is_game_over():
            return 0
        else:
            # draw
            if self.winner == None:                  
                return 0
            # player wins
            elif self.winner == sym:
                return 1
            # player loses
            else:
                return -1

    def _getPlayerSymbolByValue(self, player):
        """
        It returns the symbol of the agent by means its symbol value
        
        INPUT: player, int, agent's symbol value
        OUTPUT: str, agent's symbol 
        """
        return configFile['Agent_1_symbol'] if player == self.x else configFile['Agent_2_symbol']

    def is_game_over(self):
        """
        returns true if game over (a player has won or it's a draw) otherwise returns false
        also sets 'winner' instance variable and 'done' instance variable
        """       
        # check rows
        for i in range(LENGTH):
            for player in (self.x, self.o):
                if self.board[i].sum() == player * LENGTH:
                    self.winner = self._getPlayerSymbolByValue(player)
                    self.done = True
                    return True

        # check columns
        for j in range(LENGTH):
            for player in (self.x, self.o):
                if self.board[:, j].sum() == player * LENGTH:
                    self.winner = self._getPlayerSymbolByValue(player)
                    self.done = True
                    return True

        # check diagonals
        for player in (self.x, self.o):
            # top-left -> bottom-right diagonal
            if self.board.trace() == player * LENGTH:
                self.winner = self._getPlayerSymbolByValue(player)
                self.done = True
                return True
            # top-right -> bottom-left diagonal
            if np.fliplr(self.board).trace() == player * LENGTH:
                self.winner = self._getPlayerSymbolByValue(player)
                self.done = True
                return True

        # check if draw
        if np.all((self.board == 0) == False):
            # winner stays None
            self.winner = None
            self.done = True
            return True

        # game is not over
        self.winner = None

        return False

    def get_obs(self):
        """
        the current state value of the board
            + 1 step: a for cycle sees the symbol in a cell and it assigns a value (1 for 'x', 2 for 'o', 0 if it is empty)
            + 2 step: sum all previous values

        OUTPUT: np.array with one integer value
        """
        k = 0
        h = 0

        for i in range(LENGTH):
            for j in range(LENGTH):
                if self.board[i, j] == 0:
                    v = 0
                elif self.board[i, j] == self.x:
                    v = 1
                elif self.board[i, j] == self.o:
                    v = 2
                h += (3 ** k) * v
                k += 1

        return np.array([h], dtype = np.int16)

    def _get_available_actions(self):
            """
            the current list of available cells of the board

            OUTPUT: actions, list of tuples
            """
            actions = []

            for i in range(LENGTH):
                for j in range(LENGTH):
                    if self.board[i, j] == 0:
                        actions.append((i, j))

            return actions

    class DynamicTuple(gym.Space):
        """
        Dynamic custom actions space

        Example usage:
        from gym import spaces
        DynamicTuple(length = 3)
        """
        def __init__(self, length):
            self.available_actions = list(itertools.product(range(length), range(length)))
            self.n = length * length

        def updateAvailableActions(self, actions):
            self.available_actions = [a for a in self.available_actions if a not in actions]

        def sample(self):
            return random.choice(self.available_actions)
        
        def contains(self, x):
            return x in self.available_actions
               
        def __repr__(self):
            return "DynamicTuple(%d)" % self.n
        
        def __eq__(self, other):
            return self.n == other.n