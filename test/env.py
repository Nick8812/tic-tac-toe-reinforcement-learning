
# This script tests if tic-tac-toe envorinment is compliance to OpenAI Gym API by using Stable Baselines library

import sys
sys.path.append('src')

from config import configFile
from stable_baselines.common.env_checker import check_env
import unittest

# Class to test
from environment import TicTacToeEnv

# Import configuration data
SYMBOL_AGENT_1 = configFile['Agent_1_symbol']
SYMBOL_AGENT_2 = configFile['Agent_2_symbol']

class Testing(unittest.TestCase):
    def testOpenGymApiStyle(self):
        env = TicTacToeEnv()
        check_env(env, warn = False)


if __name__ == '__main__':
    unittest.main()
