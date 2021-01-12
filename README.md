# Reinforcement Learning and Tic Tac Toe Game

In this repository you can find a script to train an AI agent to play tic-tac-toe game by means a reinforcement learning algorithm, Q-learning (off-policy TD control) where the tic-tac-toe envorinment is compliance to OpenAI Gym API. Moreover, a brief jupyter notebook allows you to analyze the results and the training process. Finally, a simple user interface tkinter API based is available in order to play against the trained AI agent.

## Project Structure

### src folder

The source code of the project is available inside the src folder.

In this folder you find the following python files:
- **agent.py**: this class contains the *exploration* and *exploitations* features of agent, in addition to the implementation of Q-learning algorithm.
- **config.py**: this is the configuration file to play the tic-tac-toe game.
- **envorinment.py**: this class is the tic-tac-toe envorinment where the agent interacts and learns by the experience. Its development is based to standard OpenAI Gym API. So here you can find the main methods, such as *reset* to initialize the game, *step* in order to see the next state after a choosed action, *render* to visualize the tic-tac-toe board and the *close* method.
- **qlearning.py**: this is the mail script which allows to train the agent against a completely random player. The agent control parameters are settable at the begging of the script. In order to control the reinforcement learning process at every step of every episode, a log is generated and saved in a specific folder (it's automatically created when you start the training process with name log). At the end of the process, the *static* folder is created inside the project folder to save the state-action Q-value function of the agent.

### test folder

The test folder contains the unittest to verify if the tic-tac-toe envorinment is compliance to OpenAI Gym API.

### analysis folder

Inside analysis folder it is available a jupyter notebook with some simple statistics about the training process (eta per episode, total number of draws, total number of agent wins and opponent wins), the agent's epsilon and alpha values during the process, the time series of number of wins and draws, visualize Q-values dataframe and finally, some board states in order to understand the behaviour of the agent based on Q function.
