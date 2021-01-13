# Reinforcement Learning and Tic Tac Toe Game

In this repository you can find a script to train an AI agent able to play tic-tac-toe game by means reinforcement learning (Q-learning algorithm, off-policy TD control) and where the tic-tac-toe envorinment is compliance to OpenAI Gym API. Moreover, a brief jupyter notebook allows you to analyze the results and the training process. Finally, a simple user interface tkinter API based is available in order to play against the trained AI agent.

<p align="center">
  <img src="https://github.com/Nick8812/tic-tac-toe-reinforcement-learning/blob/main/images/ai_tic_tac_toe_header.png">
</p>


## Project Structure

### src

The source code of the project is available inside the src folder.

In this folder you find the following python files:
- **agent.py**: it contains the *exploration* and *exploitation* features of agent, in addition to the implementation of Q-learning algorithm ([link](https://en.wikipedia.org/wiki/Q-learning)).
- **config.py**: this is the configuration file to play the tic-tac-toe game (agents' symbol and board value).
- **envorinment.py**: it contains the tic-tac-toe envorinment where the agent interacts and learns by the experience. Its development is based to standard OpenAI Gym API ([link](https://gym.openai.com/)). So here you can find the main methods, such as *reset* to initialize the game, *step* in order to see the next state after a choosed action, *render* to visualize the tic-tac-toe board and the *close* method.
- **qlearning.py**: this is the main script which allows to train the agent against a completely random player. The agent control parameters are settable at the begging of the script. In order to control the reinforcement learning process at every step of every episode, a log is generated and saved in a specific folder named log (and automatically created when you start the training process). At the end of the process, the *static* folder is created inside the project folder to save the final state-action Q-value function of the agent.
- **ui.py**: this script based on tkinter API ([link](https://wiki.python.org/moin/TkInter)) is the game user interface to play against the previously trained agent. During the game, you can see at every step, which values the Q function assumes on the board. 

### test

The test folder contains the unittest to verify if the tic-tac-toe envorinment is compliance to OpenAI Gym API. The main method to test this property is based on RL Stable Baselines library ([link](https://stable-baselines.readthedocs.io/en/master/)).

### analysis

Inside analysis folder it is available a jupyter notebook with some simple statistics about the training process (eta per episode, total number of draws, total number of agent and opponent wins), the agent's epsilon and alpha values during the process, the time series of number of wins and draws, visualize Q-values dataframe and finally, some board states in order to understand the behaviour of the agent based on Q function values.

### images

The folder contains only the image displays in this README file.


## Install and Run the Application

The application is developed with Python 3.7 and the steps to use it are the following:

1. Install the project dependencies by the command
```
pip3 install -r requirements.txt
```

2. From the project folder, run the agent reinforcement learning by the command
```
python src/qlearning.py 
```

3. If you want play against the trained agent, run the user interface by the command
```
python src/ui.py 
```

## Improvements

The agent is able to contrast the opposite player, but at the beginning of the game it looks like it doesn't.

Some next step in order to improve the agent could be: 
- Tuning the agent's parameters (e.g. epsilon, alpha).
- Modifying the reward process.
- Using another RL algorithm to learn state-action Q function, for instance by deep learning models.

