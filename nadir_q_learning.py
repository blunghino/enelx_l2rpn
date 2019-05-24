# Define paths for submission, scoring, etc.
model_dir = 'agents/'  # Change to point to my model
# problem_dir = 'codalab_tools/ingestion_program/'
# score_dir = 'codalab_tools/scoring_program/'
# ref_data = 'codalab_tools/public_ref/'
# ingestion_output = 'codalab_tools/logs/'

input_dir = '../chronics_contestants/'
output_dir = 'codalab_tools/output/'

# Append directory paths to system path
from sys import path
path.append(model_dir)
# path.append(problem_dir)
# path.append(score_dir)
path.append(input_dir)
path.append(output_dir)

# load third-party libs from example notebook
import sys
import logging
import os
import warnings
import numpy as np
import pandas as pd
import evaluate
import matplotlib.pyplot as plt
import json

warnings.simplefilter(action='ignore', category=FutureWarning)

# Additional third-party libs for q-learning
import random

##
# Set file names for loading scenarios
# These are the files in each of the four-digit "chronics" folders
# Essentially they've provided a bunch of scenarios to train on
##
loads_p_file = '_N_loads_p.csv' #active power chronics for loads
prods_p_file = '_N_prods_p.csv'  #active power chronics for productions
datetimes_file = '_N_datetimes.csv' #timesstamps of the chronics
maintenance_file = 'maintenance.csv' #maintenance operation chronics. No maintenance considered in the first challenge
hazards_file = 'hazards.csv'   #harzard chronics that disconnect lines. No hazards considered in the first challenge
imaps_file = '_N_imaps.csv' #thermal limits of the lines

i = 0 # chronics id

# Initialize the environment
from utils.ingestion_program.runner import Runner
import pypownet.environment

parameters_path = '../chronics_contestant/chronics/0000/'

# Function from Run and Submit agent notebook
def set_environment(game_level = "datasets", start_id=0):
    """
        Load the first chronic (scenario) in the directory public_data/datasets
    """
    return pypownet.environment.RunEnv(parameters_folder=os.path.abspath(parameters_path),
                                              game_level=game_level,
                                              chronic_looping_mode='natural', start_id=start_id,
                                              game_over_mode="soft")


# ---- From Tutorial on Q Learning ----

# import gym
# import numpy as np
# import random
# from IPython.display import clear_output
#
# # Init Taxi-V2 Env
# env = gym.make("Taxi-v2").env
#
# # Init arbitary values
# q_table = np.zeros([env.observation_space.n, env.action_space.n])
#
# # Hyperparameters
# alpha = 0.1
# gamma = 0.6
# epsilon = 0.1
#
#
# all_epochs = []
# all_penalties = []
#
# for i in range(1, 100001):
#     state = env.reset()
#
#     # Init Vars
#     epochs, penalties, reward, = 0, 0, 0
#     done = False
#
#     while not done:
#         if random.uniform(0, 1) < epsilon:
#             # Check the action space
#             action = env.action_space.sample()
#         else:
#             # Check the learned values
#             action = np.argmax(q_table[state])
#
#         next_state, reward, done, info = env.step(action)
#
#         old_value = q_table[state, action]
#         next_max = np.max(q_table[next_state])
#
#         # Update the new value
#         new_value = (1 - alpha) * old_value + alpha * \
#             (reward + gamma * next_max)
#         q_table[state, action] = new_value
#
#         if reward == -10:
#             penalties += 1
#
#         state = next_state
#         epochs += 1
#
#     if i % 100 == 0:
#         clear_output(wait=True)
#         print("Episode: {i}")
#
# print("Training finished.\n")