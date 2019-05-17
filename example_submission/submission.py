
import pypownet.agent
import pypownet.environment

import sys

import pickle
import os

import numpy as np


class Submission(pypownet.agent.Agent):
    def act(self, observation):
        action_length = self.environment.action_space.action_length
        return np.zeros(action_length)

#if you want to load a file (in this directory) names "model.dupm"
#open("program/model.dump"