
import pypownet.agent
import pypownet.environment

import sys

import pickle
import os

import numpy as np

import agents.brent_agents


Submission = agents.brent_agents.AgentActorCritic

#if you want to load a file (in this directory) names "model.dupm"
#open("program/model.dump"