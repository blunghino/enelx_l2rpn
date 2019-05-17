__author__ = 'marvinler'
# Copyright (C) 2017-2018 RTE and INRIA (France)
# Authors: Marvin Lerousseau <marvin.lerousseau@gmail.com>
# This file is under the LGPL-v3 license and is part of PyPowNet.
""" This is the machinnery that runs your agent in an environment. Note that this is not the machinnery of the update of the
environment; it is purely related to perform policy inference at each timestep given the last observation, and feeding
the reward signal to the appropriate function (feed_reward) of the Agent.

This is not intented to be modified during the practical.
"""
from pypownet.environment import RunEnv
from pypownet.agent import Agent
import logging
import logging.handlers
import csv
import datetime
import time
import json

LOG_FILENAME = 'runner.log'


class TimestepTimeout(Exception):
    pass


class Runner(object):
    def __init__(self, environment, agent, render=False, verbose=False, vverbose=False, parameters=None, level=None,
                 max_iter=None, log_filepath='runner.log', machinelog_filepath='machine_log.json'):
        # Sanity checks: both environment and agent should inherit resp. RunEnv and Agent
        assert isinstance(environment, RunEnv)
        assert isinstance(agent, Agent)

        # Logger part
        self.logger = logging.getLogger('pypownet')
        if machinelog_filepath is not None:
            self.parameters = parameters
            self.level = level
            self.max_iter = max_iter
            self.machinelog_filepath = machinelog_filepath
        else:
            # self.csv_writer = None
            self.machinelog_filepath = None
            self.parameters, self.level, self.max_iter = None, None, None

        # Always create a log file for runners
        fh = logging.FileHandler(filename=log_filepath, mode='w+')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(fh)

        if verbose or vverbose:
            # create console handler, set level to debug, create formatter
            ch = logging.StreamHandler()
            ch.setLevel(logging.DEBUG if vverbose and verbose else logging.INFO)
            ch.setFormatter(logging.Formatter('%(levelname)s        %(message)s'))
            self.ch = ch
            # add ch to logger
            self.logger.addHandler(ch)
            self.logger.setLevel(logging.DEBUG if vverbose else logging.INFO)

        self.environment = environment
        self.agent = agent
        self.verbose = verbose
        self.render = render

        self.max_seconds_per_timestep = self.environment.game.get_max_seconds_per_timestep()

        if self.render:
            self.environment.render()

    def step(self, observation):
        """
        Performs a full RL step: the agent acts given an observation, receives and process the reward, and the env is
        resetted if done was returned as True; this also logs the variables of the system including actions,
        observations.
        :param observation: input observation to be given to the agent
        :return: (new observation, action taken, reward received)
        """
        self.logger.debug('observation: ' + str(self.environment.observation_space.array_to_observation(observation)))
        action = self.agent.act(observation)
        action = self.environment.action_space._verify_action_shape(action).as_array()
        observation, reward_aslist, done, info = self.environment.step(action, do_sum=False)
        if done:
            self.logger.warning('\b\b\bGAME OVER! Resetting grid... (hint: %s)' % info.text)
            observation = self.environment.reset()
        elif info:
            self.logger.warning(info.text)

        reward = sum(reward_aslist)

        if self.render:
            self.environment.render()

        self.agent.feed_reward(action, observation, reward_aslist)

        self.logger.debug('action: {}'.format(action))
        self.logger.debug('reward: {}'.format('[' + ','.join(list(map(str, reward_aslist))) + ']'))
        self.logger.debug('done: {}'.format(done))
        self.logger.debug('info: {}'.format(info if not info else info.text))

        return observation, action, reward, reward_aslist, done

    def loop(self, iterations, episodes=1):
        """
        Runs the simulator for the given number of iterations time the number of episodes.
        :param iterations: int of number of iterations per episode
        :param episodes: int of number of episodes, each resetting the environment at the beginning
        :return:
        """
        cumul_rew = 0.0

        all_logs = list()
        for i_episode in range(episodes):
            observation = self.environment.reset()
            machine_logs = list()
            current_chronic_name = self.environment.get_current_chronic_name()
            t0 = time.time()
            for i in range(1, iterations + 1):
                (observation, action, reward, reward_aslist, done) = self.step(observation)
                cumul_rew += reward
                self.logger.info("step %d/%d - reward: %.2f; cumulative reward: %.2f" %
                                 (i, iterations, reward, cumul_rew))
                action_taken = action.tolist()
                machine_logs.append([i,
                                     done,
                                     reward,
                                     [float(x) for x in reward_aslist],
                                     cumul_rew,
                                     str(self.environment.get_current_datetime()),
                                     time.time() - t0,
                                     action_taken,
                                     list(observation)
                                     ])

            all_logs.append(machine_logs)
        self.dump_machinelogs(all_logs, current_chronic_name)
        return cumul_rew

    def dump_machinelogs(self, log, chronic_name):
        if self.machinelog_filepath is not None:
            output = dict()
            output["param_env_name"] = self.parameters
            output["level"] = self.level
            output["n_iter"] = len(log[0])
            output["chronic_name"] = chronic_name
            output["labels"] = {"iter": 0,
                                "game_over": 1,
                                "timestep_reward": 2,
                                "timestep_reward_aslist": 3,
                                "cumulated_reward": 4,
                                "game_time": 5,
                                "runtime": 6,
                                "action": 7,
                                "obsevation": 8}
            output["log"] = log

            with open(self.machinelog_filepath, 'w') as f:
                json.dump(output, f)


