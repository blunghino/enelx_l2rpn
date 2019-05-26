import os

import numpy as np
import pandas as pd
import json

from utils.ingestion_program.runner import Runner #an override of pypownet.runner 
import pypownet.environment
from agents import brent_agents


class PolicyGradientRunner(Runner):

    def loop(self, iterations, episodes=1):
        """
        Runs the simulator for the given number of iterations time the number of episodes.
        :param iterations: int of number of iterations per episode
        :param episodes: int of number of episodes, each resetting the environment at the beginning
        :return:
        """
        cumul_rew = 0.0
        for i_episode in range(episodes):
            S = []
            A = []
            R = []
            observation = self.environment.reset()
            for i in range(1, iterations + 1):
                (observation, action, reward, reward_aslist, done) = self.step(observation)
                cumul_rew += reward
                obs_min = self.environment.observation_space.array_to_observation(observation) \
                                                            .as_ac_minimalist() \
                                                            .as_array() \
                                                            .astype(int)
                S.append(obs_min)
                A.append(np.where(action)[0][0])
                R.append(reward)
                self.logger.info("step %d/%d - reward: %.2f; cumulative reward: %.2f" %
                                 (i, iterations, reward, cumul_rew))
                if done:
                    self.agent.fit(np.array(S), np.array(A), np.array(R))
                    break

        return cumul_rew


def set_environment(game_level="datasets", start_id=0, input_dir='public_data/'):
    """
    Load the first chronic (scenario) in the directory public_data/datasets 
    """
    return pypownet.environment.RunEnv(parameters_folder=os.path.abspath(input_dir),
                                              game_level=game_level,
                                              chronic_looping_mode='natural', start_id=start_id,
                                              game_over_mode="soft")



env_train = set_environment()
agent = brent_agents.AgentPolicyGradient(env_train, mode='train')
rewards = []
n_episodes = 100
n_iterations = 100
pg_runner = PolicyGradientRunner(env_train, agent, verbose=True)
# Run the planned experiment of this phase with the submitted model
score = pg_runner.loop(n_iterations, n_episodes)
print(score)
