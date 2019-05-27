import os

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import json

from utils.ingestion_program.runner import Runner #an override of pypownet.runner 
import pypownet.environment
from agents import brent_agents

# https://towardsdatascience.com/reinforcement-learning-w-keras-openai-actor-critic-models-f084612cfd69

class PolicyGradientRunner(Runner):

    def loop(self, iterations, episodes=1):
        """
        Runs the simulator for the given number of iterations time the number of episodes.
        :param iterations: int of number of iterations per episode
        :param episodes: int of number of episodes, each resetting the environment at the beginning
        :return:
        """
        cumul_rew = 0.0
        best_rew = 0.
        episode_rewards = []
        for i_episode in range(episodes):
            print('Episode', i_episode+1, 'of', episodes)
            S = []
            A = []
            R = []
            episode_rew = 0.
            observation = self.environment.reset()
            for i in range(1, iterations + 1):
                (observation, action, reward, reward_aslist, done) = self.step(observation)
                cumul_rew += reward
                episode_rew += reward
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
                    best_rew = max(best_rew, episode_rew)
                    self.agent.fit(np.array(S), np.array(A), np.array(R))
                    break
            # self.agent.fit(np.array(S), np.array(A), np.array(R))
            
            print(episode_rew)
            episode_rewards.append(episode_rew)
            # at the end of the inner for loop we train our neural net
            best_rew = max(best_rew, episode_rew)

        print('best episode reward =', best_rew)
        print('cumulative reward =', cumul_rew)
        fig = plt.figure()
        plt.plot(episode_rewards)
        fig.savefig('PolicyGradientRunner_rewards.png', dpi=300)
        plt.close()
        return cumul_rew


def set_environment(game_level="datasets", start_id=0, input_dir='public_data/'):
    """
    Load the first chronic (scenario) in the directory public_data/datasets 
    """
    return pypownet.environment.RunEnv(parameters_folder=os.path.abspath(input_dir),
                                              game_level=game_level,
                                              chronic_looping_mode='natural', start_id=start_id,
                                              game_over_mode="soft")


if __name__ == '__main__':
    save_weights_path = 'program/policy_grad_weights.h5'
    env_train = set_environment()
    agent = brent_agents.AgentPolicyGradient(env_train, mode='train')
    n_episodes = 100
    n_iterations = 32
    pg_runner = PolicyGradientRunner(env_train, agent, verbose=False)
    # Run the planned experiment of this phase with the submitted model
    score = pg_runner.loop(n_iterations, n_episodes)
    # todo serialize trained model
    agent.model.save_weights(save_weights_path)
