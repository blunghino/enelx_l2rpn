import os
from os.path import join
import time 

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import json

from utils.ingestion_program.runner import Runner #an override of pypownet.runner 
from utils.scoring_program import evaluate
import pypownet.environment
from agents import brent_agents


class A2CRunner(Runner):
    """
    Runner for proper A2C with batched updates and entropy loss
    """
    def loop(self, iterations, episodes=1, break_on_done=True):

        episode_rewards = []
        all_logs = []
        cumul_reward = 0.
        best_reward = 0.
        for e in range(episodes):
            # reset environment for new episode
            observation = self.environment.reset()
            S = []
            A = []
            R = []
            done = False
            episode_reward = 0.
            current_chronic_name = self.environment.get_current_chronic_name()
            machine_logs = []
            t0 = time.time()
            for i in range(iterations):
                # one step of the environment/game
                (next_observation, action, reward, reward_aslist, done) = self.step(observation)
                # memory to use to update model at the end of each game
                S.append(observation)
                A.append(action)
                R.append(reward)
                # track rewards
                episode_reward += reward
                cumul_reward += reward
                observation = next_observation
                # write logs
                self.logger.info("step %d/%d - reward: %.2f; cumulative reward: %.2f" %
                                    (i, iterations, reward, cumul_reward))
                machine_logs.append([i,
                                     done,
                                     reward,
                                     [float(x) for x in reward_aslist],
                                     cumul_reward,
                                     str(self.environment.get_current_datetime()),
                                     time.time() - t0,
                                     action.tolist(),
                                     list(observation)
                                     ])

                # stop this episode when a "done" signal is received
                if done and break_on_done:
                    break
            # update model weights based on the past episode of experiences
            self.agent.train_model(np.array(S), np.array(A), np.array(R))
            # episode clean up
            episode_rewards.append(episode_reward)
            best_reward = max(best_reward, episode_reward)
            print("episode:", e, "  reward:", episode_reward)
            all_logs.append(machine_logs)

        print('best episode reward =', best_reward)
        print('cumulative reward =', cumul_reward)
        # write logs for plotting later
        self.dump_machinelogs(all_logs, current_chronic_name)
        # plot of rewards by episode
        fig = plt.figure()
        plt.plot(episode_rewards)
        plt.xlabel('episode #')
        plt.ylabel('reward')
        save_fig_dir = join(
            os.path.split(self.machinelog_filepath)[0], 
            'A2CRunner_rewards.png'
        )
        fig.savefig(save_fig_dir, dpi=300)
        plt.close()         
        return cumul_reward    


# https://github.com/rlcode/reinforcement-learning/blob/master/2-cartpole/4-actor-critic/cartpole_a2c.py
class ActorCriticRunner(Runner):

    def loop(self, iterations, episodes=1):

        episode_rewards = []
        all_logs = []
        cumul_reward = 0.
        best_reward = 0.
        for e in range(episodes):
            done = False
            episode_reward = 0.
            # reset environment for new episode
            observation = self.environment.reset()
            current_chronic_name = self.environment.get_current_chronic_name()
            machine_logs = []
            t0 = time.time()
            for i in range(iterations):
                # one step of the environment/game
                (next_observation, action, reward, reward_aslist, done) = self.step(observation)
                # update models given information from step
                self.agent.train_model(observation, action, reward, next_observation, done)
                # track rewards
                episode_reward += reward
                cumul_reward += reward
                observation = next_observation
                # write logs
                self.logger.info("step %d/%d - reward: %.2f; cumulative reward: %.2f" %
                                    (i, iterations, reward, cumul_reward))
                machine_logs.append([i,
                                     done,
                                     reward,
                                     [float(x) for x in reward_aslist],
                                     cumul_reward,
                                     str(self.environment.get_current_datetime()),
                                     time.time() - t0,
                                     action.tolist(),
                                     list(observation)
                                     ])

                # stop this episode when a "done" signal is received
                if done:
                    break
            # episode clean up
            episode_rewards.append(episode_reward)
            best_reward = max(best_reward, episode_reward)
            print("episode:", e, "  reward:", episode_reward)
            all_logs.append(machine_logs)

        print('best episode reward =', best_reward)
        print('cumulative reward =', cumul_reward)
        # write logs for plotting later
        self.dump_machinelogs(all_logs, current_chronic_name)
        # plot of rewards by episode
        fig = plt.figure()
        plt.plot(episode_rewards)
        plt.xlabel('episode #')
        plt.ylabel('reward')
        save_fig_dir = join(
            os.path.split(self.machinelog_filepath)[0], 
            'ActorCriticRunner_rewards.png'
        )
        fig.savefig(save_fig_dir, dpi=300)
        plt.close()         
        return cumul_reward

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
                    break
            # self.agent.fit(np.array(S), np.array(A), np.array(R))
            
            best_rew = max(best_rew, episode_rew)
            self.agent.fit(np.array(S), np.array(A), np.array(R))
            print(episode_rew)
            episode_rewards.append(episode_rew)
            # at the end of the inner for loop we train our neural net
            best_rew = max(best_rew, episode_rew)

        print('best episode reward =', best_rew)
        print('cumulative reward =', cumul_rew)
        fig = plt.figure()
        plt.plot(episode_rewards)
        fig.savefig('PolicyGradientRunner_rewards.png', dpi=300)
        return cumul_rew


def set_environment(game_level="datasets", start_id=0, input_dir='public_data/'):
    """
    Load the first chronic (scenario) in the directory public_data/datasets 
    """
    return pypownet.environment.RunEnv(parameters_folder=os.path.abspath(input_dir),
                                              game_level=game_level,
                                              chronic_looping_mode='natural', 
                                              start_id=start_id,
                                              game_over_mode="soft")


if __name__ == '__main__':
    n_episodes = 100
    n_iterations = 100
    # where to save stuff
    log_dir = join('utils', 'logs')
## policy gradient agent training
    # save_weights_path = 'program/policy_grad_weights.h5'
    # env_train = set_environment()
    # agent = brent_agents.AgentPolicyGradient(env_train, mode='train')
    # pg_runner = PolicyGradientRunner(env_train, agent, verbose=False)
    # # Run the planned experiment of this phase with the submitted model
    # score = pg_runner.loop(n_iterations, n_episodes)
    # # todo serialize trained model
    # agent.model.save_weights(save_weights_path)

## Actor/critic agent training
    # machinelog_filepath = join(log_dir, 'ActorCriticRunner_machinelog.json')
    # # agent in train mode
    # env_train = set_environment()
    # agent = brent_agents.AgentActorCritic(env_train, mode='train')
    # ac_runner = ActorCriticRunner(env_train, agent, verbose=False, 
    #                               machinelog_filepath=machinelog_filepath)
    # # Run the planned experiment of this phase with the submitted model
    # score = ac_runner.loop(n_iterations, n_episodes)


## A2C agent training
    machinelog_filepath = join(log_dir, 'A2CRunner_machinelog.json')
    # agent in train mode
    env_train = set_environment()
    agent = brent_agents.A2CAgent(
        env_train, 
        mode='train', 
        gamma=0.9, 
        actor_lr=1e-4, 
        critic_lr=1e-3, 
        entropy_weight=32.,
        actor_hidden_dims=[100],
        critic_hidden_dims=[30],
    )
    a2c_runner = A2CRunner(env_train, agent, verbose=False, 
                                  machinelog_filepath=machinelog_filepath)
    # Run the planned experiment of this phase with the submitted model
    score = a2c_runner.loop(n_iterations, n_episodes)


## serialize trained model
    agent.actor.save_weights('program/{}_actor_weights.h5'.format(agent.__class__.__name__))
    agent.critic.save_weights('program/{}_critic_weights.h5'.format(agent.__class__.__name__))
## FIGURES
    # more figures, showing action distributions from log data
    action_space = env_train.action_space
    # action_space = evaluate.get_action_space(join('utils', "ref"))
    with open(machinelog_filepath, 'r') as json_file:
        data = json.load(json_file)
    action_label = data["labels"]["action"]
    # track action counts in this dict
    action_counter_cumul = {k: 0 for k in evaluate.list_possible_actions(action_space)}
    # get action counts over all episodes 
    for j in range(len(data['log'])):
        actions = np.array(np.array(data["log"][j], dtype=object)[:, action_label])
        action_counter, count_three_types = evaluate.action_count(action_space, actions)
        for k in action_counter_cumul.keys():
            action_counter_cumul[k] += action_counter[k]
    # plot action distribution
    fig = plt.figure(figsize=(11,6))
    x = list(action_counter_cumul.keys())
    h = [action_counter_cumul[el] for el in x]
    plt.bar(x, h, 1)
    plt.title("Distribution of actions")
    plt.xticks(rotation=45)
    fig.savefig(join(log_dir, '{}_action_distribution.png'.format(agent.__class__.__name__)), dpi=300)
    # from utils.visualize_grid import plot_grid
    # from plotly.offline import plot
    # # Plot the grid
    # grid_after_action = plot_grid(env_train, 
    #                               env_train.game.export_observation(), 
    #                               action_space.get_do_nothing_action())
    # plot(grid_after_action)
