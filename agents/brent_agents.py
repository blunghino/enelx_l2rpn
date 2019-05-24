import os

import numpy as np

from pypownet.agent import Agent, ActIOnManager
import pypownet.environment
import pypownet.agent


class AgentQ(Agent):
    """
    agent implementing Q learning
    this agent is limited to 1 action per timestep
    following https://github.com/simoninithomas/Deep_reinforcement_learning_Course/tree/master/Q%20learning
    """
    def __init__(self, environment, mode='test', qtable=None):
        super().__init__(environment)
        self.ioman = ActIOnManager(destination_path=os.path.join('saved_actions', 'AgentZero.csv'))
        n_state = environment.observation_space.n
        n_action = environment.action_space.n
        # Q(S,A)
        # note the Q table is initialized with the Agent. to train over multiple episodes the
        # environment must be reset but the agent must persist
        if isinstance(qtable, np.array) && qtable.shape == (n_state, n_action):
            self.qtable = qtable
        else:
            self.qtable = np.zeros((n_state, n_action))
        # todo these should probably live somewhere more accessible?
        # hyperparameters
        self.total_episodes = 15000        # Total episodes
        self.learning_rate = 0.8           # Learning rate
        self.max_steps = 99                # Max steps per episode
        self.gamma = 0.95                  # Discounting rate
        # Exploration parameters
        self.epsilon = 1.0                 # Exploration rate
        self.max_epsilon = 1.0             # Exploration probability at start
        self.min_epsilon = 0.01            # Minimum exploration probability 
        self.decay_rate = 0.005   

    def act(self, observation):
        # This agent needs to manipulate actions using grid contextual information, so the observation object needs
        # to be of class pypownet.environment.Observation: convert from array or raise error if that is not the case
        if not isinstance(observation, pypownet.environment.Observation):
            try:
                observation = self.environment.observation_space.array_to_observation(observation)
            except Exception as e:
                raise e
        # Sanity check: an observation is a structured object defined in the environment file.
        assert isinstance(observation, pypownet.environment.Observation)
        action_space = self.environment.action_space
        ## First we randomize a number
        exp_exp_tradeoff = random.uniform(0, 1)
        
        ## If this number > greater than epsilon --> exploitation (taking the biggest Q value for this state)
        if exp_exp_tradeoff > self.epsilon:
            action_id = np.argmax(self.qtable[observation,:])
            action = action_space.get_do_nothing_action()
            action[action_id] = 1
        # Else doing a random choice --> exploration
        else:
            action = action_space.sample()
        return action

    def feed_reward(self, action, new_observation, reward_aslist):
        """
        update the q table using the new observation
        """

class AgentZero(Agent):
    """
    do a combination of both random actions, should do nothing about 1/3 of the time
    """
    def __init__(self, environment):
        super().__init__(environment)
        self.ioman = ActIOnManager(destination_path=os.path.join('saved_actions', 'AgentZero.csv'))

    def act(self, observation):
        # This agent needs to manipulate actions using grid contextual information, so the observation object needs
        # to be of class pypownet.environment.Observation: convert from array or raise error if that is not the case
        if not isinstance(observation, pypownet.environment.Observation):
            try:
                observation = self.environment.observation_space.array_to_observation(observation)
            except Exception as e:
                raise e
        # Sanity check: an observation is a structured object defined in the environment file.
        assert isinstance(observation, pypownet.environment.Observation)
        action_space = self.environment.action_space

        # Create template of action with no switch activated (do-nothing action)
        action = action_space.get_do_nothing_action(as_class_Action=True)

        if np.random.choice([0, 1]):
            action_space.set_lines_status_switch_from_id(
                action=action,
                line_id=np.random.randint(action_space.lines_status_subaction_length),
                new_switch_value=1
            )
        if np.random.choice([0, 1]):
            # Select a random substation ID on which to perform node-splitting
            target_substation_id = np.random.choice(action_space.substations_ids)
            target_config_size = action_space.get_number_elements_of_substation(target_substation_id)
            # Choses a new switch configuration (binary array)
            target_configuration = np.random.choice([0, 1], size=(target_config_size,))

            action_space.set_substation_switches_in_action(
                action=action, 
                substation_id=target_substation_id,
                new_values=target_configuration
            )

            # Ensure changes have been done on action
            current_configuration, _ = action_space.get_substation_switches_in_action(action, target_substation_id)
            assert np.all(current_configuration == target_configuration)

        # Dump best action into stored actions file
        self.ioman.dump(action)

        return action

        # No learning (i.e. self.feed_reward does pass)


