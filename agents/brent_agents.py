import os

import numpy as np

from pypownet.agent import Agent, ActIOnManager
import pypownet.environment
import pypownet.agent


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


