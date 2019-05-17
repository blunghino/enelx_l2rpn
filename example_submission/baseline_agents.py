import pypownet.agent
from pypownet.agent import Agent, ActIOnManager
import pypownet.environment
import numpy as np


class DoNothing(Agent):
    def act(self, observation):
        action_length = self.environment.action_space.action_length
        return np.zeros(action_length)

class RandomLineSwitch(Agent):
    """
    An example of a baseline controller that randomly switches the status of one random power line per timestep (if the
    random line is previously online, switch it off, otherwise switch it on).
    """

    def __init__(self, environment):
        super().__init__(environment)

        self.ioman = ActIOnManager(destination_path='saved_actions_RandomLineSwitch.csv')

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
        action_space.set_lines_status_switch_from_id(action=action,
                                                     line_id=np.random.randint(
                                                         action_space.lines_status_subaction_length),
                                                     new_switch_value=1)

        # Dump best action into stored actions file
        self.ioman.dump(action)

        return action

        # No learning (i.e. self.feed_reward does pass)


class RandomNodeSplitting(Agent):
    """ Implements a "random node-splitting" agent: at each timestep, this controller will select a random substation
    (id), then select a random switch configuration such that switched elements of the selected substations change the
    node within the substation on which they are directly wired.
    """

    def __init__(self, environment):
        super().__init__(environment)

        self.ioman = ActIOnManager(destination_path='saved_actions_RandomNodeSplitting.csv')

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

        # Select a random substation ID on which to perform node-splitting
        target_substation_id = np.random.choice(action_space.substations_ids)
        expected_target_configuration_size = action_space.get_number_elements_of_substation(target_substation_id)
        # Choses a new switch configuration (binary array)
        target_configuration = np.random.choice([0, 1], size=(expected_target_configuration_size,))

        action_space.set_substation_switches_in_action(action=action, substation_id=target_substation_id,
                                                       new_values=target_configuration)

        # Ensure changes have been done on action
        current_configuration, _ = action_space.get_substation_switches_in_action(action, target_substation_id)
        assert np.all(current_configuration == target_configuration)

        # Dump best action into stored actions file
        self.ioman.dump(action)

        return action