import pypownet.agent
import pypownet.environment

class DoNothingAgent(pypownet.agent.Agent):
    """ The template to be used to create an agent: any controller of the power grid is expected to be a daughter of this
    class.
    """

    def __init__(self, environment):
        """Initialize a new agent."""
        assert isinstance(environment, pypownet.environment.RunEnv)
        self.environment = environment

    def act(self, observation):
        return self.environment.action_space.get_do_nothing_action()

    def feed_reward(self, action, consequent_observation, rewards_aslist):
        pass