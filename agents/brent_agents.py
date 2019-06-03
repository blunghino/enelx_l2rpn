"""
https://github.com/rlcode/reinforcement-learning/blob/master/2-cartpole/4-actor-critic/cartpole_a2c.py
"""
import os

import numpy as np
from keras import layers
from keras.models import Model
from keras import backend as K
from keras import utils as np_utils
from keras import optimizers
from keras import callbacks
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam

from pypownet.agent import Agent, ActIOnManager
import pypownet.environment
import pypownet.agent


def compute_discounted_R(R, discount_rate=.99):
    """
    https://gist.github.com/kkweon/c8d1caabaf7b43317bc8825c226045d2

    Returns discounted rewards
    Args:
        R (1-D array): a list of `reward` at each time step
        discount_rate (float): Will discount the future value by this rate
    Returns:
        discounted_r (1-D array): same shape as input `R`
            but the values are discounted
    Examples:
        >>> R = [1, 1, 1]
        >>> compute_discounted_R(R, .99) # before normalization
        [1 + 0.99 + 0.99**2, 1 + 0.99, 1]
    """
    discounted_r = np.zeros_like(R, dtype=np.float32)
    running_add = 0
    for t in reversed(range(len(R))):

        running_add = running_add * discount_rate + R[t]
        discounted_r[t] = running_add
    sigma = discounted_r.std()
    discounted_r -= discounted_r.mean() 
    discounted_r /= sigma

    return discounted_r




# A2C(Advantage Actor-Critic) agent
class AgentActorCritic(Agent):
    """
    https://github.com/rlcode/reinforcement-learning/blob/master/2-cartpole/4-actor-critic/cartpole_a2c.py
    """
    def __init__(self, environment, mode='test', 
                 actor_weights_file='program/actor_weights.h5', 
                 critic_weights_file='program/critic_weights.h5'):
        super().__init__(environment)
        self.ioman = ActIOnManager(destination_path=os.path.join('saved_actions', 'AgentActorCritic.csv'))
        self.mode = mode
        game = self.environment.game

        self.state_size = game.export_observation().as_array().shape[0]
        self.action_size = environment.action_space.get_do_nothing_action().shape[0]

        self.value_size = 1

        # These are hyper parameters for the Policy Gradient
        self.discount_factor = 0.99
        self.actor_lr = 0.001
        self.critic_lr = 0.005

        # create model for policy network
        self.actor = self.build_actor()
        self.critic = self.build_critic()

        if mode == 'test':
            self.actor.load_weights(actor_weights_file)
            self.critic.load_weights(critic_weights_file)


    # approximate policy and value using Neural Network
    # actor: state is input and probability of each action is output of model
    def build_actor(self):
        actor = Sequential()
        actor.add(Dense(100, input_dim=self.state_size, activation='relu',
                        kernel_initializer='he_uniform'))
        actor.add(Dense(self.action_size, activation='softmax',
                        kernel_initializer='he_uniform'))
        actor.summary()
        # See note regarding crossentropy in cartpole_reinforce.py
        actor.compile(loss='categorical_crossentropy',
                      optimizer=Adam(lr=self.actor_lr))
        return actor

    # critic: state is input and value of state is output of model
    def build_critic(self):
        critic = Sequential()
        critic.add(Dense(100, input_dim=self.state_size, activation='relu',
                         kernel_initializer='he_uniform'))
        critic.add(Dense(self.value_size, activation='linear',
                         kernel_initializer='he_uniform'))
        critic.summary()
        critic.compile(loss="mse", optimizer=Adam(lr=self.critic_lr))
        return critic

    # using the output of policy network, pick action stochastically
    def act(self, state):

        policy = self.actor.predict(state.reshape(1,-1)).flatten()

        action_idx = np.random.choice(self.action_size, 1, p=policy)[0]
        action_vec = self.environment.action_space.get_do_nothing_action()
        action_vec[action_idx] = 1

        return action_vec

    def train_model(self, state, action, reward, next_state, done):

        target = np.zeros((1, self.value_size))
        advantages = np.zeros((1, self.action_size))

        value = self.critic.predict(state.reshape(1,-1))[0]
        next_value = self.critic.predict(next_state.reshape(1,-1))[0]

        action_idx = np.where(action)[0][0]
        if done:
            advantages[0][action_idx] = reward - value
            target[0][0] = reward
        else:
            advantages[0][action_idx] = reward + self.discount_factor * (next_value) - value
            target[0][0] = reward + self.discount_factor * next_value

        self.actor.fit(state.reshape(1,-1), advantages, epochs=1, verbose=0)
        self.critic.fit(state.reshape(1,-1), target, epochs=1, verbose=0)



class AgentPolicyGradient(Agent):
    """
    https://gist.github.com/kkweon/c8d1caabaf7b43317bc8825c226045d2
    """
    def __init__(self, environment, mode='test', model_weights_file='program/policy_grad_weights.h5'):
        super().__init__(environment)
        self.ioman = ActIOnManager(destination_path=os.path.join('saved_actions', 'AgentPolicyGradient.csv'))
        self.mode = mode
        game = self.environment.game

        self.n_state = game.export_observation().as_array().shape[0]
        self.n_action = environment.action_space.get_do_nothing_action().shape[0]
        # self.prob_thresh = 0.9
        # self.choose_from_top_n = 5
        self.prev_action_idx = self.n_action

        self.__build_network(self.n_state, self.n_action)
        self.__build_train_fn()

        if mode == 'test' and model_weights_file is not None:
            self.model.load_weights(model_weights_file)



    def __build_network(self, input_dim, output_dim, hidden_dims=[100]):
        """Create a simple neural network"""
        self.X = layers.Input(shape=(input_dim,))
        net = self.X

        for h_dim in hidden_dims:
            net = layers.Dense(h_dim)(net)
            net = layers.Activation("relu")(net)

        net = layers.Dense(output_dim)(net)
        net = layers.Activation("softmax")(net)
        # net = layers.Activation("sigmoid")(net) # sigmoid for multilabel??

        self.model = Model(inputs=self.X, outputs=net)

    def __build_train_fn(self):
        """Create a train function

        like REINFORCE https://youtu.be/KHZVXao4qXs?t=2979
        Monte Carlo policy gradient

        It replaces `model.fit(X, y)` because we use the output of model and use it for training.
        For example, we need action placeholder
        called `action_one_hot` that stores, which action we took at state `s`.
        Hence, we can update the same action.
        This function will create
        `self.train_fn([state, action_one_hot, discount_reward])`
        which would train the model.
        """
        action_prob_placeholder = self.model.output
        action_onehot_placeholder = K.placeholder(shape=(None, self.n_action),
                                                  name="action_onehot")
        discount_reward_placeholder = K.placeholder(shape=(None,),
                                                    name="discount_reward")

        action_prob = K.sum(action_prob_placeholder * action_onehot_placeholder, axis=1)
        log_action_prob = K.log(action_prob)

        loss = - log_action_prob * discount_reward_placeholder
        loss = K.mean(loss)

        adam = optimizers.Adam()

        updates = adam.get_updates(params=self.model.trainable_weights,
                                   loss=loss)

        self.train_fn = K.function(inputs=[self.model.input,
                                           action_onehot_placeholder,
                                           discount_reward_placeholder],
                                    outputs=[],
                                    updates=updates)

    def act(self, observation):
        """Returns an action at given `state`
        Args:
            state (1-D or 2-D Array): It can be either 1-D array of shape (state_dimension, )
                or 2-D array shape of (n_samples, state_dimension)
        Returns:
            action: an integer action value ranging from 0 to (n_actions - 1)
        """
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
        state_vec = observation.as_ac_minimalist().as_array().astype(int)
        shape = state_vec.shape   

        if len(shape) == 1:
            assert shape == (self.n_state,)
            state = np.expand_dims(state_vec, axis=0)

        elif len(shape) == 2:
            assert shape[1] == (self.n_state)

        action_prob = np.squeeze(self.model.predict(state))
        assert len(action_prob) == self.n_action


        # action_idxs = np.random.choice(np.arange(self.n_action), p=action_prob, size=2).tolist()
        # action_idx = action_idxs.pop(np.random.randint(0, 2))
        # # don't allow same action twice in a row
        # if action_idx == self.prev_action_idx:
        #     action_idx = action_idxs.pop()
        # self.prev_action_idx = action_idx

        action_idx = np.random.choice(np.arange(self.n_action), p=action_prob)
        action_vec = action_space.get_do_nothing_action()
        action_vec[action_idx] = 1

        # note right now we can only take one action at a time. 
        # we could modify this to optimize the whole action vector at once?
        # return (action_prob > self.prob_thresh).astype(int)
        return action_vec

    def fit(self, S, A, R):
        """Train a network
        Args:
            S (2-D Array): `state` array of shape (n_samples, state_dimension)
            A (1-D Array): `action` array of shape (n_samples,)
                It's simply a list of int that stores which actions the agent chose
            R (1-D Array): `reward` array of shape (n_samples,)
                A reward is given after each action.
        """
        action_onehot = np_utils.to_categorical(A, num_classes=self.n_action)
        discount_reward = compute_discounted_R(R)

        assert S.shape[1] == self.n_state
        assert action_onehot.shape[0] == S.shape[0]
        assert action_onehot.shape[1] == self.n_action
        assert len(discount_reward.shape) == 1

        self.train_fn([S, action_onehot, discount_reward])


class AgentQ(Agent):
    """
    agent implementing Q learning
    this agent is limited to 1 action per timestep
    following https://github.com/simoninithomas/Deep_reinforcement_learning_Course/tree/master/Q%20learning
    """
    def __init__(self, environment, mode='test', qtable=None):
        super().__init__(environment)
        self.ioman = ActIOnManager(destination_path=os.path.join('saved_actions', 'AgentZero.csv'))
        self.game = self.environment.game
        self.mode = mode
        # initialize storage of state vectors (for Q-learning)
        self.state_vec = self.get_state_vector()
        self.new_state_vec = self.get_state_vector()
        self.n_state = self.state_vec.shape[0]
        self.n_action = environment.action_space.get_do_nothing_action().shape[0]
        # Q(S,A)
        # note the Q table is initialized with the Agent. to train over multiple episodes the
        # environment must be reset but the agent must persist
        if isinstance(qtable, np.ndarray) and (qtable.shape == (self.n_state, self.n_action)):
            self.qtable = qtable
        else:
            self.qtable = np.zeros((self.n_state, self.n_action))
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
        self.state_vec = self.get_state_vector()
        ## First we randomize a number
        exp_exp_tradeoff = np.random.uniform(0, 1)
        
        ## If this number > greater than epsilon --> exploitation (taking the biggest Q value for this state)
        if (self.mode == 'test') or (exp_exp_tradeoff > self.epsilon):
            action_idx = np.argmax(self.qtable[self.state_vec])
            action = action_space.get_do_nothing_action()
            # flip the bit on the best action
            action[action_idx] = int(not action[action_idx])
        # Else doing a random choice --> exploration
        else:
            action = action_space.sample()
        return action

    def get_state_vector(self):
        obs_min = self.game.export_observation().as_ac_minimalist().as_array()
        return obs_min.astype(int)  # A bit of a hack converting values to ints for q-learning indices


    def feed_reward(self, action, new_observation, reward_aslist):
        """
        update the q table using the new observation
        """
        # this is an array of all zeros and one 1?
        self.new_state_vec = self.get_state_vector()

        reward = sum(reward_aslist)
        # Get the old q-value
        old_value = self.qtable[self.state_vec, action]  
        
        # Look-up the next maximum q action given the new state
        new_max = np.max(self.qtable[self.new_state_vec])
        # Update the new q value
        new_value = (1 - self.alpha) * old_value + self.alpha * \
            (reward + self.gamma * new_max)
        self.qtable[self.state_vec, action] = new_value


