import torch
from torch import nn
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
import numpy as np


def layer_init(layer, std=0.5, bias_const=0.0):
    if hasattr(layer, "weight"):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class ActorCritic(nn.Module):
    
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.shared = nn.Sequential(
            layer_init(nn.Linear(state_dim,64)),
            nn.ReLU(),
            layer_init(nn.Linear(64,64)),
            nn.ReLU(),
        )
 
        self.actor = layer_init(nn.Linear(64, action_dim)) # returns actions
        self.critic = layer_init(nn.Linear(64, 1)) # a value only

    def action_value(self, state, action=None):
        """
        Returns actions, log probability of the actions, the entropy of the distribution and the value at the states

        :param state: The state
        :param action: If action is None then the action is randomly sampled from the policy distribution. 
                       Otherwise, the log probs are computed from the given action. 
        """ 
        # Hint: Use the Categorical distribution
        hidden_output = self.shared(state)
        action_logits = self.actor(hidden_output)
        state_value = self.critic(hidden_output).reshape((-1,)) # ensure the shape here
                                                         # is (1,)
        action_distribution = Categorical(logits=action_logits)
        if action is None:
            action = action_distribution.sample()
        action_log_probability = action_distribution.log_prob(action)
        entropy = action_distribution.entropy()
        return action, action_log_probability, entropy, state_value
    
    @torch.no_grad()
    def value(self, state):
        return self.critic(self.shared(state)).reshape((-1,))

class ContinuousActorCritic(nn.Module):

    def __init__(self, state_dim, action_dim):
        super().__init__()
        # Hint: Use the Normal distribution, and have 
        # a single logstd paramater for each action dim irrespective of state
        self.shared = nn.Sequential(
            layer_init(nn.Linear(state_dim,64)),
            nn.Tanh(),
            layer_init(nn.Linear(64,64)),
            nn.Tanh(),
        )

        self.log_standard_deviation = nn.Parameter(torch.zeros(action_dim))
 
        self.actor = layer_init(nn.Linear(64, action_dim)) # returns actions
        self.critic = layer_init(nn.Linear(64, 1)) # a value only

    def action_value(self, state, action=None):
        """
        Returns actions, log probability of the actions, the entropy of the distribution and the value at the states

        :param state: The state
        :param action: If action is None then the action is randomly sampled from the policy distribution. 
                       Otherwise, the log probs are computed from the given action. 
        """ 
        hidden_output = self.shared(state)
        state_value = self.critic(hidden_output).reshape((-1,))
        action_mus = self.actor(hidden_output)
        standard_deviation = torch.exp(self.log_standard_deviation)
        action_distribution = Normal(action_mus, standard_deviation)

        if action is None:
            action = action_distribution.sample()

        action_log_probability = action_distribution.log_prob(action).sum(-1)
        entropy = action_distribution.entropy().sum(-1)

        return action, action_log_probability, entropy, state_value
    
    @torch.no_grad()
    def value(self, state):
        return self.critic(self.shared(state)).reshape((-1,))
    