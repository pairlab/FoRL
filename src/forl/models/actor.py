import torch
import torch.nn as nn
from torch.distributions.normal import Normal

from forl.models import model_utils


class ActorDeterministicMLP(nn.Module):
    def __init__(self, obs_dim, action_dim, units, activation: str, init_gain = 2.0**0.5):
        super(ActorDeterministicMLP, self).__init__()

        self.layer_dims = [obs_dim] + units + [action_dim]

        init_ = lambda m: model_utils.init(
            m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), init_gain
        )

        modules = []
        for i in range(len(self.layer_dims) - 1):
            modules.append(init_(nn.Linear(self.layer_dims[i], self.layer_dims[i + 1])))
            if i < len(self.layer_dims) - 2:
                modules.append(model_utils.get_activation_func(activation))
                modules.append(torch.nn.LayerNorm(self.layer_dims[i + 1]))

        self.actor = nn.Sequential(*modules)

        self.action_dim = action_dim
        self.obs_dim = obs_dim

    def forward(self, observations, deterministic=False):
        return self.actor(observations)


class ActorStochasticMLP(nn.Module):
    def __init__(
        self,
        obs_dim,
        action_dim,
        units,
        activation: str,
        init_gain = 2.0**0.5,
        logstd_init: float = -1.0,
    ):
        super(ActorStochasticMLP, self).__init__()

        self.layer_dims = [obs_dim] + units + [action_dim]

        init_ = lambda m: model_utils.init(
            m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), init_gain
        )

        modules = []
        for i in range(len(self.layer_dims) - 1):
            modules.append(init_(nn.Linear(self.layer_dims[i], self.layer_dims[i + 1])))
            if i < len(self.layer_dims) - 2:
                modules.append(model_utils.get_activation_func(activation))
                modules.append(torch.nn.LayerNorm(self.layer_dims[i + 1]))
            else:
                modules.append(model_utils.get_activation_func("identity"))


        self.logstd = torch.nn.Parameter(
            torch.ones(action_dim, dtype=torch.float32) * logstd_init
        )

        self.action_dim = action_dim
        self.obs_dim = obs_dim

    def get_logstd(self):
        return self.logstd

    def forward(self, obs, deterministic=False):
        mu = self.mu_net(obs)

        if deterministic:
            return mu
        else:
            std = self.logstd.exp()
            dist = Normal(mu, std)
            sample = dist.rsample()
            return sample

    def forward_with_dist(self, obs, deterministic=False):
        mu = self.mu_net(obs)
        std = self.logstd.exp()

        if deterministic:
            return mu, mu, std
        else:
            dist = Normal(mu, std)
            sample = dist.rsample()
            return sample, mu, std

    def evaluate_actions_log_probs(self, obs, actions):
        mu = self.mu_net(obs)

        std = self.logstd.exp()
        dist = Normal(mu, std)

        return dist.log_prob(actions)
