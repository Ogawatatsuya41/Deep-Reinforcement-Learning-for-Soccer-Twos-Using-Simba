import torch
import torch.nn as nn
import torch.nn.functional as F
from mlagents.torch_utils import default_device
from mlagents.trainers.torch_entities.networks import NetworkBody
from mlagents.trainers.settings import NetworkSettings

class SimBaBaseNetwork(nn.Module):
    """
    Base network for SimBa architecture.
    Used for both the Actor (policy) and Critic (value function).
    """

    def __init__(self, input_size: int, output_size: int, hidden_units: int = 64, num_layers: int = 2):
        super(SimBaBaseNetwork, self).__init__()
        self.hidden_units = hidden_units
        self.num_layers = num_layers

        # Fully connected layers
        layers = []
        for i in range(num_layers):
            layers.append(nn.Linear(input_size if i == 0 else hidden_units, hidden_units))
            layers.append(nn.ReLU())
        self.hidden_layers = nn.Sequential(*layers)

        # Output layer
        self.output_layer = nn.Linear(hidden_units, output_size)

    def forward(self, x):
        x = self.hidden_layers(x)
        x = self.output_layer(x)
        return x


class SimBaActor(nn.Module):
    """
    SimBa-based Actor for policy.
    Outputs action logits for a discrete action space or action mean/sigma for continuous.
    """

    def __init__(self, input_size: int, action_size: int, network_settings: NetworkSettings):
        super(SimBaActor, self).__init__()
        self.body = SimBaBaseNetwork(
            input_size=input_size,
            output_size=action_size,
            hidden_units=network_settings.hidden_units,
            num_layers=network_settings.num_layers,
        )

    def forward(self, obs, masks=None):
        """
        Forward pass through the actor network.
        :param obs: Observations from the environment.
        :param masks: Action masks (optional, for discrete actions).
        :return: Action logits (discrete) or action mean/sigma (continuous).
        """
        logits = self.body(obs)
        if masks is not None:
            logits += masks  # Apply action masks if provided
        return logits

    def get_action_logits(self, obs, masks=None):
        """
        Get action logits for policy loss calculation.
        """
        logits = self.forward(obs, masks)
        entropy = -(torch.softmax(logits, dim=-1) * torch.log_softmax(logits, dim=-1)).sum(dim=-1)
        return logits, entropy


class SimBaCritic(nn.Module):
    """
    SimBa-based Critic for value estimation.
    Estimates the value of a state for one or more reward signals.
    """

    def __init__(self, stream_names, observation_specs, network_settings: NetworkSettings):
        super(SimBaCritic, self).__init__()
        # Create a SimBa network for each reward signal
        self.streams = nn.ModuleDict()
        input_size = sum(obs.shape[0] for obs in observation_specs)

        for name in stream_names:
            self.streams[name] = SimBaBaseNetwork(
                input_size=input_size,
                output_size=1,
                hidden_units=network_settings.hidden_units,
                num_layers=network_settings.num_layers,
            )

    def critic_pass(self, obs, memories=None, sequence_length=None):
        """
        Forward pass through the critic network.
        :param obs: Observations from the environment.
        :param memories: Optional recurrent memories (not used here).
        :param sequence_length: Optional sequence length (not used here).
        :return: Values for each reward signal.
        """
        values = {}
        x = torch.cat(obs, dim=-1)  # Flatten observation inputs
        for name, stream in self.streams.items():
            values[name] = stream(x).squeeze(-1)
        return values, None
