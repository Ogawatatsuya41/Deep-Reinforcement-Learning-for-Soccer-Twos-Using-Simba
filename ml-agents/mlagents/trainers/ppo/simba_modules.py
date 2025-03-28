import torch
import torch.nn as nn

class RSNorm(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(hidden_size))
        self.shift = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.scale * (x - mean) / (std + 1e-6) + self.shift

class ResidualFeedforwardBlock(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, input_size)
        self.activation = nn.ReLU()
        self.norm = RSNorm(input_size)

    def forward(self, x):
        residual = x
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return self.norm(x + residual)

class SimBaActor(nn.Module):
    def __init__(self, input_size, action_size, hidden_size):
        super().__init__()
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.residual_block = ResidualFeedforwardBlock(hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.residual_block(x)
        return self.output_layer(x)
