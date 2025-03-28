# import torch
# import torch.nn as nn

# class RSNorm(nn.Module):
#     def __init__(self, hidden_size):
#         super().__init__()
#         self.scale = nn.Parameter(torch.ones(hidden_size))
#         self.shift = nn.Parameter(torch.zeros(hidden_size))

#     def forward(self, x):
#         mean = x.mean(dim=-1, keepdim=True)
#         std = x.std(dim=-1, keepdim=True)
#         return self.scale * (x - mean) / (std + 1e-6) + self.shift

# class ResidualFeedforwardBlock(nn.Module):
#     def __init__(self, input_size, hidden_size):
#         super().__init__()
#         self.fc1 = nn.Linear(input_size, hidden_size)
#         self.fc2 = nn.Linear(hidden_size, input_size)
#         self.activation = nn.ReLU()
#         self.norm = RSNorm(input_size)

#     def forward(self, x):
#         residual = x
#         x = self.activation(self.fc1(x))
#         x = self.fc2(x)
#         return self.norm(x + residual)

# # class SimBaNetwork(nn.Module):
# #     def __init__(self, input_size, output_size, hidden_size, use_rs_norm=True, use_residual_blocks=True):
# #         super().__init__()
# #         self.input_layer = nn.Linear(input_size, hidden_size)
# #         self.residual_block = (
# #             ResidualFeedforwardBlock(hidden_size, hidden_size) if use_residual_blocks else nn.Identity()
# #         )
# #         self.output_layer = nn.Linear(hidden_size, output_size)
# #         self.use_rs_norm = use_rs_norm
# #         if use_rs_norm:
# #             self.norm = RSNorm(hidden_size)

# #     def forward(self, x):
# #         x = self.input_layer(x)
# #         if self.use_rs_norm:
# #             x = self.norm(x)
# #         x = self.residual_block(x)
# #         return self.output_layer(x)
# class SimBaActor(nn.Module):
#     def __init__(self, input_size, action_size, hidden_size):
#         super().__init__()
#         self.input_layer = nn.Linear(input_size, hidden_size)
#         self.residual_block = ResidualFeedforwardBlock(hidden_size, hidden_size)
#         self.output_layer = nn.Linear(hidden_size, action_size)

#     def forward(self, x):
#         x = self.input_layer(x)
#         x = self.residual_block(x)
#         return self.output_layer(x)


import torch
import torch.nn as nn
from typing import List, Optional, Dict, Any, Tuple, Union
from mlagents.trainers.torch_entities.model import (
    ActionSpec,
    ObservationSpec,
    NetworkBody,
    ActionModel,
    AgentAction,
)
from mlagents.trainers.settings import NetworkSettings
from mlagents.trainers.torch_entities.utils import ModelUtils

class RunningStatsNorm(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.register_buffer('mean', torch.zeros(num_features))
        self.register_buffer('var', torch.ones(num_features))
        self.count = 0

    def update(self, x: torch.Tensor):
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        batch_count = x.size(0)

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * self.count * batch_count / total_count
        new_var = M2 / total_count

        self.mean.copy_(new_mean)
        self.var.copy_(new_var)
        self.count += batch_count

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            self.update(x)
        return (x - self.mean) / torch.sqrt(self.var + self.eps)

class SimBaResidualBlock(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.linear1 = nn.Linear(hidden_size, 4 * hidden_size)
        self.linear2 = nn.Linear(4 * hidden_size, hidden_size)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.layer_norm(x)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return residual + x

class SimBaActor(nn.Module, Actor):
    MODEL_EXPORT_VERSION = 3

    def __init__(
        self,
        observation_specs: List[ObservationSpec],
        network_settings: NetworkSettings,
        action_spec: ActionSpec,
        conditional_sigma: bool = False,
        tanh_squash: bool = False,
        hidden_size: int = 512,  # 隠れ層のサイズを指定
        num_blocks: int = 3,     # 残差ブロックの数を指定
    ):
        super().__init__()
        self.action_spec = action_spec
        self.version_number = torch.nn.Parameter(
            torch.Tensor([self.MODEL_EXPORT_VERSION]), requires_grad=False
        )
        self.is_continuous_int_deprecated = torch.nn.Parameter(
            torch.Tensor([int(self.action_spec.is_continuous())]), requires_grad=False
        )
        self.continuous_act_size_vector = torch.nn.Parameter(
            torch.Tensor([int(self.action_spec.continuous_size)]), requires_grad=False
        )
        self.discrete_act_size_vector = torch.nn.Parameter(
            torch.Tensor([self.action_spec.discrete_branches]), requires_grad=False
        )
        self.act_size_vector_deprecated = torch.nn.Parameter(
            torch.Tensor(
                [
                    self.action_spec.continuous_size
                    + sum(self.action_spec.discrete_branches)
                ]
            ),
            requires_grad=False,
        )

        # 観測正規化層
        self.obs_norm = RunningStatsNorm(sum(spec.shape[0] for spec in observation_specs))

        # ネットワークボディの変更
        self.network_body = self._build_network_body(network_settings, hidden_size, num_blocks)

        if network_settings.memory is not None:
            self.encoding_size = network_settings.memory.memory_size // 2
        else:
            self.encoding_size = hidden_size
        self.memory_size_vector = torch.nn.Parameter(
            torch.Tensor([int(self.network_body.memory_size)]), requires_grad=False
        )

        self.action_model = ActionModel(
            self.encoding_size,
            action_spec,
            conditional_sigma=conditional_sigma,
            tanh_squash=tanh_squash,
            deterministic=network_settings.deterministic,
        )

    def _build_network_body(self, settings: NetworkSettings, hidden_size: int, num_blocks: int) -> nn.Module:
        # 新しいネットワーク構造
        layers = []
        input_size = sum(spec.shape[0] for spec in self.observation_specs)
        
        layers.append(nn.Linear(input_size, hidden_size))
        for _ in range(num_blocks):
            layers.append(SimBaResidualBlock(hidden_size))
        layers.append(nn.LayerNorm(hidden_size))
        
        return nn.Sequential(*layers)

    @property
    def memory_size(self) -> int:
        return self.network_body.memory_size

    def update_normalization(self, buffer: AgentBuffer) -> None:
        # 観測値の正規化を更新
        obs = torch.cat([torch.tensor(buffer[ObsUtil.get_name_at(i)]) for i in range(len(self.observation_specs))], dim=-1)
        self.obs_norm.update(obs)

    def get_action_and_stats(
        self,
        inputs: List[torch.Tensor],
        masks: Optional[torch.Tensor] = None,
        memories: Optional[torch.Tensor] = None,
        sequence_length: int = 1,
    ) -> Tuple[AgentAction, Dict[str, Any], torch.Tensor]:
        # 観測値の正規化
        normalized_inputs = self.obs_norm(torch.cat(inputs, dim=-1))
        
        # ネットワーク処理
        encoding = self.network_body(normalized_inputs)
        
        action, log_probs, entropies = self.action_model(encoding, masks)
        run_out = {}
        run_out["env_action"] = action.to_action_tuple(
            clip=self.action_model.clip_action
        )
        run_out["log_probs"] = log_probs
        run_out["entropy"] = entropies

        return action, run_out, memories

    def get_stats(
        self,
        inputs: List[torch.Tensor],
        actions: AgentAction,
        masks: Optional[torch.Tensor] = None,
        memories: Optional[torch.Tensor] = None,
        sequence_length: int = 1,
    ) -> Dict[str, Any]:
        normalized_inputs = self.obs_norm(torch.cat(inputs, dim=-1))
        encoding = self.network_body(normalized_inputs)

        log_probs, entropies = self.action_model.evaluate(encoding, masks, actions)
        run_out = {}
        run_out["log_probs"] = log_probs
        run_out["entropy"] = entropies
        return run_out

    def forward(
        self,
        inputs: List[torch.Tensor],
        masks: Optional[torch.Tensor] = None,
        memories: Optional[torch.Tensor] = None,
    ) -> Tuple[Union[int, torch.Tensor], ...]:
        normalized_inputs = self.obs_norm(torch.cat(inputs, dim=-1))
        encoding, memories_out = self.network_body(normalized_inputs, memories=memories, sequence_length=1)

        (
            cont_action_out,
            disc_action_out,
            action_out_deprecated,
            deterministic_cont_action_out,
            deterministic_disc_action_out,
        ) = self.action_model.get_action_out(encoding, masks)
        export_out = [self.version_number, self.memory_size_vector]
        if self.action_spec.continuous_size > 0:
            export_out += [
                cont_action_out,
                self.continuous_act_size_vector,
                deterministic_cont_action_out,
            ]
        if self.action_spec.discrete_size > 0:
            export_out += [
                disc_action_out,
                self.discrete_act_size_vector,
                deterministic_disc_action_out,
            ]
        if self.network_body.memory_size > 0:
            export_out += [memories_out]
        return tuple(export_out)