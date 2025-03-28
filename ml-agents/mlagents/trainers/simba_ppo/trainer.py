from typing import Type, Union, Dict, Any
import numpy as np
import torch
import torch.nn as nn

from mlagents.trainers.buffer import BufferKey, RewardSignalUtil
from mlagents.trainers.trainer.on_policy_trainer import OnPolicyTrainer
from mlagents.trainers.policy.policy import Policy
from mlagents.trainers.trainer.trainer_utils import get_gae
from mlagents.trainers.policy.torch_policy import TorchPolicy
from mlagents.trainers.trajectory import Trajectory
from mlagents.trainers.behavior_id_utils import BehaviorIdentifiers
from mlagents.trainers.settings import TrainerSettings
from mlagents.trainers.torch_entities.networks import SimpleActor

# SimBa アーキテクチャ
class SimBaNetwork(nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden_size: int = 64, num_layers: int = 2):
        super(SimBaNetwork, self).__init__()
        layers = []
        for i in range(num_layers):
            layers.append(nn.Linear(input_size if i == 0 else hidden_size, hidden_size))
            layers.append(nn.ReLU())
        self.hidden_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.hidden_layers(x)
        return self.output_layer(x)

# SimBa-PPO トレーナー
class SimBaPPOTrainer(OnPolicyTrainer):
    def __init__(
        self,
        behavior_name: str,
        reward_buff_cap: int,
        trainer_settings: TrainerSettings,
        training: bool,
        load: bool,
        seed: int,
        artifact_path: str,
    ):
        super().__init__(
            behavior_name,
            reward_buff_cap,
            trainer_settings,
            training,
            load,
            seed,
            artifact_path,
        )
        self.hyperparameters = self.trainer_settings.hyperparameters
        self.policy: TorchPolicy = None  # type: ignore

    def create_policy(self, parsed_behavior_id: BehaviorIdentifiers, behavior_spec):
        """
        Creates the policy using SimBa's network architecture.
        """
        actor_cls: Union[Type[SimpleActor], Type[SimBaNetwork]] = SimBaNetwork
        actor_kwargs: Dict[str, Any] = {
            "input_size": sum(spec.shape[0] for spec in behavior_spec.observation_specs),
            "output_size": behavior_spec.action_spec.continuous_size
            or behavior_spec.action_spec.discrete_size,
            "hidden_size": self.trainer_settings.network_settings.hidden_units,
            "num_layers": self.trainer_settings.network_settings.num_layers,
        }

        return TorchPolicy(
            self.seed,
            behavior_spec,
            self.trainer_settings.network_settings,
            actor_cls,
            actor_kwargs,
        )

    def _process_trajectory(self, trajectory: Trajectory):
        """
        Takes a trajectory and processes it, putting it into the update buffer.
        """
        super()._process_trajectory(trajectory)
        agent_id = trajectory.agent_id
        agent_buffer_trajectory = trajectory.to_agentbuffer()

        # Value estimates and advantages
        (
            value_estimates,
            value_next,
            _,
        ) = self.optimizer.get_trajectory_value_estimates(
            agent_buffer_trajectory,
            trajectory.next_obs,
            trajectory.done_reached and not trajectory.interrupted,
        )

        for name, v in value_estimates.items():
            agent_buffer_trajectory[RewardSignalUtil.value_estimates_key(name)].extend(
                v
            )
            self._stats_reporter.add_stat(
                f"Policy/{self.optimizer.reward_signals[name].name.capitalize()} Value Estimate",
                np.mean(v),
            )

        # Compute GAE and returns
        advantages = []
        returns = []
        for name in self.optimizer.reward_signals:
            bootstrap_value = value_next[name]
            local_rewards = agent_buffer_trajectory[
                RewardSignalUtil.rewards_key(name)
            ].get_batch()
            local_value_estimates = agent_buffer_trajectory[
                RewardSignalUtil.value_estimates_key(name)
            ].get_batch()

            local_advantage = get_gae(
                rewards=local_rewards,
                value_estimates=local_value_estimates,
                value_next=bootstrap_value,
                gamma=self.optimizer.reward_signals[name].gamma,
                lambd=self.hyperparameters.lambd,
            )
            local_return = local_advantage + local_value_estimates
            agent_buffer_trajectory[RewardSignalUtil.returns_key(name)].set(
                local_return
            )
            agent_buffer_trajectory[RewardSignalUtil.advantage_key(name)].set(
                local_advantage
            )
            advantages.append(local_advantage)
            returns.append(local_return)

        # Global advantages
        global_advantages = list(np.mean(np.array(advantages, dtype=np.float32), axis=0))
        global_returns = list(np.mean(np.array(returns, dtype=np.float32), axis=0))
        agent_buffer_trajectory[BufferKey.ADVANTAGES].set(global_advantages)
        agent_buffer_trajectory[BufferKey.DISCOUNTED_RETURNS].set(global_returns)

        self._append_to_update_buffer(agent_buffer_trajectory)

    @staticmethod
    def get_trainer_name() -> str:
        return "simba_ppo"
