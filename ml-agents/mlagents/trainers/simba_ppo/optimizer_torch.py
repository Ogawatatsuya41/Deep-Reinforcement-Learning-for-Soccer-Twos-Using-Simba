from typing import Dict, cast
import attr

from mlagents.torch_utils import torch, default_device
from mlagents.trainers.buffer import AgentBuffer, BufferKey, RewardSignalUtil
from mlagents_envs.timers import timed
from mlagents.trainers.policy.torch_policy import TorchPolicy
from mlagents.trainers.optimizer.torch_optimizer import TorchOptimizer
from mlagents.trainers.settings import (
    TrainerSettings,
    OnPolicyHyperparamSettings,
    ScheduleType,
)
from mlagents.trainers.torch_entities.utils import ModelUtils
from mlagents.trainers.trajectory import ObsUtil


@attr.s(auto_attribs=True)
class SimBaPPOSettings(OnPolicyHyperparamSettings):
    beta: float = 5.0e-3
    epsilon: float = 0.2
    lambd: float = 0.95
    num_epoch: int = 3
    shared_critic: bool = False
    learning_rate_schedule: ScheduleType = ScheduleType.LINEAR
    beta_schedule: ScheduleType = ScheduleType.LINEAR
    epsilon_schedule: ScheduleType = ScheduleType.LINEAR


class TorchSimBaPPOOptimizer(TorchOptimizer):
    def __init__(self, policy: TorchPolicy, trainer_settings: TrainerSettings):
        """
        Optimizer for SimBa PPO, extending the base TorchOptimizer.
        Handles value estimation, policy updates, and loss calculation.
        :param policy: A TorchPolicy object to be optimized.
        :param trainer_settings: Trainer settings with SimBaPPO parameters.
        """
        super().__init__(policy, trainer_settings)
        reward_signal_configs = trainer_settings.reward_signals
        reward_signal_names = [key.value for key, _ in reward_signal_configs.items()]

        self.hyperparameters: SimBaPPOSettings = cast(
            SimBaPPOSettings, trainer_settings.hyperparameters
        )

        # Parameters for the actor and critic
        params = list(self.policy.actor.parameters())
        if self.hyperparameters.shared_critic:
            self._critic = policy.actor
        else:
            from simba_ppo.network import SimBaCritic  # Import SimBa-specific critic
            self._critic = SimBaCritic(
                reward_signal_names,
                policy.behavior_spec.observation_specs,
                network_settings=trainer_settings.network_settings,
            )
            self._critic.to(default_device())
            params += list(self._critic.parameters())

        # Decay schedules for learning rate, epsilon, and beta
        self.decay_learning_rate = ModelUtils.DecayedValue(
            self.hyperparameters.learning_rate_schedule,
            self.hyperparameters.learning_rate,
            1e-10,
            self.trainer_settings.max_steps,
        )
        self.decay_epsilon = ModelUtils.DecayedValue(
            self.hyperparameters.epsilon_schedule,
            self.hyperparameters.epsilon,
            0.1,
            self.trainer_settings.max_steps,
        )
        self.decay_beta = ModelUtils.DecayedValue(
            self.hyperparameters.beta_schedule,
            self.hyperparameters.beta,
            1e-5,
            self.trainer_settings.max_steps,
        )

        # Optimizer
        self.optimizer = torch.optim.Adam(
            params, lr=self.trainer_settings.hyperparameters.learning_rate
        )

    @property
    def critic(self):
        return self._critic

    @timed
    def update(self, batch: AgentBuffer, num_sequences: int) -> Dict[str, float]:
        """
        Performs the update step for the SimBa PPO optimizer.
        :param batch: Batch of experiences.
        :param num_sequences: Number of sequences to process.
        :return: A dictionary with update statistics.
        """
        # Decay parameters
        decay_lr = self.decay_learning_rate.get_value(self.policy.get_current_step())
        decay_eps = self.decay_epsilon.get_value(self.policy.get_current_step())
        decay_bet = self.decay_beta.get_value(self.policy.get_current_step())

        returns = {}
        old_values = {}
        for name in self.reward_signals:
            old_values[name] = ModelUtils.list_to_tensor(
                batch[RewardSignalUtil.value_estimates_key(name)]
            )
            returns[name] = ModelUtils.list_to_tensor(
                batch[RewardSignalUtil.returns_key(name)]
            )

        # Prepare observation tensors
        n_obs = len(self.policy.behavior_spec.observation_specs)
        current_obs = ObsUtil.from_buffer(batch, n_obs)
        current_obs = [ModelUtils.list_to_tensor(obs) for obs in current_obs]

        act_masks = ModelUtils.list_to_tensor(batch[BufferKey.ACTION_MASK])
        actions = batch[BufferKey.ACTIONS]

        # Forward pass through actor and critic
        logits, entropy = self.policy.actor.get_action_logits(current_obs, act_masks)
        values, _ = self.critic.critic_pass(current_obs)

        # Loss calculation
        advantages = ModelUtils.list_to_tensor(batch[BufferKey.ADVANTAGES])
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        old_log_probs = ModelUtils.list_to_tensor(batch[BufferKey.LOG_PROBS])

        # Policy loss with SimBa regularization
        policy_loss = -(log_probs * advantages).mean()
        value_loss = torch.nn.functional.mse_loss(values, old_values["extrinsic"])
        entropy_loss = -decay_bet * entropy.mean()

        # Total loss
        loss = policy_loss + 0.5 * value_loss + entropy_loss

        # Optimization step
        ModelUtils.update_learning_rate(self.optimizer, decay_lr)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update stats
        update_stats = {
            "Losses/Policy Loss": policy_loss.item(),
            "Losses/Value Loss": value_loss.item(),
            "Policy/Entropy": entropy_loss.item(),
            "Policy/Learning Rate": decay_lr,
            "Policy/Epsilon": decay_eps,
            "Policy/Beta": decay_bet,
        }

        return update_stats

    def get_modules(self):
        modules = {
            "Optimizer:actor_optimizer": self.optimizer,
            "Optimizer:critic": self._critic,
        }
        for reward_provider in self.reward_signals.values():
            modules.update(reward_provider.get_modules())
        return modules
