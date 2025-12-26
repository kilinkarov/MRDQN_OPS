"""Rainbow DQN agent for cryptocurrency trading."""

from typing import Dict, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

from crypto_dqn_ops.models.rainbow_network import Network
from crypto_dqn_ops.utils.replay_buffer import PrioritizedReplayBuffer, ReplayBuffer


class RainbowAgent:
    """Rainbow DQN Agent interacting with environment.

    Attributes:
        env: Gym environment
        memory: Prioritized replay buffer
        batch_size: Batch size for sampling
        target_update: Period for target model's hard update
        gamma: Discount factor
        dqn: Model to train and select actions
        dqn_target: Target model to update
        optimizer: Optimizer for training dqn
        v_min: Min value of support
        v_max: Max value of support
        atom_size: The unit number of support
        support: Support for categorical dqn
        use_n_step: Whether to use n_step memory
        n_step: Step number to calculate n-step td error
        memory_n: N-step replay buffer
    """

    def __init__(
        self,
        env: gym.Env,
        memory_size: int,
        batch_size: int,
        target_update: int,
        gamma: float,
        alpha: float,
        beta: float,
        prior_eps: float,
        v_min: float,
        v_max: float,
        atom_size: int,
        n_step: int,
        lr: float,
    ):
        """Initialize Rainbow DQN agent."""
        obs_dim = env.observation_space.shape[1]
        action_dim = env.action_space.n

        self.env = env
        self.batch_size = batch_size
        self.target_update = target_update
        self.gamma = gamma

        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.beta = beta
        self.prior_eps = prior_eps
        self.memory = PrioritizedReplayBuffer(
            obs_dim, memory_size, batch_size, alpha, n_step, gamma
        )

        self.use_n_step = n_step > 1
        if self.use_n_step:
            self.n_step = n_step
            self.memory_n = ReplayBuffer(obs_dim, memory_size, batch_size, n_step, gamma)

        self.v_min = v_min
        self.v_max = v_max
        self.atom_size = atom_size
        self.support = torch.linspace(self.v_min, self.v_max, self.atom_size).to(self.device)

        self.dqn = Network(obs_dim, action_dim, self.atom_size, self.support).to(self.device)
        self.dqn_target = Network(obs_dim, action_dim, self.atom_size, self.support).to(self.device)
        self.dqn_target.load_state_dict(self.dqn.state_dict())
        self.dqn_target.eval()

        self.optimizer = optim.Adam(self.dqn.parameters(), lr=lr, eps=1.5e-4)

        self.transition = list()
        self.is_test = False

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select an action from the input state."""
        selected_action = self.dqn(torch.FloatTensor(state).to(self.device)).argmax()
        selected_action = selected_action.detach().cpu().numpy()

        if not self.is_test:
            self.transition = [state, selected_action]

        return selected_action

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool]:
        """Take an action and return the response of the env."""
        next_state, reward, done, _ = self.env.step(action)

        if not self.is_test:
            self.transition += [reward, next_state, done]

            if self.use_n_step:
                one_step_transition = self.memory_n.store(*self.transition)
            else:
                one_step_transition = self.transition

            if one_step_transition:
                self.memory.store(*one_step_transition)

        return next_state, reward, done

    def update_model(self) -> float:
        """Update the model by gradient descent."""
        samples = self.memory.sample_batch(self.beta)
        weights = torch.FloatTensor(samples["weights"].reshape(-1, 1)).to(self.device)
        indices = samples["indices"]

        elementwise_loss = self._compute_dqn_loss(samples, self.gamma)
        loss = torch.mean(elementwise_loss * weights)

        if self.use_n_step:
            gamma = self.gamma**self.n_step
            samples = self.memory_n.sample_batch_from_idxs(indices)
            elementwise_loss_n_loss = self._compute_dqn_loss(samples, gamma)
            elementwise_loss += elementwise_loss_n_loss
            loss = torch.mean(elementwise_loss * weights)

        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.dqn.parameters(), 10.0)
        self.optimizer.step()

        loss_for_prior = elementwise_loss.detach().cpu().numpy()
        new_priorities = loss_for_prior + self.prior_eps
        self.memory.update_priorities(indices, new_priorities)

        self.dqn.reset_noise()
        self.dqn_target.reset_noise()

        return loss.item()

    def _compute_dqn_loss(self, samples: Dict[str, np.ndarray], gamma: float) -> torch.Tensor:
        """Return categorical dqn loss."""
        device = self.device
        state = torch.FloatTensor(samples["obs"]).to(device)
        next_state = torch.FloatTensor(samples["next_obs"]).to(device)
        action = torch.LongTensor(samples["acts"]).to(device)
        reward = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(device)
        done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)

        delta_z = float(self.v_max - self.v_min) / (self.atom_size - 1)

        with torch.no_grad():
            next_action = self.dqn(next_state).argmax(1)
            next_dist = self.dqn_target.dist(next_state)
            next_dist = next_dist[range(self.batch_size), next_action].detach().clone()

            t_z = reward + (1 - done) * gamma * self.support
            t_z = t_z.clamp(min=self.v_min, max=self.v_max)
            bin_position = (t_z - self.v_min) / delta_z
            lower_bin = bin_position.floor().long()
            upper_bin = bin_position.ceil().long()

            offset = (
                torch.linspace(0, (self.batch_size - 1) * self.atom_size, self.batch_size)
                .long()
                .unsqueeze(1)
                .expand(self.batch_size, self.atom_size)
                .to(self.device)
            )

            proj_dist = torch.zeros(next_dist.size(), device=self.device).view(-1)

            lower_idx = (lower_bin + offset).view(-1)
            lower_val = (next_dist * (upper_bin.float() - bin_position)).view(-1)
            upper_idx = (upper_bin + offset).view(-1)
            upper_val = (next_dist * (bin_position - lower_bin.float())).view(-1)

            indices = torch.cat([lower_idx, upper_idx])
            values = torch.cat([lower_val, upper_val])

            proj_dist = proj_dist.scatter_add(0, indices, values)
            proj_dist = proj_dist.view(next_dist.size())
        dist = self.dqn.dist(state)
        log_p = torch.log(dist[range(self.batch_size), action])
        elementwise_loss = -(proj_dist * log_p).sum(1)

        return elementwise_loss

    def _target_hard_update(self):
        """Hard update: target <- local."""
        self.dqn_target.load_state_dict(self.dqn.state_dict())
