"""PyTorch Lightning module for Rainbow DQN training."""

from typing import Dict

import lightning as L
import numpy as np
import torch
import torch.optim as optim

from crypto_dqn_ops.models.rainbow_network import Network
from crypto_dqn_ops.utils.replay_buffer import PrioritizedReplayBuffer, ReplayBuffer


class RainbowLightningModule(L.LightningModule):
    """PyTorch Lightning module for Rainbow DQN."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        memory_size: int,
        batch_size: int,
        target_update: int,
        gamma: float,
        lr: float,
        alpha: float,
        beta: float,
        prior_eps: float,
        v_min: float,
        v_max: float,
        atom_size: int,
        n_step: int,
    ):
        """Initialize Rainbow DQN Lightning module.

        Args:
            obs_dim: Observation dimension
            action_dim: Action dimension
            memory_size: Replay buffer size
            batch_size: Batch size for training
            target_update: Target network update frequency
            gamma: Discount factor
            lr: Learning rate
            alpha: PER alpha parameter
            beta: PER beta parameter
            prior_eps: PER epsilon
            v_min: Categorical DQN min value
            v_max: Categorical DQN max value
            atom_size: Number of atoms for categorical DQN
            n_step: N-step learning parameter
        """
        super().__init__()
        self.save_hyperparameters()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.target_update = target_update
        self.gamma = gamma
        self.lr = lr
        self.beta = beta
        self.prior_eps = prior_eps
        self.v_min = v_min
        self.v_max = v_max
        self.atom_size = atom_size
        self.n_step = n_step

        self.support = torch.linspace(self.v_min, self.v_max, self.atom_size)
        self.register_buffer("_support", self.support)

        self.dqn = Network(obs_dim, action_dim, self.atom_size, self._support)
        self.dqn_target = Network(obs_dim, action_dim, self.atom_size, self._support)
        self.dqn_target.load_state_dict(self.dqn.state_dict())
        self.dqn_target.eval()

        self.memory = PrioritizedReplayBuffer(
            obs_dim, memory_size, batch_size, alpha, n_step, gamma
        )

        self.use_n_step = n_step > 1
        if self.use_n_step:
            self.memory_n = ReplayBuffer(obs_dim, memory_size, batch_size, n_step, gamma)

        self.update_cnt = 0
        self.beta_start = beta
        self.episode_scores = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.dqn(x)

    def configure_optimizers(self):
        """Configure optimizer."""
        optimizer = optim.Adam(self.dqn.parameters(), lr=self.lr, eps=1.5e-4)
        return optimizer

    def training_step(self, batch, batch_idx):
        """Training step - Lightning вызывает это для каждого batch."""
        if len(self.memory) < self.batch_size:
            return None

        samples = self.memory.sample_batch(self.beta)
        weights = torch.FloatTensor(samples["weights"].reshape(-1, 1)).to(self.device)
        indices = samples["indices"]

        elementwise_loss = self._compute_dqn_loss(samples, self.gamma)
        loss = torch.mean(elementwise_loss * weights)

        if self.use_n_step:
            gamma = self.gamma**self.n_step
            samples_n = self.memory_n.sample_batch_from_idxs(indices)
            elementwise_loss_n = self._compute_dqn_loss(samples_n, gamma)
            elementwise_loss += elementwise_loss_n
            loss = torch.mean(elementwise_loss * weights)

        loss_for_prior = elementwise_loss.detach().cpu().numpy()
        new_priorities = loss_for_prior + self.prior_eps
        self.memory.update_priorities(indices, new_priorities)

        self.dqn.reset_noise()
        self.dqn_target.reset_noise()

        self.update_cnt += 1
        if self.update_cnt % self.target_update == 0:
            self._target_hard_update()

        self.log("train/loss", loss, prog_bar=True)
        self.log("train/beta", self.beta, prog_bar=False)

        return loss

    def _compute_dqn_loss(self, samples: Dict[str, np.ndarray], gamma: float) -> torch.Tensor:
        """Return categorical dqn loss."""
        state = torch.FloatTensor(samples["obs"]).to(self.device)
        next_state = torch.FloatTensor(samples["next_obs"]).to(self.device)
        action = torch.LongTensor(samples["acts"]).to(self.device)
        reward = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(self.device)
        done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(self.device)

        delta_z = float(self.v_max - self.v_min) / (self.atom_size - 1)

        with torch.no_grad():
            next_action = self.dqn(next_state).argmax(1)
            next_dist = self.dqn_target.dist(next_state)
            next_dist = next_dist[range(self.batch_size), next_action].detach().clone()

            t_z = reward + (1 - done) * gamma * self.support.to(self.device)
            t_z = t_z.clamp(min=self.v_min, max=self.v_max)
            bin_position = (t_z - self.v_min) / delta_z
            lower_bin = bin_position.floor().long()
            upper_bin = bin_position.ceil().long()

            proj_dist_np = np.zeros((self.batch_size, self.atom_size), dtype=np.float32)

            lower_bin_np = lower_bin.cpu().numpy()
            upper_bin_np = upper_bin.cpu().numpy()
            bin_position_np = bin_position.cpu().numpy()
            next_dist_np = next_dist.cpu().numpy()

            for batch_idx in range(self.batch_size):
                for atom_idx in range(self.atom_size):
                    l_idx = int(lower_bin_np[batch_idx, atom_idx])
                    u_idx = int(upper_bin_np[batch_idx, atom_idx])

                    l_weight = (
                        upper_bin_np[batch_idx, atom_idx] - bin_position_np[batch_idx, atom_idx]
                    )
                    u_weight = (
                        bin_position_np[batch_idx, atom_idx] - lower_bin_np[batch_idx, atom_idx]
                    )
                    dist_val = next_dist_np[batch_idx, atom_idx]

                    proj_dist_np[batch_idx, l_idx] += dist_val * l_weight
                    proj_dist_np[batch_idx, u_idx] += dist_val * u_weight

            proj_dist = torch.from_numpy(proj_dist_np).to(self.device)

        dist = self.dqn.dist(state)
        log_p = torch.log(dist[range(self.batch_size), action])
        elementwise_loss = -(proj_dist * log_p).sum(1)

        return elementwise_loss

    def _target_hard_update(self):
        """Hard update: target <- local."""
        self.dqn_target.load_state_dict(self.dqn.state_dict())
