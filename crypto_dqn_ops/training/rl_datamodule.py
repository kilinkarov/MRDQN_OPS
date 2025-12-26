"""RL DataModule for collecting experience from environment."""

import lightning as L
import torch
from torch.utils.data import DataLoader, IterableDataset


class RLExperienceDataset(IterableDataset):
    """Iterable dataset that generates experience from RL environment."""

    def __init__(self, env, agent, num_frames: int):
        """Initialize RL dataset.

        Args:
            env: Gym environment
            agent: RL agent (lightning module)
            num_frames: Total number of frames to collect
        """
        self.env = env
        self.agent = agent
        self.num_frames = num_frames
        self.frame_idx = 0

    def __iter__(self):
        """Generate experience from environment."""
        state, _ = self.env.reset()
        score = 0

        while self.frame_idx < self.num_frames:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.agent.device)
                action = self.agent(state_tensor).argmax().item()

            next_state, reward, done, _ = self.env.step(action)

            if self.agent.use_n_step:
                one_step = self.agent.memory_n.store(state, action, reward, next_state, done)
            else:
                one_step = (state, action, reward, next_state, done)

            if one_step:
                self.agent.memory.store(*one_step)

            state = next_state
            score += reward
            self.frame_idx += 1

            fraction = min(self.frame_idx / self.num_frames, 1.0)
            self.agent.beta = self.agent.beta_start + fraction * (1.0 - self.agent.beta_start)

            if done:
                state, _ = self.env.reset()
                if hasattr(self.agent, "episode_scores"):
                    self.agent.episode_scores.append(score)
                score = 0

            yield {}


class RLDataModule(L.LightningDataModule):
    """Lightning DataModule for RL training."""

    def __init__(self, env, agent, num_frames: int, batch_size: int = 128):
        """Initialize RL DataModule.

        Args:
            env: Gym environment
            agent: RL agent (lightning module)
            num_frames: Total training frames
            batch_size: Batch size
        """
        super().__init__()
        self.env = env
        self.agent = agent
        self.num_frames = num_frames
        self.batch_size = batch_size

    def train_dataloader(self):
        """Create training dataloader."""
        dataset = RLExperienceDataset(self.env, self.agent, self.num_frames)
        return DataLoader(dataset, batch_size=None, num_workers=0)

    def teardown(self, stage=None):
        """Clean up."""
        if hasattr(self, "env"):
            self.env.close()
