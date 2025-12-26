"""Inference module for trained models."""

from pathlib import Path
from typing import Union

import numpy as np
import torch

from crypto_dqn_ops.models.rainbow_network import Network


class CryptoPredictor:
    """Predictor for cryptocurrency trading decisions."""

    def __init__(
        self,
        model_path: Union[str, Path],
        obs_dim: int,
        action_dim: int = 2,
        v_min: float = 0.0,
        v_max: float = 20.0,
        atom_size: int = 51,
    ):
        """Initialize predictor.

        Args:
            model_path: Path to trained model
            obs_dim: Observation dimension
            action_dim: Action dimension
            v_min: Categorical DQN min value
            v_max: Categorical DQN max value
            atom_size: Number of atoms
        """
        self.model_path = Path(model_path)
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.v_min = v_min
        self.v_max = v_max
        self.atom_size = atom_size

        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.support = torch.linspace(self.v_min, self.v_max, self.atom_size).to(self.device)

        self.model = self._load_model()
        self.model.eval()

    def _load_model(self) -> Network:
        """Load trained model."""
        if self.model_path.suffix == ".pth":
            model = torch.load(self.model_path, map_location=self.device)
            if isinstance(model, Network):
                return model
            else:
                network = Network(self.obs_dim, self.action_dim, self.atom_size, self.support).to(
                    self.device
                )
                network.load_state_dict(model)
                return network
        elif self.model_path.suffix == ".onnx":
            raise NotImplementedError("ONNX inference not yet implemented")
        else:
            raise ValueError(f"Unsupported model format: {self.model_path.suffix}")

    def predict(self, observation: np.ndarray) -> int:
        """Predict action for given observation.

        Args:
            observation: Current observation

        Returns:
            action: Predicted action (0 or 1)
        """
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(observation).to(self.device)
            if obs_tensor.dim() == 1:
                obs_tensor = obs_tensor.unsqueeze(0)
            action = self.model(obs_tensor).argmax().item()
        return action

    def predict_batch(self, observations: np.ndarray) -> np.ndarray:
        """Predict actions for batch of observations.

        Args:
            observations: Batch of observations

        Returns:
            actions: Predicted actions
        """
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(observations).to(self.device)
            actions = self.model(obs_tensor).argmax(dim=1).cpu().numpy()
        return actions

    def get_q_values(self, observation: np.ndarray) -> np.ndarray:
        """Get Q-values for given observation.

        Args:
            observation: Current observation

        Returns:
            q_values: Q-values for each action
        """
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(observation).to(self.device)
            if obs_tensor.dim() == 1:
                obs_tensor = obs_tensor.unsqueeze(0)
            q_values = self.model(obs_tensor).cpu().numpy()
        return q_values
