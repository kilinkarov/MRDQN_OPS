#!/usr/bin/env python3
"""
Rainbow DQN Training Module - Only Rainbow-specific parts
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'common'))

from common.base_trainer import BaseTrainer, create_trainer_function
from rainbow import DQNAgent

class RainbowTrainer(BaseTrainer):
    """Rainbow DQN specific trainer"""
    
    def __init__(self):
        super().__init__("Rainbow")
    
    def create_agent(self, env):
        """Create Rainbow DQN agent with PER and all Rainbow features"""
        return DQNAgent(
            env,
            memory_size=self.unified_params['memory_size'],
            batch_size=self.unified_params['batch_size'],
            target_update=self.unified_params['target_update'],
            gamma=self.unified_params['gamma'],
            # Rainbow-specific parameters
            alpha=0.2,          # PER alpha
            beta=0.6,           # PER beta
            prior_eps=1e-6,     # PER epsilon
            v_min=0.0,          # Categorical DQN
            v_max=20.0,         # Categorical DQN
            atom_size=51,       # Categorical DQN
            n_step=self.unified_params['n_step']
        )
    
    def get_model_save_path(self):
        """Rainbow model save path"""
        return './models/Rainbow_full_training/'

# Create the training function
train_rainbow_dqn = create_trainer_function(RainbowTrainer)

if __name__ == "__main__":
    train_rainbow_dqn()
