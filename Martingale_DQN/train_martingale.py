#!/usr/bin/env python3
"""
Martingale DQN Training Module - Only Martingale-specific parts
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'common'))

from common.base_trainer import BaseTrainer, create_trainer_function
from martingale_dqn import MartingaleDQNAgent

class MartingaleTrainer(BaseTrainer):
    """Martingale DQN specific trainer"""
    
    def __init__(self):
        super().__init__("Martingale")
    
    def create_agent(self, env):
        """Create Martingale DQN agent with martingale enhancement"""
        return MartingaleDQNAgent(
            env,
            memory_size=self.unified_params['memory_size'],
            batch_size=self.unified_params['batch_size'],
            target_update=self.unified_params['target_update'],
            gamma=self.unified_params['gamma'],
            # Categorical DQN parameters
            v_min=0.0,
            v_max=20.0,
            atom_size=51,
            n_step=self.unified_params['n_step'],
            # Martingale-specific parameters
            martingale_weight=0.3,
            martingale_lr=self.unified_params['lr']
        )
    
    def get_model_save_path(self):
        """Martingale model save path"""
        return './models/Martingale_full_training/'

# Create the training function
train_martingale_dqn = create_trainer_function(MartingaleTrainer)

if __name__ == "__main__":
    train_martingale_dqn()
