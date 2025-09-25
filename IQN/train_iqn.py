#!/usr/bin/env python3
"""
IQN Training Module - Only IQN-specific parts
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'common'))

from common.base_trainer import BaseTrainer, create_trainer_function
from model import IQN
from util import calculate_huber_loss
import torch
import random
from collections import deque

class IQNAgentAdapter:
    """Adapter to make IQN compatible with base trainer interface"""
    
    def __init__(self, env, memory_size=10000, batch_size=128, target_update=100,
                 gamma=0.95, lr=6.25e-5, n_step=3, layer_size=512):
        
        obs_dim = env.observation_space.shape[1]
        action_dim = env.action_space.n
        
        self.env = env
        self.batch_size = batch_size
        self.target_update = target_update
        self.gamma = gamma
        self.n_step = n_step
        self.memory_size = memory_size
        
        # Device setup
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        
        # IQN network
        self.iqn = IQN(
            state_size=obs_dim,
            action_size=action_dim, 
            layer_size=layer_size,
            n_step=n_step,
            seed=42,
            distortion='neutral',
            con_val_at_risk=False
        ).to(self.device)
        
        # Target network
        self.iqn_target = IQN(
            state_size=obs_dim,
            action_size=action_dim,
            layer_size=layer_size, 
            n_step=n_step,
            seed=42,
            distortion='neutral',
            con_val_at_risk=False
        ).to(self.device)
        
        self.iqn_target.load_state_dict(self.iqn.state_dict())
        self.iqn_target.eval()
        
        # Optimizer
        self.optimizer = torch.optim.Adam(self.iqn.parameters(), lr=lr, eps=1.5e-4)
        
        # Memory
        self.memory = deque(maxlen=memory_size)
        
        # Make dqn attribute point to iqn for compatibility
        self.dqn = self.iqn
        
    def select_action(self, state):
        """Select action using IQN"""
        if len(self.memory) < self.batch_size:
            return random.randint(0, self.env.action_space.n - 1)
            
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            quantiles, _ = self.iqn.forward(state_tensor, 32, 'neutral')
            action = quantiles.mean(dim=1).argmax(dim=1).item()
        return action
    
    def step(self, action):
        """Take step in environment"""
        next_state, reward, done, info = self.env.step(action)
        return next_state, reward, done
        
    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in memory"""
        self.memory.append((state, action, reward, next_state, done))
        
    def update_model(self):
        """Update IQN model"""
        if len(self.memory) < self.batch_size:
            return 0
            
        # Sample batch
        batch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor([e[0] for e in batch]).to(self.device)
        actions = torch.LongTensor([e[1] for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e[2] for e in batch]).to(self.device)
        next_states = torch.FloatTensor([e[3] for e in batch]).to(self.device)
        dones = torch.BoolTensor([e[4] for e in batch]).to(self.device)
        
        # Current Q values
        current_quantiles, current_taus = self.iqn.forward(states, self.iqn.N, 'neutral')
        current_sa_quantiles = current_quantiles.gather(2, actions.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.iqn.N))
        
        # Next Q values  
        with torch.no_grad():
            next_quantiles, _ = self.iqn_target.forward(next_states, self.iqn.N, 'neutral')
            next_actions = self.iqn.forward(next_states, self.iqn.K, 'neutral')[0].mean(dim=1).argmax(dim=1)
            next_sa_quantiles = next_quantiles.gather(2, next_actions.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.iqn.N))
            
        # Target quantiles
        target_quantiles = rewards.unsqueeze(-1).unsqueeze(-1) + (self.gamma ** self.n_step) * next_sa_quantiles * (~dones).unsqueeze(-1).unsqueeze(-1)
        
        # Quantile regression loss
        td_errors = target_quantiles.unsqueeze(2) - current_sa_quantiles.unsqueeze(3)
        huber_loss = calculate_huber_loss(td_errors)
        
        # Quantile weights
        tau_hat = current_taus.unsqueeze(-1)
        weight = torch.abs(tau_hat - (td_errors < 0).float())
        loss = (weight * huber_loss).mean()
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
        
    def _target_hard_update(self):
        """Hard update target network"""
        self.iqn_target.load_state_dict(self.iqn.state_dict())


class IQNTrainer(BaseTrainer):
    """IQN specific trainer"""
    
    def __init__(self):
        super().__init__("IQN")
    
    def create_agent(self, env):
        """Create IQN agent with quantile regression"""
        return IQNAgentAdapter(
            env,
            memory_size=self.unified_params['memory_size'],
            batch_size=self.unified_params['batch_size'],
            target_update=self.unified_params['target_update'],
            gamma=self.unified_params['gamma'],
            lr=self.unified_params['lr'],
            n_step=self.unified_params['n_step'],
            layer_size=self.unified_params['layer_size']
        )
    
    def get_model_save_path(self):
        """IQN model save path"""
        return './models/IQN_full_training/'
    
    def train_agent(self, agent, env, num_frames, model_name):
        """Custom training loop for IQN with proper memory management"""
        import numpy as np
        
        print(f"🏃 Starting {model_name} training...")
        print(f"🔧 Device: {agent.device}")
        print(f"📊 Architecture: {model_name} with 512 hidden units")
        print(f"⚙️ Hyperparams: lr=6.25e-5, batch=128, memory=10k, target_update=100, gamma=0.95")
        print(f"🚀 Starting training with {num_frames:,} frames...")
        
        state, _ = env.reset()
        scores = []
        losses = []
        score = 0
        update_cnt = 0
        episode_count = 0
        
        for frame_idx in range(1, num_frames + 1):
            action = agent.select_action(state)
            next_state, reward, done = agent.step(action)
            
            # Store transition
            agent.store_transition(state, action, reward, next_state, done)
            
            state = next_state
            score += reward
            
            if done:
                state, _ = env.reset()
                scores.append(score)
                episode_count += 1
                score = 0
            
            # Update model
            if len(agent.memory) >= agent.batch_size:
                loss = agent.update_model()
                losses.append(loss)
                update_cnt += 1
                
                if update_cnt % agent.target_update == 0:
                    agent._target_hard_update()
            
            # Progress logging
            if frame_idx % 2000 == 0:
                mean_score = np.mean(scores[-50:]) if len(scores) >= 50 else (np.mean(scores) if scores else 0)
                recent_loss = np.mean(losses[-100:]) if len(losses) >= 100 else (np.mean(losses) if losses else 0)
                memory_usage = len(agent.memory)
                print(f"🔥 {model_name} - Frame: {frame_idx:,}/{num_frames:,} ({frame_idx/num_frames*100:.1f}%) | "
                      f"Episodes: {episode_count} | Score: {mean_score:.4f} | Loss: {recent_loss:.4f} | "
                      f"Memory: {memory_usage:,} | Updates: {update_cnt:,}")
                
                if frame_idx % 5000 == 0:
                    print(f"📈 {model_name} Progress: {frame_idx/num_frames*100:.1f}% complete, "
                          f"Recent episodes: {episode_count}, Avg score trend: {mean_score:.4f}")
        
        return scores, losses

# Create the training function
train_iqn = create_trainer_function(IQNTrainer)

if __name__ == "__main__":
    train_iqn()
