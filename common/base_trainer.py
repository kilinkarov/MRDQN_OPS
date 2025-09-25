#!/usr/bin/env python3
"""
Base Trainer Module - Common training logic for all DQN models
"""

import os
import sys
import numpy as np
import torch
import gymnasium as gym
import random
import pickle
from datetime import datetime
from abc import ABC, abstractmethod

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Rainbow_DQN'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'RL_Environment'))

# Import common utilities
from utils import GetPriceList, seed_torch
from CryptoEnv import CryptoEnv

# Register CryptoEnv
gym.envs.register(
    id='CryptoEnv-v0',
    entry_point='CryptoEnv:CryptoEnv',
)

class BaseTrainer(ABC):
    """
    Base trainer class with common training logic
    Subclasses only need to implement agent creation
    """
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        # ИСПОЛЬЗУЕМ ОРИГИНАЛЬНЫЕ ПАРАМЕТРЫ RAINBOW DQN!
        # Все модели должны использовать ХОРОШИЕ параметры Rainbow
        self.unified_params = {
            'memory_size': 10000,    # Оригинал Rainbow
            'batch_size': 128,       # Оригинал Rainbow  
            'target_update': 100,    # Оригинал Rainbow
            'gamma': 0.95,
            'lr': 6.25e-5,
            'n_step': 3,
            'layer_size': 512
        }
    
    @abstractmethod
    def create_agent(self, env):
        """
        Create and return the specific agent for this trainer
        Must be implemented by subclasses
        """
        pass
    
    @abstractmethod
    def get_model_save_path(self):
        """
        Return the path where to save the trained model
        Must be implemented by subclasses
        """
        pass
    
    def load_and_split_data(self, data_path='../Back-testing/Data.pkl'):
        """Load cryptocurrency data and split into train/test"""
        with open(data_path, 'rb') as f:
            content = pickle.load(f)
        
        total_data, start_date = GetPriceList(content, name_num=0)
        start_index = 1000  # Увеличиваем данные для обучения
        useful_data = total_data[start_index:]
        
        # Split: Use last 365 days for testing, rest for training
        test_days = 365
        train_data = useful_data[:-test_days]
        
        return train_data, start_date
    
    def setup_training_environment(self, train_data, seed=777):
        """Setup training environment and seeds"""
        # Create environment
        train_env = gym.make('CryptoEnv-v0', data=train_data, wnd_t=30, cycle_T=9)
        
        # Set seeds
        np.random.seed(seed)
        random.seed(seed)
        seed_torch(seed)
        
        return train_env
    
    def train_agent(self, agent, env, num_frames, model_name):
        """Common training loop for all agents"""
        print(f"🏃 Starting {model_name} training...")
        print(f"🔧 Device: {agent.device}")
        print(f"📊 Architecture: {model_name} with 512 hidden units")
        print(f"⚙️ Hyperparams: lr=6.25e-5, batch=32, memory=20k, target_update=8000, gamma=0.95")
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
    
    def save_model_and_results(self, agent, scores, losses):
        """Save trained model and results"""
        # Save model
        model_dir = self.get_model_save_path()
        os.makedirs(model_dir, exist_ok=True)
        torch.save(agent.dqn, f'{model_dir}/model.pth')
        print(f"✅ {self.model_name} training completed! Model saved to {model_dir}")
        
        # Save results
        results = {
            self.model_name.lower(): {
                'scores': scores,
                'losses': losses
            }
        }
        
        os.makedirs('../logs', exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f'../logs/{self.model_name.lower()}_training_results_{timestamp}.pkl'
        with open(results_file, 'wb') as f:
            pickle.dump(results, f)
        
        print(f"💾 {self.model_name} results saved: {results_file}")
        return results_file
    
    def train(self, data_path='../Back-testing/Data.pkl', num_frames=100000, seed=777):
        """
        Main training method - orchestrates the entire training process
        This is the only method that needs to be called externally
        """
        print(f"🚀 === {self.model_name.upper()} TRAINING ===")
        print(f"📊 Frames: {num_frames}")
        print(f"🌱 Seed: {seed}")
        
        # Load and split data
        train_data, start_date = self.load_and_split_data(data_path)
        print(f"🏋️ Training data: {len(train_data)} days")
        
        # Setup environment
        train_env = self.setup_training_environment(train_data, seed)
        
        # Create agent (implemented by subclass)
        agent = self.create_agent(train_env)
        
        # Train agent
        scores, losses = self.train_agent(agent, train_env, num_frames, self.model_name)
        
        # Save results
        self.save_model_and_results(agent, scores, losses)
        
        return scores, losses


def create_trainer_function(trainer_class):
    """
    Factory function to create a training function from a trainer class
    This allows each model to have a simple train_model_name() function
    """
    def train_function(data_path='../Back-testing/Data.pkl', num_frames=100000, seed=777):
        trainer = trainer_class()
        return trainer.train(data_path, num_frames, seed)
    
    return train_function
