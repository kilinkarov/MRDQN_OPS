#!/usr/bin/env python3
"""
Full Training and Comparison Script for Martingale DQN vs Rainbow DQN

This script:
1. Trains both models on ALL years except the last year
2. Tests on the LAST YEAR only
3. Generates comprehensive comparison plots with boxplots and price intervals
4. Uses increased training frames for better convergence
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import torch
import gymnasium as gym
import random
from datetime import datetime
import seaborn as sns
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# Add paths for all models
sys.path.append('./Rainbow_DQN')
sys.path.append('./Martingale_DQN')
sys.path.append('./IQN')
sys.path.append('./RL_Environment')

# Register CryptoEnv programmatically
from CryptoEnv import CryptoEnv
gym.envs.register(
    id='CryptoEnv-v0',
    entry_point='CryptoEnv:CryptoEnv',
)

from rainbow import DQNAgent as RainbowAgent, GetPriceList, seed_torch
from martingale_dqn import MartingaleDQNAgent, sigmoid
from model import IQN

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class FullTrainingComparison:
    def __init__(self, data_path='Back-testing/Data.pkl', wnd_t=30, cycle_T=9, name_num=0):
        self.data_path = data_path
        self.wnd_t = wnd_t
        self.cycle_T = cycle_T
        self.name_num = name_num
        
        # Load and split data
        self.load_and_split_data()
        
        # Results storage
        self.results = {
            'rainbow': {'scores': [], 'losses': [], 'rewards': [], 'actions': [], 'stopping_times': [], 'prices': []},
            'martingale': {'scores': [], 'losses': [], 'rewards': [], 'actions': [], 'stopping_times': [], 'prices': []},
            'iqn': {'scores': [], 'losses': [], 'rewards': [], 'actions': [], 'stopping_times': [], 'prices': []},
            'random': {'rewards': [], 'stopping_times': [], 'prices': []}
        }
        
        print(f"📊 Data split:")
        print(f"   Training data: {len(self.train_data)} days")
        print(f"   Test data: {len(self.test_data)} days")
        
        # Load IQN training results if available
        self.load_iqn_training_results()

    def load_iqn_training_results(self):
        """Load IQN training results from pickle file"""
        try:
            # Найдем самый свежий файл результатов IQN
            import glob
            iqn_files = glob.glob('./logs/iqn_training_results_*.pkl')
            # Также ищем в старых местах
            iqn_files.extend(glob.glob('./IQN/iqn_training_results_*.pkl'))
            if iqn_files:
                latest_file = max(iqn_files, key=os.path.getctime)
                print(f"📂 Loading IQN training results from: {latest_file}")
                
                with open(latest_file, 'rb') as f:
                    iqn_results = pickle.load(f)
                
                if 'iqn' in iqn_results:
                    self.results['iqn']['scores'] = iqn_results['iqn']['scores']
                    self.results['iqn']['losses'] = iqn_results['iqn']['losses']
                    print(f"✅ Loaded IQN training results: {len(self.results['iqn']['scores'])} episodes, {len(self.results['iqn']['losses'])} loss values")
                else:
                    print("⚠️ IQN results file found but no 'iqn' key")
            else:
                print("⚠️ No IQN training results found")
        except Exception as e:
            print(f"❌ Error loading IQN training results: {e}")

    def load_and_split_data(self):
        """Load price data and split into train (all years except last) and test (last year)"""
        print("📂 Loading cryptocurrency data...")
        
        with open(self.data_path, 'rb') as f:
            content = pickle.load(f)
        
        total_data, start_date = GetPriceList(content, name_num=self.name_num)
        
        if self.name_num == 0:  # BTC
            start_index = 1000  # Увеличиваем данные для обучения
        else:  # ETH
            start_index = 1000  # Унифицируем для ETH тоже
            
        useful_data = total_data[start_index:]
        
        # Split: Use last 365 days for testing, rest for training
        test_days = 365
        self.train_data = useful_data[:-test_days]
        self.test_data = useful_data[-test_days-self.wnd_t:]  # Include window for proper initialization
        
        print(f"📅 Total useful data: {len(useful_data)} days")
        print(f"🏋️ Training data: {len(self.train_data)} days")
        print(f"🧪 Test data: {len(self.test_data)} days")

    def train_models(self, num_frames=100000, seed=777):
        """Train all models using external trainers"""
        print(f"\n🚀 === FULL TRAINING PHASE ===")
        print(f"🎯 Training frames: {num_frames}")
        print(f"🌱 Seed: {seed}")
        
        # Import external trainers (refactored versions)
        sys.path.append('./Rainbow_DQN')
        sys.path.append('./Martingale_DQN')
        sys.path.append('./IQN')
        
        from Rainbow_DQN.train_rainbow import train_rainbow_dqn
        from Martingale_DQN.train_martingale import train_martingale_dqn
        from IQN.train_iqn import train_iqn
        
        # Train Rainbow DQN
        print("\n🌈 === Training Rainbow DQN ===")
        rainbow_scores, rainbow_losses = train_rainbow_dqn(
            data_path='./Back-testing/Data.pkl',
            num_frames=num_frames,
            seed=seed
        )
        self.results['rainbow']['scores'] = rainbow_scores
        self.results['rainbow']['losses'] = rainbow_losses
        
        # Train Martingale DQN
        print("\n🎯 === Training Martingale DQN ===")
        martingale_scores, martingale_losses = train_martingale_dqn(
            data_path='./Back-testing/Data.pkl',
            num_frames=num_frames,
            seed=seed
        )
        self.results['martingale']['scores'] = martingale_scores
        self.results['martingale']['losses'] = martingale_losses
        
        # Train IQN
        print("\n⭐ === Training IQN ===")
        iqn_scores, iqn_losses = train_iqn(
            data_path='./Back-testing/Data.pkl',
            num_frames=num_frames,
            seed=seed
        )
        self.results['iqn']['scores'] = iqn_scores
        self.results['iqn']['losses'] = iqn_losses

    def _train_agent(self, agent, env, num_frames, agent_type, seed):
        """Train a single agent with detailed logging"""
        print(f"🏃 Starting {agent_type} DQN training...")
        print(f"🔧 Device: {agent.device}")
        print(f"📊 Architecture: 512 hidden units (upgraded from 128)")
        print(f"⚙️ Hyperparams: lr={agent.optimizer.param_groups[0]['lr']:.2e}, batch={agent.batch_size}, target_update={agent.target_update}")
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
            
            # Progress logging (увеличиваем частоту)
            if frame_idx % 2000 == 0:  # ✅ Каждые 2K frames вместо 10K
                mean_score = np.mean(scores[-50:]) if len(scores) >= 50 else (np.mean(scores) if scores else 0)
                recent_loss = np.mean(losses[-100:]) if len(losses) >= 100 else (np.mean(losses) if losses else 0)
                memory_usage = len(agent.memory) if hasattr(agent, 'memory') else 0
                print(f"🔥 {agent_type} - Frame: {frame_idx:,}/{num_frames:,} ({frame_idx/num_frames*100:.1f}%) | "
                      f"Episodes: {episode_count} | Score: {mean_score:.4f} | Loss: {recent_loss:.4f} | "
                      f"Memory: {memory_usage:,} | Updates: {update_cnt:,}")
                
                # Дополнительные принты каждые 5K frames
                if frame_idx % 5000 == 0:
                    print(f"📈 {agent_type} Progress: {frame_idx/num_frames*100:.1f}% complete, "
                          f"Recent episodes: {episode_count}, Avg score trend: {mean_score:.4f}")
        
        # Save model
        model_dir = f'./models/{agent_type}_full_training/'
        os.makedirs(model_dir, exist_ok=True)
        torch.save(agent.dqn, f'{model_dir}/model.pth')
        print(f"✅ {agent_type} training completed! Model saved to {model_dir}")
        
        return scores, losses

    def evaluate_models(self, test_episodes=200):
        """Evaluate both models on test data (last year)"""
        print(f"\n🧪 === EVALUATION PHASE ===")
        print(f"📊 Test episodes: {test_episodes}")
        print(f"📅 Testing on last year data ({len(self.test_data)} days)")
        
        # Create test environment
        test_env = gym.make('CryptoEnv-v0', data=self.test_data, wnd_t=self.wnd_t, cycle_T=self.cycle_T)
        
        for model_type in ['rainbow', 'martingale', 'iqn']:
            print(f"\n🔍 Evaluating {model_type.upper()} DQN...")
            
            # Create agent
            if model_type == 'rainbow':
                agent = RainbowAgent(test_env, memory_size=1000, batch_size=32, target_update=100)
            elif model_type == 'martingale':
                agent = MartingaleDQNAgent(test_env, memory_size=1000, batch_size=32, target_update=100)
            elif model_type == 'iqn':
                # Create IQN agent (no training needed, just for evaluation)
                # ВАЖНО: используем shape[1] как в train_iqn.py!
                obs_dim = test_env.observation_space.shape[1]  # 32 как в обучении
                action_dim = test_env.action_space.n  # 9
                agent = IQN(obs_dim, action_dim, layer_size=512, n_step=3, seed=0, 
                           distortion='neutral', con_val_at_risk=False)
                # Set device
                if torch.backends.mps.is_available():
                    device = torch.device("mps")
                elif torch.cuda.is_available():
                    device = torch.device("cuda")
                else:
                    device = torch.device("cpu")
                agent = agent.to(device)
                agent.device = device
            
            # Load trained model
            if model_type == 'iqn':
                model_path = f'./IQN/models/IQN_full_training/model_fixed.pth'
                if os.path.exists(model_path):
                    try:
                        # Загружаем исправленную модель (state_dict)
                        state_dict = torch.load(model_path, map_location=device, weights_only=False)
                        agent.load_state_dict(state_dict, strict=False)
                        agent.train()  # ✅ Оставляем в training режиме для exploration!
                        print(f"✅ Loaded IQN model from {model_path} (with exploration)")
                    except Exception as e:
                        print(f"❌ Error loading IQN model: {e}")
                        print(f"🔄 Trying alternative loading method...")
                        # Alternative: load just the state dict
                        try:
                            checkpoint = torch.load(model_path, map_location=device, weights_only=True)
                            if hasattr(checkpoint, 'state_dict'):
                                agent.load_state_dict(checkpoint.state_dict(), strict=False)
                            else:
                                agent.load_state_dict(checkpoint, strict=False)
                            agent.train()
                            print(f"✅ Loaded IQN model with alternative method")
                        except Exception as e2:
                            print(f"❌ Both loading methods failed: {e2}")
                            print(f"⏭️  Skipping IQN evaluation - will show empty results on dashboard")
                            # Создаем пустые результаты для IQN
                            self.results[model_type]['rewards'] = []
                            self.results[model_type]['actions'] = []
                            self.results[model_type]['stopping_times'] = []
                            self.results[model_type]['prices'] = []
                            continue
                else:
                    print(f"❌ IQN Model not found: {model_path}")
                    continue
            else:
                model_path = f'./models/{model_type.title()}_full_training/model.pth'
                if os.path.exists(model_path):
                    agent.dqn = torch.load(model_path, map_location=agent.device, weights_only=False)
                    agent.dqn.train()  # ✅ Оставляем в training режиме для Noisy Nets exploration!
                    print(f"✅ Loaded model from {model_path} (with exploration)")
                else:
                    print(f"❌ Model not found: {model_path}")
                    continue
            
            # Evaluate
            if model_type == 'iqn':
                results = self._evaluate_iqn_agent(agent, test_env, test_episodes)
            else:
                results = self._evaluate_agent(agent, test_env, test_episodes)
            
            self.results[model_type]['rewards'] = results['rewards']
            self.results[model_type]['actions'] = results['actions']
            self.results[model_type]['stopping_times'] = results['stopping_times']
            self.results[model_type]['prices'] = results['prices']
            
            # Print results
            print(f"📈 {model_type.upper()} Results:")
            print(f"   Average Reward: {np.mean(results['rewards']):.4f} ± {np.std(results['rewards']):.4f}")
            print(f"   Average Stopping Time: {np.mean(results['stopping_times']):.2f} ± {np.std(results['stopping_times']):.2f}")
            print(f"   Stop Rate: {np.mean([1 if a == 1 else 0 for a in results['actions']]):.3f}")
        
        # Evaluate Random Agent
        print(f"\n🎲 Evaluating RANDOM AGENT...")
        random_results = self._evaluate_random_agent(test_env, test_episodes)
        self.results['random']['rewards'] = random_results['rewards']
        self.results['random']['stopping_times'] = random_results['stopping_times']
        self.results['random']['prices'] = random_results['prices']
        
        # Calculate investment profits
        print(f"\n💰 Calculating investment profits...")
        self.profit_results = self._calculate_investment_profits()

    def _evaluate_agent(self, agent, test_env, max_episodes):
        """Evaluate a single agent on test environment using proper CryptoEnv API"""
        # Unwrap the environment to get the actual CryptoEnv
        unwrapped_env = test_env.unwrapped
        
        # Get all episodes from environment
        ev_episodes = unwrapped_env.prepare_episodes()
        original_episodes = unwrapped_env.prepare_original_episodes()
        
        rewards = []
        actions = []
        stopping_times = []
        prices = []
        
        # Limit to max_episodes
        num_episodes = min(max_episodes, len(ev_episodes))
        
        for e in range(num_episodes):
            episode = ev_episodes[e]
            original_episode = original_episodes[e]
            
            # ✅ ИСПРАВЛЯЕМ: используем цену ДО эпизода как референсную
            if e > 0:
                prev_episode = original_episodes[e-1]
                refer_value = prev_episode[-1][-1]  # Последняя цена предыдущего эпизода
            else:
                refer_value = original_episode[0][-1]  # Для первого эпизода
            episode_actions = []
            episode_prices = []
            episode_reward_sum = 0  # Сумма всех rewards за эпизод
            
            # ✅ ИСПРАВЛЯЕМ: используем ТУ ЖЕ нормализацию что и в обучении!
            # Нормализуем цены всего эпизода как в CryptoEnv.getPriceList()
            original_price_list = [original_episode[t][-1] for t in range(len(original_episode))]
            from RL_Environment.CryptoEnv import normalization
            normalized_price_list = normalization(original_price_list)
            
            for t, state in enumerate(episode):
                # Get current price
                current_price = original_episode[t][-1]
                episode_prices.append(current_price)
                
                # Prepare observation (ИСПРАВЛЯЕМ position_value)
                remain_t = (self.cycle_T - t) / self.cycle_T
                
                # ✅ НОРМАЛИЗУЕМ разность цен БЕЗ заглядывания в будущее
                price_diff = current_price - refer_value
                # Используем только цены ДО ТЕКУЩЕГО дня включительно
                typical_range = max(episode_prices) - min(episode_prices) if len(episode_prices) > 1 else abs(refer_value * 0.1)
                if typical_range > 0:
                    normalized_diff = price_diff / typical_range
                else:
                    normalized_diff = 0
                position_value = sigmoid(normalized_diff)
                
                obs = np.concatenate(([position_value, remain_t], state))
                
                obs = obs.reshape(1, self.wnd_t + 2)
                
                # Get action from agent
                with torch.no_grad():
                    obs_tensor = torch.FloatTensor(obs).to(agent.device)
                    action = agent.dqn(obs_tensor).argmax().item()
                
                episode_actions.append(action)
                
                # ✅ ИСПОЛЬЗУЕМ ТОТ ЖЕ REWARD ЧТО И В ОБУЧЕНИИ!
                # Берем нормализованную цену из normalized_price_list (как в CryptoEnv.step)
                current_normalized_price = normalized_price_list[t]
                
                import math
                # Защита от крайних значений (как в CryptoEnv.step)
                if current_normalized_price == 0:
                    current_normalized_price += 0.001
                if current_normalized_price == 1:
                    current_normalized_price -= 0.001
                
                if action == 1:  # Остановка
                    step_reward = math.log((1 - current_normalized_price) / current_normalized_price)
                else:  # Ожидание
                    step_reward = -0.5 * math.log((1 - current_normalized_price) / current_normalized_price)
                
                episode_reward_sum += step_reward
                
                # Check stopping condition
                if action == 1 or t == self.cycle_T - 1:
                    # ✅ ИСПОЛЬЗУЕМ ТОЛЬКО REWARD ЗА ДЕЙСТВИЕ ОСТАНОВКИ
                    # Это справедливо - все агенты оцениваются по качеству финального решения
                    stopping_reward = step_reward  # Reward именно за остановку на этом шаге
                    rewards.append(stopping_reward)
                    stopping_times.append(t)
                    
                    
                    actions.extend(episode_actions)
                    prices.extend(episode_prices)
                    break
        
        return {
            'rewards': rewards,
            'actions': actions,
            'stopping_times': stopping_times,
            'prices': prices
        }

    def _evaluate_iqn_agent(self, iqn_model, test_env, max_episodes):
        """Evaluate IQN model on test environment"""
        # Unwrap the environment to get the actual CryptoEnv
        unwrapped_env = test_env.unwrapped
        
        # Get all episodes from environment
        ev_episodes = unwrapped_env.prepare_episodes()
        original_episodes = unwrapped_env.prepare_original_episodes()
        
        rewards = []
        actions = []
        stopping_times = []
        prices = []
        
        # Limit to max_episodes
        num_episodes = min(max_episodes, len(ev_episodes))
        
        for e in range(num_episodes):
            episode = ev_episodes[e]
            original_episode = original_episodes[e]
            
            # ✅ ИСПРАВЛЯЕМ: используем цену ДО эпизода как референсную
            if e > 0:
                prev_episode = original_episodes[e-1]
                refer_value = prev_episode[-1][-1]  # Последняя цена предыдущего эпизода
            else:
                refer_value = original_episode[0][-1]  # Для первого эпизода
            episode_actions = []
            episode_prices = []
            episode_reward_sum = 0  # Сумма всех rewards за эпизод
            
            # ✅ ИСПРАВЛЯЕМ: используем ТУ ЖЕ нормализацию что и в обучении!
            # Нормализуем цены всего эпизода как в CryptoEnv.getPriceList()
            original_price_list = [original_episode[t][-1] for t in range(len(original_episode))]
            from RL_Environment.CryptoEnv import normalization
            normalized_price_list = normalization(original_price_list)
            
            for t, state in enumerate(episode):
                # Get current price
                current_price = original_episode[t][-1]
                episode_prices.append(current_price)
                
                # Prepare observation (ИСПРАВЛЯЕМ position_value)
                remain_t = (self.cycle_T - t) / self.cycle_T
                
                # ✅ НОРМАЛИЗУЕМ разность цен БЕЗ заглядывания в будущее
                price_diff = current_price - refer_value
                # Используем только цены ДО ТЕКУЩЕГО дня включительно
                typical_range = max(episode_prices) - min(episode_prices) if len(episode_prices) > 1 else abs(refer_value * 0.1)
                if typical_range > 0:
                    normalized_diff = price_diff / typical_range
                else:
                    normalized_diff = 0
                position_value = sigmoid(normalized_diff)
                
                # Create observation
                obs = np.array([remain_t, position_value] + list(state), dtype=np.float32)
                
                # Get action from IQN model
                with torch.no_grad():
                    obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(iqn_model.device)
                    # Use K=32 samples for action selection
                    quantiles, _ = iqn_model.forward(obs_tensor, 32, 'neutral')
                    action = quantiles.mean(dim=1).argmax(dim=1).item()
                
                episode_actions.append(action)
                
                # ✅ ИСПОЛЬЗУЕМ ТОТ ЖЕ REWARD ЧТО И В ОБУЧЕНИИ!
                # Берем нормализованную цену из normalized_price_list (как в CryptoEnv.step)
                current_normalized_price = normalized_price_list[t]
                
                import math
                # Защита от крайних значений (как в CryptoEnv.step)
                if current_normalized_price == 0:
                    current_normalized_price += 0.001
                if current_normalized_price == 1:
                    current_normalized_price -= 0.001
                
                if action == 1:  # Остановка
                    step_reward = math.log((1 - current_normalized_price) / current_normalized_price)
                else:  # Ожидание
                    step_reward = -0.5 * math.log((1 - current_normalized_price) / current_normalized_price)
                
                episode_reward_sum += step_reward
                
                # Check stopping condition
                if action == 1 or t == self.cycle_T - 1:
                    # ✅ ИСПОЛЬЗУЕМ ТОЛЬКО REWARD ЗА ДЕЙСТВИЕ ОСТАНОВКИ
                    # Это справедливо - все агенты оцениваются по качеству финального решения
                    stopping_reward = step_reward  # Reward именно за остановку на этом шаге
                    rewards.append(stopping_reward)
                    stopping_times.append(t)
                    
                    actions.extend(episode_actions)
                    prices.extend(episode_prices)
                    break
        
        return {
            'rewards': rewards,
            'actions': actions,
            'stopping_times': stopping_times,
            'prices': prices
        }

    def generate_comprehensive_plots(self):
        """Generate interactive Plotly dashboard with investment intervals"""
        print("\n📊 === GENERATING INTERACTIVE PLOTLY DASHBOARD ===")
        
        # Create subplots
        fig = make_subplots(
            rows=5, cols=2,
            subplot_titles=(
                'Bitcoin Price with Investment Intervals & Decisions',
                'Stopping Decision Reward (quality of final decision)',
                'Cumulative Evaluation Rewards',
                'Rank from Episode Minimum (Histogram)',
                'Cumulative Investment Profit ($)',
                'Investment Profit per Episode ($)',
                'Decision Day Comparison (Box Plot)',
                'Training Scores (Smoothed)',
                'Training Loss (Smoothed)'
            ),
            specs=[
                [{"colspan": 2}, None],  # Top row spans both columns
                [{}, {}],                # Second row: two separate plots
                [{}, {}],                # Third row: two separate plots
                [{}, {}],                # Fourth row: two separate plots
                [{}, {}]                 # Fifth row: two separate plots
            ],
            vertical_spacing=0.08,
            horizontal_spacing=0.10
        )
        
        # Colors
        btc_color = '#F7931E'
        rainbow_color = '#FF6B6B'
        martingale_color = '#4ECDC4'
        random_color = '#95A5A6'
        interval_color = 'rgba(200, 200, 200, 0.3)'
        
        # 1. Main Bitcoin Price Chart with Investment Intervals
        if len(self.test_data) > 0:
            days = list(range(len(self.test_data)))
            
            # Add Bitcoin price line
            fig.add_trace(
                go.Scatter(
                    x=days,
                    y=self.test_data,
                    mode='lines',
                    name='Bitcoin Price (Test Period)',
                    line=dict(color=btc_color, width=2),
                    hovertemplate='Day: %{x}<br>Price: $%{y:,.0f}<extra></extra>'
                ),
                row=1, col=1
            )
            
            # Add investment intervals and decisions
            self._add_investment_intervals_to_plot(fig, days)
        
        # 2. Evaluation Rewards per Episode (200 эпизодов)
        if self.results['rainbow']['rewards'] and self.results['martingale']['rewards'] and self.results['random']['rewards']:
            # Используем evaluation rewards - ПРОСТЫЕ НАГРАДЫ ЗА 200 ЭПИЗОДОВ
            rainbow_rewards = self.results['rainbow']['rewards']
            martingale_rewards = self.results['martingale']['rewards']
            iqn_rewards = self.results['iqn']['rewards'] if self.results['iqn']['rewards'] else []
            random_rewards = self.results['random']['rewards']
            episodes = list(range(len(rainbow_rewards)))
            
            # Rainbow DQN evaluation rewards
            fig.add_trace(
                go.Scatter(
                    x=episodes,
                    y=rainbow_rewards,
                    mode='lines+markers',
                    name='Rainbow DQN',
                    line=dict(color=rainbow_color, width=2),
                    marker=dict(size=4),
                    hovertemplate='Episode: %{x}<br>Reward: %{y:.4f}<extra></extra>'
                ),
                row=2, col=1
            )
            
            # Martingale DQN evaluation rewards
            fig.add_trace(
                go.Scatter(
                    x=episodes,
                    y=martingale_rewards,
                    mode='lines+markers',
                    name='Martingale DQN',
                    line=dict(color=martingale_color, width=2),
                    marker=dict(size=4),
                    hovertemplate='Episode: %{x}<br>Reward: %{y:.4f}<extra></extra>'
                ),
                row=2, col=1
            )
            
            # IQN evaluation rewards (может быть пустым)
            if iqn_rewards:
                fig.add_trace(
                    go.Scatter(
                        x=episodes[:len(iqn_rewards)],
                        y=iqn_rewards,
                        mode='lines+markers',
                        name='IQN',
                        line=dict(color='purple', width=2),
                        marker=dict(size=4),
                        hovertemplate='Episode: %{x}<br>Reward: %{y:.4f}<extra></extra>'
                    ),
                    row=2, col=1
                )

            # Random Agent evaluation rewards
            fig.add_trace(
                go.Scatter(
                    x=episodes,
                    y=random_rewards,
                    mode='lines+markers',
                    name='Random Agent',
                    line=dict(color=random_color, width=2, dash='dot'),
                    marker=dict(size=4),
                    hovertemplate='Episode: %{x}<br>Reward: %{y:.4f}<extra></extra>'
                ),
                row=2, col=1
            )
            
            # Random baseline
            fig.add_hline(y=0.5, line_dash="dash", line_color="black", 
                         annotation_text="Random (0.5)", row=2, col=1)
        
        # 3. Cumulative Evaluation Rewards
        if self.results['rainbow']['rewards'] and self.results['martingale']['rewards'] and self.results['random']['rewards']:
            rainbow_cumulative = np.cumsum(rainbow_rewards)
            martingale_cumulative = np.cumsum(martingale_rewards)
            iqn_cumulative = np.cumsum(iqn_rewards) if iqn_rewards else []
            random_cumulative = np.cumsum(random_rewards)
            
            fig.add_trace(
                go.Scatter(
                    x=episodes,
                    y=rainbow_cumulative,
                    mode='lines',
                    name='Rainbow DQN Cumulative',
                    line=dict(color=rainbow_color, width=3),
                    hovertemplate='Episode: %{x}<br>Cumulative: %{y:.4f}<extra></extra>',
                    showlegend=False
                ),
                row=2, col=2
            )
            
            fig.add_trace(
                go.Scatter(
                    x=episodes,
                    y=martingale_cumulative,
                    mode='lines',
                    name='Martingale DQN Cumulative',
                    line=dict(color=martingale_color, width=3),
                    hovertemplate='Episode: %{x}<br>Cumulative: %{y:.4f}<extra></extra>',
                    showlegend=False
                ),
                row=2, col=2
            )
            
            # IQN cumulative (может быть пустым)
            if len(iqn_cumulative) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=episodes[:len(iqn_cumulative)],
                        y=iqn_cumulative,
                        mode='lines',
                        name='IQN Cumulative',
                        line=dict(color='purple', width=3),
                        hovertemplate='Episode: %{x}<br>Cumulative: %{y:.4f}<extra></extra>',
                        showlegend=False
                    ),
                    row=2, col=2
                )
            
            # Random cumulative
            fig.add_trace(
                go.Scatter(
                    x=episodes,
                    y=random_cumulative,
                    mode='lines',
                    name='Random Agent Cumulative',
                    line=dict(color=random_color, width=3, dash='dot'),
                    hovertemplate='Episode: %{x}<br>Cumulative: %{y:.4f}<extra></extra>',
                    showlegend=False
                ),
                row=2, col=2
            )
            
            
        
        
        # 4. Investment Profit per Episode (создаем простые данные если нет profit_results)
        if hasattr(self, 'profit_results') and self.profit_results and 'iqn' in self.profit_results:
            # Используем реальные profit_results
            profit_episodes = []
            rainbow_profits = []
            martingale_profits = []
            iqn_profits = []
            random_profits = []
            
            # ✅ ИСПРАВЛЯЕМ: правильно собираем данные по эпизодам для всех агентов
            # Сначала найдем общие эпизоды для всех агентов
            all_episodes = set()
            agent_data = {}
            
            for agent_name in ['rainbow', 'martingale', 'iqn', 'random']:
                investments = self.profit_results[agent_name]['investments']
                agent_data[agent_name] = {inv['episode']: inv['episode_profit'] for inv in investments}
                all_episodes.update(agent_data[agent_name].keys())
            
            # Сортируем эпизоды и берем только те, где есть данные для всех агентов
            common_episodes = sorted([ep for ep in all_episodes 
                                    if all(ep in agent_data[agent] for agent in ['rainbow', 'martingale', 'iqn', 'random'])])
            
            # Формируем массивы в правильном порядке
            profit_episodes = common_episodes
            rainbow_profits = [agent_data['rainbow'][ep] for ep in common_episodes]
            martingale_profits = [agent_data['martingale'][ep] for ep in common_episodes]
            iqn_profits = [agent_data['iqn'][ep] for ep in common_episodes]
            random_profits = [agent_data['random'][ep] for ep in common_episodes]
            
            if profit_episodes:
                # Rainbow profits
                fig.add_trace(
                    go.Scatter(
                        x=profit_episodes,
                        y=rainbow_profits,
                        mode='markers+lines',
                        name='Rainbow DQN Profit',
                        line=dict(color=rainbow_color, width=2),
                        marker=dict(size=6),
                        hovertemplate='Episode: %{x}<br>Profit: $%{y:,.0f}<extra></extra>',
                        showlegend=False
                    ),
                    row=4, col=1
                )
                
                # Martingale profits
                fig.add_trace(
                    go.Scatter(
                        x=profit_episodes,
                        y=martingale_profits,
                        mode='markers+lines',
                        name='Martingale DQN Profit',
                        line=dict(color=martingale_color, width=2),
                        marker=dict(size=6),
                        hovertemplate='Episode: %{x}<br>Profit: $%{y:,.0f}<extra></extra>',
                        showlegend=False
                    ),
                    row=4, col=1
                )
                
                # IQN profits
                fig.add_trace(
                    go.Scatter(
                        x=profit_episodes,
                        y=iqn_profits,
                        mode='markers+lines',
                        name='IQN Profit',
                        line=dict(color='purple', width=2),
                        marker=dict(size=6),
                        hovertemplate='Episode: %{x}<br>Profit: $%{y:,.0f}<extra></extra>',
                        showlegend=False
                    ),
                    row=4, col=1
                )
                
                # Random profits
                fig.add_trace(
                    go.Scatter(
                        x=profit_episodes,
                        y=random_profits,
                        mode='markers+lines',
                        name='Random Agent Profit',
                        line=dict(color=random_color, width=2, dash='dot'),
                        marker=dict(size=6),
                        hovertemplate='Episode: %{x}<br>Profit: $%{y:,.0f}<extra></extra>',
                        showlegend=False
                    ),
                    row=4, col=1
                )
                
                # Add zero line
                fig.add_hline(y=0, line_dash="dash", line_color="black", 
                             annotation_text="Break Even", row=4, col=1)
        else:
            # НЕТ ДАННЫХ - ПОКАЗЫВАЕМ ПУСТОЙ ГРАФИК
            fig.add_annotation(
                text="❌ No Investment Profit Data Available<br>Run evaluation with --train to generate profit data",
                xref="x4", yref="y4",
                x=0.5, y=0.5,
                xanchor='center', yanchor='middle',
                showarrow=False,
                font=dict(size=14, color="red"),
                row=4, col=1
            )
        
        # 5. Cumulative Investment Profit
        if hasattr(self, 'profit_results') and self.profit_results and 'iqn' in self.profit_results and 'profit_episodes' in locals() and profit_episodes:
            rainbow_cumulative_profit = np.cumsum(rainbow_profits)
            martingale_cumulative_profit = np.cumsum(martingale_profits)
            iqn_cumulative_profit = np.cumsum(iqn_profits)
            random_cumulative_profit = np.cumsum(random_profits)
            
            # Rainbow cumulative profit
            fig.add_trace(
                go.Scatter(
                    x=profit_episodes,
                    y=rainbow_cumulative_profit,
                    mode='lines',
                    name='Rainbow DQN Cumulative Profit',
                    line=dict(color=rainbow_color, width=3),
                    hovertemplate='Episode: %{x}<br>Cumulative Profit: $%{y:,.0f}<extra></extra>',
                    showlegend=False
                ),
                row=3, col=2
            )
            
            # Martingale cumulative profit
            fig.add_trace(
                go.Scatter(
                    x=profit_episodes,
                    y=martingale_cumulative_profit,
                    mode='lines',
                    name='Martingale DQN Cumulative Profit',
                    line=dict(color=martingale_color, width=3),
                    hovertemplate='Episode: %{x}<br>Cumulative Profit: $%{y:,.0f}<extra></extra>',
                    showlegend=False
                ),
                row=3, col=2
            )
            
            # IQN cumulative profit
            fig.add_trace(
                go.Scatter(
                    x=profit_episodes,
                    y=iqn_cumulative_profit,
                    mode='lines',
                    name='IQN Cumulative Profit',
                    line=dict(color='purple', width=3),
                    hovertemplate='Episode: %{x}<br>Cumulative Profit: $%{y:,.0f}<extra></extra>',
                    showlegend=False
                ),
                row=3, col=2
            )
            
            # Random cumulative profit
            fig.add_trace(
                go.Scatter(
                    x=profit_episodes,
                    y=random_cumulative_profit,
                    mode='lines',
                    name='Random Agent Cumulative Profit',
                    line=dict(color=random_color, width=3, dash='dot'),
                    hovertemplate='Episode: %{x}<br>Cumulative Profit: $%{y:,.0f}<extra></extra>',
                    showlegend=False
                ),
                row=3, col=2
            )
            
            # Add zero line
            fig.add_hline(y=0, line_dash="dash", line_color="black", 
                         annotation_text="Break Even", row=3, col=2)
        else:
            # НЕТ ДАННЫХ - ПОКАЗЫВАЕМ ПУСТОЙ ГРАФИК
            fig.add_annotation(
                text="❌ No Cumulative Investment Profit Data Available<br>Run evaluation with --train to generate profit data",
                xref="x3", yref="y3",
                x=0.5, y=0.5,
                xanchor='center', yanchor='middle',
                showarrow=False,
                font=dict(size=14, color="red"),
                row=3, col=2
            )
        
        # 6. Rank-from-Minimum Distribution (Histogram)
        if (self.results['rainbow']['stopping_times'] and 
            self.results['martingale']['stopping_times'] and 
            self.results['random']['stopping_times']):

            # Compute per-episode price rank using SAME data as evaluation
            rainbow_ranks = []
            martingale_ranks = []
            iqn_ranks = []
            random_ranks = []

            # Используем ТЕ ЖЕ данные что и в evaluation - prepare_episodes()
            import gymnasium as gym
            test_env = gym.make('CryptoEnv-v0', data=self.test_data, wnd_t=self.wnd_t, cycle_T=self.cycle_T)
            unwrapped_env = test_env.unwrapped
            original_episodes = unwrapped_env.prepare_original_episodes()
            
            total_episodes = min(
                len(self.results['rainbow']['stopping_times']),
                len(self.results['martingale']['stopping_times']),
                len(self.results['random']['stopping_times']),
                len(original_episodes)
            )
            
            print(f"🔍 DEBUG: Total episodes from prepare_episodes: {len(original_episodes)}")
            print(f"🔍 DEBUG: Using episodes for histogram: {total_episodes}")

            for ep in range(total_episodes):
                # Используем original_episodes как в evaluation
                original_episode = original_episodes[ep]
                prices_ep = [original_episode[t][-1] for t in range(len(original_episode))]
                
                if len(prices_ep) < self.cycle_T:
                    continue

                # Rank: 1 = lowest price in episode, 2 = second lowest, ..., T = highest
                order = np.argsort(prices_ep)
                ranks = np.empty_like(order)
                for pos, idx in enumerate(order):
                    ranks[idx] = pos

                r_stop = int(self.results['rainbow']['stopping_times'][ep])
                m_stop = int(self.results['martingale']['stopping_times'][ep])
                z_stop = int(self.results['random']['stopping_times'][ep])

                rainbow_ranks.append(int(ranks[r_stop]) + 1)
                martingale_ranks.append(int(ranks[m_stop]) + 1)
                random_ranks.append(int(ranks[z_stop]) + 1)
                
                # IQN может быть пустым
                if self.results['iqn']['stopping_times'] and ep < len(self.results['iqn']['stopping_times']):
                    i_stop = int(self.results['iqn']['stopping_times'][ep])
                    iqn_ranks.append(int(ranks[i_stop]) + 1)

            # Debug: проверим реальные данные рангов
            print(f"🔍 DEBUG: Rainbow ranks sample: {rainbow_ranks[:10]}")
            print(f"🔍 DEBUG: Martingale ranks sample: {martingale_ranks[:10]}")
            print(f"🔍 DEBUG: IQN ranks sample: {iqn_ranks[:10]}")
            print(f"🔍 DEBUG: Random ranks sample: {random_ranks[:10]}")
            
            # Проверим частоты
            import collections
            rainbow_freq = collections.Counter(rainbow_ranks)
            martingale_freq = collections.Counter(martingale_ranks)
            iqn_freq = collections.Counter(iqn_ranks)
            random_freq = collections.Counter(random_ranks)
            
            print(f"🔍 DEBUG: Rainbow frequencies: {dict(sorted(rainbow_freq.items()))}")
            print(f"🔍 DEBUG: Martingale frequencies: {dict(sorted(martingale_freq.items()))}")
            print(f"🔍 DEBUG: IQN frequencies: {dict(sorted(iqn_freq.items()))}")
            print(f"🔍 DEBUG: Random frequencies: {dict(sorted(random_freq.items()))}")
            
            # Plot histograms (1..cycle_T)
            if rainbow_ranks and martingale_ranks and random_ranks:
                fig.add_trace(
                    go.Histogram(
                        x=rainbow_ranks,
                        name='Rainbow DQN',
                        marker_color=rainbow_color,
                        opacity=0.65,
                        xbins=dict(start=0.5, end=self.cycle_T + 0.5, size=1),
                        histnorm='probability',
                        hovertemplate='Rank: %{x}<br>P: %{y:.3f}<extra></extra>'
                    ),
                    row=3, col=1
                )

                fig.add_trace(
                    go.Histogram(
                        x=martingale_ranks,
                        name='Martingale DQN',
                        marker_color=martingale_color,
                        opacity=0.65,
                        xbins=dict(start=0.5, end=self.cycle_T + 0.5, size=1),
                        histnorm='probability',
                        hovertemplate='Rank: %{x}<br>P: %{y:.3f}<extra></extra>'
                    ),
                    row=3, col=1
                )

                # IQN histogram только если есть данные
                if iqn_ranks:
                    fig.add_trace(
                        go.Histogram(
                            x=iqn_ranks,
                            name='IQN',
                            marker_color='purple',
                            opacity=0.65,
                            xbins=dict(start=0.5, end=self.cycle_T + 0.5, size=1),
                            histnorm='probability',
                            hovertemplate='Rank: %{x}<br>P: %{y:.3f}<extra></extra>'
                        ),
                        row=3, col=1
                    )

                fig.add_trace(
                    go.Histogram(
                        x=random_ranks,
                        name='Random Agent',
                        marker_color=random_color,
                        opacity=0.5,
                        xbins=dict(start=0.5, end=self.cycle_T + 0.5, size=1),
                        histnorm='probability',
                        hovertemplate='Rank: %{x}<br>P: %{y:.3f}<extra></extra>'
                    ),
                    row=3, col=1
                )
        
        # 7. Decision Day Box Plot Comparison
        if (self.results['rainbow']['stopping_times'] and 
            self.results['martingale']['stopping_times'] and 
            self.results['random']['stopping_times']):
            
            rainbow_stops = [t + 1 for t in self.results['rainbow']['stopping_times']]
            martingale_stops = [t + 1 for t in self.results['martingale']['stopping_times']]
            iqn_stops = [t + 1 for t in self.results['iqn']['stopping_times']] if self.results['iqn']['stopping_times'] else []
            random_stops = [t + 1 for t in self.results['random']['stopping_times']]
            
            # Box plot for Rainbow
            fig.add_trace(
                go.Box(
                    y=rainbow_stops,
                    name='Rainbow DQN',
                    marker_color=rainbow_color,
                    boxpoints='all',
                    jitter=0.3,
                    pointpos=-2.0,
                    hovertemplate='Decision Day: %{y}<extra></extra>'
                ),
                row=4, col=2
            )
            
            # Box plot for Martingale
            fig.add_trace(
                go.Box(
                    y=martingale_stops,
                    name='Martingale DQN',
                    marker_color=martingale_color,
                    boxpoints='all',
                    jitter=0.3,
                    pointpos=0,
                    hovertemplate='Decision Day: %{y}<extra></extra>'
                ),
                row=4, col=2
            )
            
            # Box plot for IQN только если есть данные
            if iqn_stops:
                fig.add_trace(
                    go.Box(
                        y=iqn_stops,
                        name='IQN',
                        marker_color='purple',
                        boxpoints='all',
                        jitter=0.3,
                        pointpos=0,
                        hovertemplate='Decision Day: %{y}<extra></extra>'
                    ),
                    row=4, col=2
                )
            
            # Box plot for Random
            fig.add_trace(
                go.Box(
                    y=random_stops,
                    name='Random Agent',
                    marker_color=random_color,
                    boxpoints='all',
                    jitter=0.3,
                    pointpos=2.0,
                    hovertemplate='Decision Day: %{y}<extra></extra>'
                ),
                row=4, col=2
            )
        
        # Update layout
        fig.update_layout(
            height=1200,
            title_text="🚀 Martingale DQN vs Rainbow DQN vs Random Agent: Investment Analysis Dashboard",
            title_x=0.5,
            title_font_size=18,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(l=60, r=60, t=100, b=60)
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Days", row=1, col=1)
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        
        fig.update_xaxes(title_text="Evaluation Episode", row=2, col=1)
        fig.update_yaxes(title_text="Reward", row=2, col=1)
        
        fig.update_xaxes(title_text="Evaluation Episode", row=2, col=2)
        fig.update_yaxes(title_text="Cumulative Reward", row=2, col=2)
        
        fig.update_xaxes(title_text="Rank from Minimum (1 = best)", row=3, col=1)
        fig.update_yaxes(title_text="Probability", row=3, col=1)
        
        fig.update_xaxes(title_text="Episode", row=3, col=2)
        fig.update_yaxes(title_text="Cumulative Profit ($)", row=3, col=2)
        
        fig.update_xaxes(title_text="Episode", row=4, col=1)
        fig.update_yaxes(title_text="Profit ($)", row=4, col=1)
        
        fig.update_xaxes(title_text="Agent", row=4, col=2)
        fig.update_yaxes(title_text="Average Decision Day", row=4, col=2)
        
        fig.update_xaxes(title_text="Episodes", row=5, col=1)
        fig.update_yaxes(title_text="Score", row=5, col=1)
        
        fig.update_xaxes(title_text="Training Steps", row=5, col=2)
        fig.update_yaxes(title_text="Loss", row=5, col=2)
        
        fig.update_xaxes(title_text="Episodes", row=5, col=1)
        fig.update_yaxes(title_text="Score", row=5, col=1)
        
        fig.update_xaxes(title_text="Training Steps", row=5, col=2)
        fig.update_yaxes(title_text="Loss", row=5, col=2            )
        
        # 8. Training Scores (СГЛАЖЕННЫЕ)
        print(f"🔍 DEBUG: Rainbow scores count: {len(self.results['rainbow']['scores'])}")
        print(f"🔍 DEBUG: Martingale scores count: {len(self.results['martingale']['scores'])}")
        print(f"🔍 DEBUG: IQN scores count: {len(self.results['iqn']['scores'])}")
        if self.results['rainbow']['scores'] and self.results['martingale']['scores']:
            # ✅ ВОЗВРАЩАЕМ СГЛАЖИВАНИЕ
            rainbow_raw = self.results['rainbow']['scores']
            martingale_raw = self.results['martingale']['scores']
            iqn_raw = self.results['iqn']['scores'] if self.results['iqn']['scores'] else []
            
            # Добавим статистику для анализа
            print(f"🔍 DEBUG: Rainbow scores stats: min={min(rainbow_raw):.3f}, max={max(rainbow_raw):.3f}, mean={sum(rainbow_raw)/len(rainbow_raw):.3f}")
            print(f"🔍 DEBUG: Martingale scores stats: min={min(martingale_raw):.3f}, max={max(martingale_raw):.3f}, mean={sum(martingale_raw)/len(martingale_raw):.3f}")
            if iqn_raw:
                print(f"🔍 DEBUG: IQN scores stats: min={min(iqn_raw):.3f}, max={max(iqn_raw):.3f}, mean={sum(iqn_raw)/len(iqn_raw):.3f}")
            
            # ✅ ПРИМЕНЯЕМ СГЛАЖИВАНИЕ (rolling average)
            import pandas as pd
            window_size = 50
            rainbow_smooth = pd.Series(rainbow_raw).rolling(window=window_size, min_periods=1).mean()
            martingale_smooth = pd.Series(martingale_raw).rolling(window=window_size, min_periods=1).mean()
            iqn_smooth = pd.Series(iqn_raw).rolling(window=window_size, min_periods=1).mean() if len(iqn_raw) > 0 else []
            
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(rainbow_smooth))),
                    y=rainbow_smooth,
                    mode='lines',
                    name='Rainbow DQN Training (Smoothed)',
                    line=dict(color=rainbow_color, width=2),
                    hovertemplate='Episode: %{x}<br>Smoothed Score: %{y:.4f}<extra></extra>',
                    showlegend=False
                ),
                row=5, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(martingale_smooth))),
                    y=martingale_smooth,
                    mode='lines',
                    name='Martingale DQN Training (Smoothed)',
                    line=dict(color=martingale_color, width=2),
                    hovertemplate='Episode: %{x}<br>Smoothed Score: %{y:.4f}<extra></extra>',
                    showlegend=False
                ),
                row=5, col=1
            )
            
            # IQN Training Scores (если есть)
            if len(iqn_smooth) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=list(range(len(iqn_smooth))),
                        y=iqn_smooth,
                        mode='lines',
                        name='IQN Training (Smoothed)',
                        line=dict(color='purple', width=2),
                        hovertemplate='Episode: %{x}<br>Smoothed Score: %{y:.4f}<extra></extra>',
                        showlegend=False
                    ),
                    row=5, col=1
                )
        
        # 9. Training Loss (добавляем обратно!)
        if self.results['rainbow']['losses'] and self.results['martingale']['losses']:
            rainbow_loss_smooth = pd.Series(self.results['rainbow']['losses']).rolling(window=100, min_periods=1).mean()
            martingale_loss_smooth = pd.Series(self.results['martingale']['losses']).rolling(window=100, min_periods=1).mean()
            iqn_loss_smooth = pd.Series(self.results['iqn']['losses']).rolling(window=100, min_periods=1).mean() if self.results['iqn']['losses'] else []
            
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(rainbow_loss_smooth))),
                    y=rainbow_loss_smooth,
                    mode='lines',
                    name='Rainbow DQN Loss',
                    line=dict(color=rainbow_color, width=2),
                    hovertemplate='Step: %{x}<br>Loss: %{y:.4f}<extra></extra>',
                    showlegend=False
                ),
                row=5, col=2
            )
            
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(martingale_loss_smooth))),
                    y=martingale_loss_smooth,
                    mode='lines',
                    name='Martingale DQN Loss',
                    line=dict(color=martingale_color, width=2),
                    hovertemplate='Step: %{x}<br>Loss: %{y:.4f}<extra></extra>',
                    showlegend=False
                ),
                row=5, col=2
            )
            
            # IQN Training Loss (если есть)
            if len(iqn_loss_smooth) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=list(range(len(iqn_loss_smooth))),
                        y=iqn_loss_smooth,
                        mode='lines',
                        name='IQN Loss',
                        line=dict(color='purple', width=2),
                        hovertemplate='Step: %{x}<br>Loss: %{y:.4f}<extra></extra>',
                        showlegend=False
                    ),
                    row=5, col=2
                )
        
        # Save as HTML in logs directory
        os.makedirs('logs', exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        html_filename = f'logs/interactive_investment_dashboard_{timestamp}.html'
        fig.write_html(html_filename)
        
        print(f"📊 Interactive Dashboard saved as: {html_filename}")
        print(f"🌐 Open the HTML file in your browser to view the interactive dashboard!")
        
        return html_filename

    def _add_investment_intervals_to_plot(self, fig, days):
        """Add investment intervals and decision points to the main plot"""
        
        # Process both models
        for model_name, color in [('rainbow', '#FF6B6B'), ('martingale', '#4ECDC4')]:
            if not (self.results[model_name]['stopping_times'] and self.results[model_name]['prices']):
                continue
                
            stopping_times = self.results[model_name]['stopping_times']
            all_prices = self.results[model_name]['prices']
            
            # Simple approach: map episodes directly to test data timeline
            investment_days = []
            investment_prices = []
            interval_shapes = []
            
            # Each episode maps to a segment of test data
            price_idx = 0
            
            for episode_idx, stop_time in enumerate(stopping_times):
                if episode_idx >= len(self.test_data) // self.cycle_T:
                    break  # Don't exceed available test data
                    
                # Calculate episode boundaries in test data
                episode_start = episode_idx * self.cycle_T
                episode_end = min(episode_start + self.cycle_T - 1, len(self.test_data) - 1)
                
                # Get episode prices from test data
                episode_prices = self.test_data[episode_start:episode_end + 1]
                
                if len(episode_prices) > stop_time:
                    # Investment decision point
                    investment_day = episode_start + stop_time
                    investment_price = episode_prices[stop_time]
                    
                    investment_days.append(investment_day)
                    investment_prices.append(investment_price)
                    
                    # Create interval rectangle (min to max price in episode)
                    min_price = min(episode_prices)
                    max_price = max(episode_prices)
                    
                    # Add interval rectangle
                    interval_shapes.append(
                        dict(
                            type="rect",
                            x0=episode_start,
                            x1=episode_end,
                            y0=min_price,
                            y1=max_price,
                            fillcolor=f'rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.15)',
                            line=dict(color=color, width=1, dash='dot'),
                            layer="below"
                        )
                    )
            
            # Add investment decision points
            if investment_days and investment_prices:
                fig.add_trace(
                    go.Scatter(
                        x=investment_days,
                        y=investment_prices,
                        mode='markers',
                        name=f'{model_name.title()} DQN Investments ({len(investment_days)})',
                        marker=dict(
                            color=color,
                            size=12,
                            symbol='circle',
                            line=dict(color='white', width=2)
                        ),
                        hovertemplate=f'{model_name.title()} Investment<br>Day: %{{x}}<br>Price: $%{{y:,.0f}}<br>Episode: {episode_idx}<br>Stop Time: {stop_time}<extra></extra>'
                    ),
                    row=1, col=1
                )
            
            # Add interval shapes to figure
            for shape in interval_shapes:
                fig.add_shape(shape, row=1, col=1)

    def _calculate_price_intervals(self, model_type):
        """Calculate price intervals for each episode"""
        intervals = []
        prices = self.results[model_type]['prices']
        stopping_times = self.results[model_type]['stopping_times']
        
        price_idx = 0
        for stop_time in stopping_times:
            episode_prices = prices[price_idx:price_idx + stop_time + 1]
            if len(episode_prices) > 1:
                interval = max(episode_prices) - min(episode_prices)
                intervals.append(interval)
            price_idx += stop_time + 1
        
        return intervals

    def _calculate_convergence(self, scores, window=100):
        """Calculate convergence measure (variance over time)"""
        convergence = []
        for i in range(window, len(scores), window):
            variance = np.var(scores[i-window:i])
            convergence.append(variance)
        return convergence

    def _generate_summary_stats(self):
        """Generate comprehensive summary statistics"""
        summary = "FULL TRAINING COMPARISON\n"
        summary += "=" * 25 + "\n\n"
        
        for model in ['rainbow', 'martingale']:
            summary += f"{model.upper()} DQN:\n"
            if self.results[model]['rewards']:
                rewards = self.results[model]['rewards']
                summary += f"  Rewards: μ={np.mean(rewards):.4f}, σ={np.std(rewards):.4f}\n"
                summary += f"  Median: {np.median(rewards):.4f}\n"
            if self.results[model]['stopping_times']:
                stops = [t + 1 for t in self.results[model]['stopping_times']]  # Convert to 1-9
                summary += f"  Decision Days: μ={np.mean(stops):.2f}, σ={np.std(stops):.2f}\n"
                summary += f"  Most Common Day: {stats.mode(stops)[0][0]} (Day {stats.mode(stops)[0][0]})\n"
                summary += f"  Early Decisions (Days 1-3): {np.mean(np.array(stops) <= 3)*100:.1f}%\n"
                summary += f"  Late Decisions (Days 7-9): {np.mean(np.array(stops) >= 7)*100:.1f}%\n"
            summary += "\n"
        
        # Performance comparison
        if self.results['rainbow']['rewards'] and self.results['martingale']['rewards']:
            improvement = (np.mean(self.results['martingale']['rewards']) - 
                          np.mean(self.results['rainbow']['rewards'])) / np.mean(self.results['rainbow']['rewards']) * 100
            summary += f"Martingale Improvement: {improvement:.2f}%\n"
            
            # Win rate
            rainbow_rewards = np.array(self.results['rainbow']['rewards'])
            martingale_rewards = np.array(self.results['martingale']['rewards'])
            win_rate = np.mean(martingale_rewards > rainbow_rewards) * 100
            summary += f"Martingale Win Rate: {win_rate:.1f}%\n"
            
            # Decision timing comparison
            if self.results['rainbow']['stopping_times'] and self.results['martingale']['stopping_times']:
                rainbow_stops = [t + 1 for t in self.results['rainbow']['stopping_times']]
                martingale_stops = [t + 1 for t in self.results['martingale']['stopping_times']]
                
                # Statistical test for decision timing differences
                from scipy.stats import mannwhitneyu
                try:
                    statistic, p_value = mannwhitneyu(rainbow_stops, martingale_stops, alternative='two-sided')
                    summary += f"\nDecision Timing Analysis:\n"
                    summary += f"  Mann-Whitney U test p-value: {p_value:.4f}\n"
                    if p_value < 0.05:
                        summary += f"  ✅ Significant difference in decision timing\n"
                    else:
                        summary += f"  ❌ No significant difference in decision timing\n"
                except Exception as e:
                    summary += f"\nDecision Timing Analysis: Error - {e}\n"
        
        return summary
    
    def _evaluate_random_agent(self, test_env, max_episodes):
        """Evaluate random agent baseline"""
        # Используем разный seed каждый раз для НАСТОЯЩЕЙ случайности
        import time
        np.random.seed(int(time.time() * 1000) % 2**32)  # Истинно случайный seed
        
        # Unwrap the environment to get the actual CryptoEnv
        unwrapped_env = test_env.unwrapped
        
        # Get all episodes from environment
        ev_episodes = unwrapped_env.prepare_episodes()
        original_episodes = unwrapped_env.prepare_original_episodes()
        
        # Limit episodes
        num_episodes = min(max_episodes, len(ev_episodes))
        
        rewards = []
        stopping_times = []
        prices = []
        actions = []
        
        for e in range(num_episodes):
            episode = ev_episodes[e]
            original_episode = original_episodes[e]
            
            # Random stopping time (0 to cycle_T-1)
            stop_time = np.random.randint(0, self.cycle_T)
            
            # Get episode prices
            episode_prices = [original_episode[t][-1] for t in range(len(original_episode))]
            
            if len(episode_prices) > stop_time:
                # ✅ ИСПРАВЛЯЕМ: используем ТУ ЖЕ нормализацию что и в обучении!
                # Нормализуем цены всего эпизода как в CryptoEnv.getPriceList()
                from RL_Environment.CryptoEnv import normalization
                normalized_price_list = normalization(episode_prices)
                
                episode_reward_sum = 0
                
                import math
                for t in range(stop_time + 1):  # До момента остановки включительно
                    # ✅ ИСПОЛЬЗУЕМ ТОТ ЖЕ REWARD ЧТО И В ОБУЧЕНИИ!
                    # Берем нормализованную цену из normalized_price_list (как в CryptoEnv.step)
                    current_normalized_price = normalized_price_list[t]
                    
                    # Защита от крайних значений (как в CryptoEnv.step)
                    if current_normalized_price == 0:
                        current_normalized_price += 0.001
                    if current_normalized_price == 1:
                        current_normalized_price -= 0.001
                    
                    if t == stop_time:  # Остановка
                        step_reward = math.log((1 - current_normalized_price) / current_normalized_price)
                    else:  # Ожидание
                        step_reward = -0.5 * math.log((1 - current_normalized_price) / current_normalized_price)
                    
                    episode_reward_sum += step_reward
                
                # ✅ ИСПОЛЬЗУЕМ ТОЛЬКО REWARD ЗА ДЕЙСТВИЕ ОСТАНОВКИ (как у других агентов)
                # Берем reward за последний шаг (остановку)
                final_normalized_price = normalized_price_list[stop_time]
                if final_normalized_price == 0:
                    final_normalized_price += 0.001
                if final_normalized_price == 1:
                    final_normalized_price -= 0.001
                
                stopping_reward = math.log((1 - final_normalized_price) / final_normalized_price)
                rewards.append(stopping_reward)  # Только reward за остановку!
                stopping_times.append(stop_time)
                actions.extend([0] * stop_time + [1])  # 0 for wait, 1 for stop
                prices.extend(episode_prices[:stop_time + 1])
        
        print(f"📈 RANDOM Results:")
        print(f"   Average Reward: {np.mean(rewards):.4f} ± {np.std(rewards):.4f}")
        print(f"   Average Stopping Time: {np.mean(stopping_times):.2f} ± {np.std(stopping_times):.2f}")
        print(f"   Stop Rate: {np.mean([1 if a == 1 else 0 for a in actions]):.3f}")
        
        return {
            'rewards': rewards,
            'stopping_times': stopping_times,
            'prices': prices,
            'actions': actions
        }
    
    def _calculate_investment_profits(self, investment_amount=10000):
        """Calculate real investment profits for all agents"""
        print("\n💰 === CALCULATING INVESTMENT PROFITS ===")
        
        # Use test data for profit calculation
        test_episodes = min(len(self.results['rainbow']['stopping_times']), 
                           len(self.results['martingale']['stopping_times']),
                           len(self.results['random']['stopping_times']))
        
        # Final BTC price (end of test period)
        final_btc_price = self.test_data[-1]
        start_btc_price = self.test_data[0]
        
        print(f"📊 Analyzing {test_episodes} investment episodes")
        print(f"💰 Investment amount per episode: ${investment_amount:,}")
        print(f"📈 Test period BTC change: {(final_btc_price - start_btc_price) / start_btc_price * 100:+.1f}%")
        
        profit_results = {
            'rainbow': {'investments': [], 'total_profit': 0, 'total_invested': 0},
            'martingale': {'investments': [], 'total_profit': 0, 'total_invested': 0},
            'iqn': {'investments': [], 'total_profit': 0, 'total_invested': 0},
            'random': {'investments': [], 'total_profit': 0, 'total_invested': 0}
        }
        
        for episode in range(test_episodes):
            episode_start = episode * self.cycle_T
            episode_end = min(episode_start + self.cycle_T, len(self.test_data))
            episode_prices = self.test_data[episode_start:episode_end]
            
            if len(episode_prices) < self.cycle_T:
                continue
                
            for agent_name in ['rainbow', 'martingale', 'iqn', 'random']:
                if episode >= len(self.results[agent_name]['stopping_times']):
                    continue
                    
                stop_time = self.results[agent_name]['stopping_times'][episode]
                
                if stop_time < len(episode_prices):
                    # Investment price
                    investment_price = episode_prices[stop_time]
                    
                    # BTC acquired
                    btc_acquired = investment_amount / investment_price
                    
                    # Final value (if held to end of test period)
                    final_value = btc_acquired * final_btc_price
                    profit = final_value - investment_amount
                    
                    # Episode end value (if sold at end of episode)
                    episode_end_price = episode_prices[-1]
                    episode_end_value = btc_acquired * episode_end_price
                    episode_profit = episode_end_value - investment_amount
                    
                    investment_data = {
                        'episode': episode,
                        'stop_day': stop_time + 1,  # 1-indexed
                        'investment_price': investment_price,
                        'btc_acquired': btc_acquired,
                        'final_value': final_value,
                        'profit': profit,
                        'episode_profit': episode_profit,
                        'roi': (episode_profit / investment_amount) * 100  # ROI на основе episode_profit
                    }
                    
                    profit_results[agent_name]['investments'].append(investment_data)
                    profit_results[agent_name]['total_profit'] += episode_profit  # Используем episode_profit вместо profit
                    profit_results[agent_name]['total_invested'] += investment_amount
        
        # Print summary
        print(f"\n📊 PROFIT ANALYSIS SUMMARY:")
        for agent_name in ['rainbow', 'martingale', 'iqn', 'random']:
            investments = profit_results[agent_name]['investments']
            if investments:
                total_profit = profit_results[agent_name]['total_profit']
                total_invested = profit_results[agent_name]['total_invested']
                avg_roi = np.mean([inv['roi'] for inv in investments])
                
                print(f"\n{agent_name.upper()} AGENT:")
                print(f"   Total Invested: ${total_invested:,.0f}")
                print(f"   Total Profit: ${total_profit:,.0f}")
                print(f"   ROI: {(total_profit/total_invested)*100:+.1f}%")
                print(f"   Average ROI per investment: {avg_roi:+.1f}%")
                print(f"   Investments: {len(investments)}")
        
        return profit_results

    def _add_investment_intervals_to_plot(self, fig, days):
        """Add investment intervals as box plots and decision markers to the Bitcoin price chart"""
        if not (self.results['rainbow']['stopping_times'] and 
                self.results['martingale']['stopping_times'] and 
                self.results['random']['stopping_times']):
            return
        
        # Colors for decision markers
        rainbow_color = '#FF6B6B'
        martingale_color = '#4ECDC4'
        iqn_color = 'purple'
        random_color = '#95A5A6'
        box_color = 'rgba(100, 100, 100, 0.6)'
        box_line_color = 'rgba(50, 50, 50, 0.8)'
        
        # Add investment intervals as box plots (9-day windows)
        num_intervals = min(50, len(self.results['rainbow']['stopping_times']))  # Show first 50 intervals
        
        # Prepare data for box plots
        box_x = []
        box_y_min = []
        box_y_max = []
        box_x_start = []
        box_x_end = []
        
        for i in range(num_intervals):
            start_day = i * self.cycle_T
            end_day = min(start_day + self.cycle_T - 1, len(self.test_data) - 1)
            
            if start_day < len(self.test_data) and end_day < len(self.test_data):
                # Get price data for this interval
                interval_prices = self.test_data[start_day:end_day+1]
                if len(interval_prices) > 0:
                    min_price = min(interval_prices)
                    max_price = max(interval_prices)
                    
                    # Add rectangle for this interval (box plot style)
                    fig.add_shape(
                        type="rect",
                        x0=start_day, y0=min_price,
                        x1=end_day, y1=max_price,
                        fillcolor=box_color,
                        line=dict(color=box_line_color, width=1),
                        opacity=0.6,
                        layer="below",
                        row=1, col=1
                    )
                    
        
        # Get stopping times for each model
        rainbow_stops = self.results['rainbow']['stopping_times']
        martingale_stops = self.results['martingale']['stopping_times']
        iqn_stops = self.results['iqn']['stopping_times'] if self.results['iqn']['stopping_times'] else []
        random_stops = self.results['random']['stopping_times']
        
        # Add decision markers for each model
        models = [
            ('Rainbow DQN', rainbow_stops, rainbow_color, 'diamond'),
            ('Martingale DQN', martingale_stops, martingale_color, 'square'),
            ('Random Agent', random_stops, random_color, 'circle')
        ]
        
        # Add IQN if available
        if iqn_stops:
            models.append(('IQN', iqn_stops, iqn_color, 'star'))
        
        for model_name, stops, color, symbol in models:
            if stops:
                # Convert stopping times to actual days in test period
                decision_days = []
                decision_prices = []
                
                for i, stop_time in enumerate(stops[:num_intervals]):
                    if i * self.cycle_T + stop_time < len(self.test_data):
                        day_idx = i * self.cycle_T + stop_time
                        decision_days.append(day_idx)
                        decision_prices.append(self.test_data[day_idx])
                
                if decision_days:
                    fig.add_trace(
                        go.Scatter(
                            x=decision_days,
                            y=decision_prices,
                            mode='markers',
                            name=f'{model_name} Decisions',
                            marker=dict(
                                color=color,
                                size=12,
                                symbol=symbol,
                                line=dict(width=2, color='white')
                            ),
                            hovertemplate=f'{model_name}<br>Day: %{{x}}<br>Price: $%{{y:,.0f}}<br>Interval: %{{customdata}}<extra></extra>',
                            customdata=[f"{i+1}" for i in range(len(decision_days))],
                            showlegend=True
                        ),
                        row=1, col=1
                    )

    def save_results(self):
        """Save comprehensive results"""
        os.makedirs('logs', exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save as pickle
        results_file = f'logs/full_training_results_{timestamp}.pkl'
        with open(results_file, 'wb') as f:
            pickle.dump(self.results, f)
        
        # Save as CSV
        csv_data = []
        for model in ['rainbow', 'martingale']:
            for i, reward in enumerate(self.results[model]['rewards']):
                csv_data.append({
                    'Model': model.title(),
                    'Episode': i,
                    'Reward': reward,
                    'Stopping_Time': self.results[model]['stopping_times'][i] if i < len(self.results[model]['stopping_times']) else None
                })
        
        df = pd.DataFrame(csv_data)
        csv_file = f'logs/full_training_results_{timestamp}.csv'
        df.to_csv(csv_file, index=False)
        
        print(f"💾 Results saved:")
        print(f"   📦 {results_file}")
        print(f"   📊 {csv_file}")


def dashboard_only_mode():
    """Dashboard-only mode: load existing models and generate dashboard"""
    print("📊 === DASHBOARD ONLY MODE ===")
    print(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Parameters
    data_path = 'Back-testing/Data.pkl'
    wnd_t = 30
    cycle_T = 9
    name_num = 0
    
    print(f"🔧 Configuration:")
    print(f"   🪟 Window size: {wnd_t}")
    print(f"   🔄 Cycle length: {cycle_T}")
    print(f"   📊 Mode: Dashboard Only (no training)")
    
    # Create comparison object
    comparison = FullTrainingComparison(data_path, wnd_t, cycle_T, name_num)
    
    # Check if models exist
    rainbow_model_path = './models/Rainbow_full_training/model.pth'
    martingale_model_path = './models/Martingale_full_training/model.pth'
    
    if not (os.path.exists(rainbow_model_path) and os.path.exists(martingale_model_path)):
        print("❌ ERROR: Trained models not found!")
        print(f"   Expected: {rainbow_model_path}")
        print(f"   Expected: {martingale_model_path}")
        print("   Please run full training first.")
        return
    
    print("✅ Found existing trained models")
    
    # Попробуем загрузить сохраненные результаты обучения
    print("\n📂 Loading training results...")
    training_results_files = []
    # Ищем сначала в graphics/ (там есть training data), потом в logs/ (новые), потом в корне (совсем старые)
    for search_dir in ['graphics', 'logs', '.']:
        if os.path.exists(search_dir):
            files = [os.path.join(search_dir, f) for f in os.listdir(search_dir) 
                    if f.startswith('full_training_results_') and f.endswith('.pkl')]
            training_results_files.extend(files)
    if training_results_files:
        # Приоритет: graphics/ файлы (там есть training data), потом остальные
        graphics_files = [f for f in training_results_files if f.startswith('graphics/')]
        if graphics_files:
            latest_results_file = sorted(graphics_files)[-1]
        else:
            latest_results_file = sorted(training_results_files)[-1]
        print(f"📁 Found training results: {latest_results_file}")
        try:
            with open(latest_results_file, 'rb') as f:
                saved_results = pickle.load(f)
            # Копируем данные обучения
            if 'rainbow' in saved_results and 'scores' in saved_results['rainbow']:
                comparison.results['rainbow']['scores'] = saved_results['rainbow'].get('scores', [])
                comparison.results['rainbow']['losses'] = saved_results['rainbow'].get('losses', [])
            if 'martingale' in saved_results and 'scores' in saved_results['martingale']:
                comparison.results['martingale']['scores'] = saved_results['martingale'].get('scores', [])
                comparison.results['martingale']['losses'] = saved_results['martingale'].get('losses', [])
            print(f"✅ Loaded training data: Rainbow scores={len(comparison.results['rainbow']['scores'])}, Martingale scores={len(comparison.results['martingale']['scores'])}")
        except Exception as e:
            print(f"⚠️ Could not load training results: {e}")
    else:
        print("⚠️ No saved training results found")
    
    # Evaluate on test set
    print("\n" + "="*60)
    comparison.evaluate_models(test_episodes=200)
    
    # Generate comprehensive plots
    print("\n" + "="*60)
    comparison.generate_comprehensive_plots()
    
    print(f"\n✅ === DASHBOARD GENERATION COMPLETED ===")
    print(f"📅 Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


def main():
    """Main function for full training and comparison"""
    # Check command line arguments for dashboard-only mode
    if len(sys.argv) > 1 and sys.argv[1] == '--dashboard-only':
        dashboard_only_mode()
        return
    
    print("🚀 === FULL TRAINING COMPARISON: MARTINGALE DQN vs RAINBOW DQN ===")
    print(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n📝 Available modes:")
    print("   1. Full training + evaluation + dashboard (default)")
    print("   2. Dashboard only: python full_training_comparison.py --dashboard-only")
    
    # Parameters
    data_path = 'Back-testing/Data.pkl'
    wnd_t = 30
    cycle_T = 9
    name_num = 0  # 0 for BTC, 1 for ETH
    num_frames = 100000  # Full training for better convergence
    seed = 777
    
    print(f"🔧 Configuration:")
    print(f"   📊 Training frames: {num_frames}")
    print(f"   🪟 Window size: {wnd_t}")
    print(f"   🔄 Cycle length: {cycle_T}")
    print(f"   🌱 Seed: {seed}")
    
    # Create comparison object
    comparison = FullTrainingComparison(data_path, wnd_t, cycle_T, name_num)
    
    # Run full training
    print("\n" + "="*60)
    comparison.train_models(num_frames=num_frames, seed=seed)
    
    # Evaluate on test set
    print("\n" + "="*60)
    comparison.evaluate_models(test_episodes=200)
    
    # Generate comprehensive plots
    print("\n" + "="*60)
    comparison.generate_comprehensive_plots()
    
    # Save results
    print("\n" + "="*60)
    comparison.save_results()
    
    print(f"\n✅ === FULL COMPARISON COMPLETED ===")
    print(f"📅 Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
