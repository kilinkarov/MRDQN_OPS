import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import torch
import gymnasium as gym
import random
from datetime import datetime
import seaborn as sns

# Add paths for both models
sys.path.append('../Rainbow_DQN')
sys.path.append('../Martingale_DQN')
sys.path.append('../RL_Environment')

# Register CryptoEnv programmatically
from CryptoEnv import CryptoEnv
gym.envs.register(
    id='CryptoEnv-v0',
    entry_point='CryptoEnv:CryptoEnv',
)

from rainbow import DQNAgent as RainbowAgent, load_price_data, GetPriceList, seed_torch
from martingale_dqn import MartingaleDQNAgent, sigmoid

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class ModelComparison:
    def __init__(self, data_path, wnd_t=30, cycle_T=9, name_num=0):
        self.data_path = data_path
        self.wnd_t = wnd_t
        self.cycle_T = cycle_T
        self.name_num = name_num
        
        # Load data
        self.useful_data, self.train_data, self.test_data = load_price_data(name_num)
        
        # Create environment
        self.env = gym.make('CryptoEnv-v0', data=self.train_data, wnd_t=wnd_t, cycle_T=cycle_T)
        
        # Results storage
        self.results = {
            'rainbow': {'scores': [], 'losses': [], 'rewards': [], 'actions': [], 'stopping_times': []},
            'martingale': {'scores': [], 'losses': [], 'rewards': [], 'actions': [], 'stopping_times': []}
        }

    def train_and_compare(self, num_frames=30000, seeds=[777, 888, 999]):
        """Train both models and collect training metrics"""
        
        for seed in seeds:
            print(f"\n=== Training with seed {seed} ===")
            
            # Set random seeds
            np.random.seed(seed)
            random.seed(seed)
            seed_torch(seed)
            self.env.seed(seed)
            
            # Train Rainbow DQN
            print("Training Rainbow DQN...")
            rainbow_agent = RainbowAgent(
                self.env, memory_size=10000, batch_size=128, target_update=100,
                gamma=0.95, v_min=0, v_max=20, atom_size=51, n_step=3
            )
            
            rainbow_scores, rainbow_losses = self._train_agent(rainbow_agent, num_frames, "Rainbow", seed)
            self.results['rainbow']['scores'].extend(rainbow_scores)
            self.results['rainbow']['losses'].extend(rainbow_losses)
            
            # Reset environment and seeds
            np.random.seed(seed)
            random.seed(seed)
            seed_torch(seed)
            self.env.seed(seed)
            
            # Train Martingale DQN
            print("Training Martingale DQN...")
            martingale_agent = MartingaleDQNAgent(
                self.env, memory_size=10000, batch_size=128, target_update=100,
                gamma=0.95, v_min=0, v_max=20, atom_size=51, n_step=3,
                martingale_weight=0.3
            )
            
            martingale_scores, martingale_losses = self._train_agent(martingale_agent, num_frames, "Martingale", seed)
            self.results['martingale']['scores'].extend(martingale_scores)
            self.results['martingale']['losses'].extend(martingale_losses)

    def _train_agent(self, agent, num_frames, agent_type, seed):
        """Train a single agent and return metrics"""
        state = self.env.reset()
        scores = []
        losses = []
        score = 0
        update_cnt = 0
        
        for frame_idx in range(1, num_frames + 1):
            action = agent.select_action(state)
            next_state, reward, done = agent.step(action)
            
            state = next_state
            score += reward
            
            if done:
                state = self.env.reset()
                scores.append(score)
                score = 0
            
            # Update model
            if len(agent.memory) >= agent.batch_size:
                loss = agent.update_model()
                losses.append(loss)
                update_cnt += 1
                
                if update_cnt % agent.target_update == 0:
                    agent._target_hard_update()
            
            if frame_idx % 5000 == 0:
                mean_score = np.mean(scores[-10:]) if len(scores) >= 10 else 0
                print(f"{agent_type} - Step: {frame_idx}, Mean Score: {mean_score:.2f}")
        
        # Save model
        model_dir = f'./models/{agent_type}_seed_{seed}/'
        os.makedirs(model_dir, exist_ok=True)
        torch.save(agent.dqn, f'{model_dir}/model.pth')
        
        return scores, losses

    def evaluate_models(self, test_episodes=100):
        """Evaluate both models on test data"""
        print("\n=== Evaluating Models ===")
        
        # Load test data
        F = open(self.data_path, 'rb')
        content = pickle.load(F)
        total_data, start_date = GetPriceList(content, name_num=self.name_num)
        test_data_list = [total_data[-136-self.wnd_t:]]  # Use last 136 days for testing
        
        for model_type in ['rainbow', 'martingale']:
            print(f"\nEvaluating {model_type.upper()} DQN...")
            
            # Load model
            if model_type == 'rainbow':
                agent = RainbowAgent(
                    self.env, memory_size=10000, batch_size=128, target_update=100
                )
            else:
                agent = MartingaleDQNAgent(
                    self.env, memory_size=10000, batch_size=128, target_update=100
                )
            
            # Load trained model (using seed 777)
            model_path = f'./models/{model_type.title()}_seed_777/model.pth'
            if os.path.exists(model_path):
                agent.dqn = torch.load(model_path)
                agent.dqn.eval()
            else:
                print(f"Model not found: {model_path}")
                continue
            
            # Evaluate
            results = self._evaluate_agent(agent, test_data_list[0])
            
            self.results[model_type]['rewards'] = results['rewards']
            self.results[model_type]['actions'] = results['actions']
            self.results[model_type]['stopping_times'] = results['stopping_times']
            
            # Print results
            print(f"{model_type.upper()} Results:")
            print(f"  Average Reward: {np.mean(results['rewards']):.4f}")
            print(f"  Average Stopping Time: {np.mean(results['stopping_times']):.2f}")
            print(f"  Stop Rate: {np.mean([1 if a == 1 else 0 for a in results['actions']]):.2f}")

    def _evaluate_agent(self, agent, test_data):
        """Evaluate a single agent"""
        test_env = gym.make('CryptoEnv-v0', data=test_data, wnd_t=self.wnd_t, cycle_T=self.cycle_T)
        ev_episodes = test_env.prepare_episodes()
        original_episodes = test_env.prepare_original_episodes()
        
        rewards = []
        actions = []
        stopping_times = []
        
        for e, episode in enumerate(ev_episodes):
            if e >= 100:  # Limit evaluation episodes
                break
                
            refer_value = original_episodes[e][0][-2]
            episode_actions = []
            
            for t, state in enumerate(episode):
                remain_t = (self.cycle_T - t) / self.cycle_T
                price = original_episodes[e][t][-1]
                position_value = sigmoid(price - refer_value)
                obs = np.concatenate(([position_value, remain_t], state))
                obs = obs.reshape(1, self.wnd_t + 2)
                
                action = agent.dqn(torch.FloatTensor(obs).to(agent.device)).argmax()
                action = action.detach().cpu().numpy()
                episode_actions.append(action)
                
                if action == 1 or t == self.cycle_T - 1:
                    # Calculate reward based on stopping decision
                    current_price = original_episodes[e][t][-1]
                    min_price = min([original_episodes[e][i][-1] for i in range(self.cycle_T)])
                    max_price = max([original_episodes[e][i][-1] for i in range(self.cycle_T)])
                    
                    # Normalized reward: how close to optimal stopping
                    normalized_reward = (max_price - current_price) / (max_price - min_price) if max_price != min_price else 0
                    
                    rewards.append(normalized_reward)
                    stopping_times.append(t)
                    actions.extend(episode_actions)
                    break
        
        return {
            'rewards': rewards,
            'actions': actions,
            'stopping_times': stopping_times
        }

    def generate_comparison_plots(self):
        """Generate comprehensive comparison plots"""
        print("\n=== Generating Comparison Plots ===")
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Training Scores Comparison
        plt.subplot(3, 3, 1)
        if self.results['rainbow']['scores'] and self.results['martingale']['scores']:
            # Smooth the scores for better visualization
            rainbow_smooth = self._smooth_curve(self.results['rainbow']['scores'], window=50)
            martingale_smooth = self._smooth_curve(self.results['martingale']['scores'], window=50)
            
            plt.plot(rainbow_smooth, label='Rainbow DQN', linewidth=2, alpha=0.8)
            plt.plot(martingale_smooth, label='Martingale DQN', linewidth=2, alpha=0.8)
            plt.xlabel('Episodes')
            plt.ylabel('Score')
            plt.title('Training Scores Comparison')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # 2. Training Losses Comparison
        plt.subplot(3, 3, 2)
        if self.results['rainbow']['losses'] and self.results['martingale']['losses']:
            rainbow_loss_smooth = self._smooth_curve(self.results['rainbow']['losses'], window=100)
            martingale_loss_smooth = self._smooth_curve(self.results['martingale']['losses'], window=100)
            
            plt.plot(rainbow_loss_smooth, label='Rainbow DQN', linewidth=2, alpha=0.8)
            plt.plot(martingale_loss_smooth, label='Martingale DQN', linewidth=2, alpha=0.8)
            plt.xlabel('Training Steps')
            plt.ylabel('Loss')
            plt.title('Training Loss Comparison')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # 3. Reward Distribution
        plt.subplot(3, 3, 3)
        if self.results['rainbow']['rewards'] and self.results['martingale']['rewards']:
            plt.hist(self.results['rainbow']['rewards'], alpha=0.7, label='Rainbow DQN', bins=30, density=True)
            plt.hist(self.results['martingale']['rewards'], alpha=0.7, label='Martingale DQN', bins=30, density=True)
            plt.xlabel('Reward')
            plt.ylabel('Density')
            plt.title('Reward Distribution')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # 4. Stopping Times Distribution
        plt.subplot(3, 3, 4)
        if self.results['rainbow']['stopping_times'] and self.results['martingale']['stopping_times']:
            plt.hist(self.results['rainbow']['stopping_times'], alpha=0.7, label='Rainbow DQN', bins=range(self.cycle_T+1), density=True)
            plt.hist(self.results['martingale']['stopping_times'], alpha=0.7, label='Martingale DQN', bins=range(self.cycle_T+1), density=True)
            plt.xlabel('Stopping Time')
            plt.ylabel('Density')
            plt.title('Stopping Times Distribution')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # 5. Performance Metrics Comparison
        plt.subplot(3, 3, 5)
        metrics = ['Avg Reward', 'Avg Stop Time', 'Stop Rate']
        rainbow_metrics = [
            np.mean(self.results['rainbow']['rewards']) if self.results['rainbow']['rewards'] else 0,
            np.mean(self.results['rainbow']['stopping_times']) if self.results['rainbow']['stopping_times'] else 0,
            np.mean([1 if a == 1 else 0 for a in self.results['rainbow']['actions']]) if self.results['rainbow']['actions'] else 0
        ]
        martingale_metrics = [
            np.mean(self.results['martingale']['rewards']) if self.results['martingale']['rewards'] else 0,
            np.mean(self.results['martingale']['stopping_times']) if self.results['martingale']['stopping_times'] else 0,
            np.mean([1 if a == 1 else 0 for a in self.results['martingale']['actions']]) if self.results['martingale']['actions'] else 0
        ]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        plt.bar(x - width/2, rainbow_metrics, width, label='Rainbow DQN', alpha=0.8)
        plt.bar(x + width/2, martingale_metrics, width, label='Martingale DQN', alpha=0.8)
        plt.xlabel('Metrics')
        plt.ylabel('Value')
        plt.title('Performance Metrics Comparison')
        plt.xticks(x, metrics)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 6. Convergence Analysis
        plt.subplot(3, 3, 6)
        if self.results['rainbow']['scores'] and self.results['martingale']['scores']:
            rainbow_convergence = self._calculate_convergence(self.results['rainbow']['scores'])
            martingale_convergence = self._calculate_convergence(self.results['martingale']['scores'])
            
            plt.plot(rainbow_convergence, label='Rainbow DQN', linewidth=2)
            plt.plot(martingale_convergence, label='Martingale DQN', linewidth=2)
            plt.xlabel('Episodes (x100)')
            plt.ylabel('Score Variance')
            plt.title('Convergence Analysis')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # 7. Action Frequency Analysis
        plt.subplot(3, 3, 7)
        if self.results['rainbow']['actions'] and self.results['martingale']['actions']:
            rainbow_actions = np.array(self.results['rainbow']['actions'])
            martingale_actions = np.array(self.results['martingale']['actions'])
            
            actions = ['Hold (0)', 'Stop (1)']
            rainbow_freq = [np.mean(rainbow_actions == 0), np.mean(rainbow_actions == 1)]
            martingale_freq = [np.mean(martingale_actions == 0), np.mean(martingale_actions == 1)]
            
            x = np.arange(len(actions))
            plt.bar(x - width/2, rainbow_freq, width, label='Rainbow DQN', alpha=0.8)
            plt.bar(x + width/2, martingale_freq, width, label='Martingale DQN', alpha=0.8)
            plt.xlabel('Actions')
            plt.ylabel('Frequency')
            plt.title('Action Frequency Analysis')
            plt.xticks(x, actions)
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # 8. Statistical Significance Test
        plt.subplot(3, 3, 8)
        if self.results['rainbow']['rewards'] and self.results['martingale']['rewards']:
            from scipy import stats
            
            # Perform t-test
            t_stat, p_value = stats.ttest_ind(
                self.results['rainbow']['rewards'],
                self.results['martingale']['rewards']
            )
            
            plt.text(0.1, 0.7, f'T-statistic: {t_stat:.4f}', transform=plt.gca().transAxes, fontsize=12)
            plt.text(0.1, 0.6, f'P-value: {p_value:.4f}', transform=plt.gca().transAxes, fontsize=12)
            
            significance = "Significant" if p_value < 0.05 else "Not Significant"
            plt.text(0.1, 0.5, f'Result: {significance}', transform=plt.gca().transAxes, fontsize=12, 
                    color='green' if p_value < 0.05 else 'red')
            
            plt.text(0.1, 0.3, f'Martingale Mean: {np.mean(self.results["martingale"]["rewards"]):.4f}', 
                    transform=plt.gca().transAxes, fontsize=10)
            plt.text(0.1, 0.2, f'Rainbow Mean: {np.mean(self.results["rainbow"]["rewards"]):.4f}', 
                    transform=plt.gca().transAxes, fontsize=10)
            
            plt.title('Statistical Significance Test')
            plt.axis('off')
        
        # 9. Summary Statistics
        plt.subplot(3, 3, 9)
        summary_text = self._generate_summary_stats()
        plt.text(0.1, 0.9, summary_text, transform=plt.gca().transAxes, fontsize=10, 
                verticalalignment='top', fontfamily='monospace')
        plt.title('Summary Statistics')
        plt.axis('off')
        
        plt.tight_layout()
        
        # Save the plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f'./comparison_results_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'./comparison_results_{timestamp}.pdf', bbox_inches='tight')
        
        print(f"Comparison plots saved as comparison_results_{timestamp}.png/pdf")
        
        plt.show()

    def _smooth_curve(self, data, window=50):
        """Smooth curve using moving average"""
        if len(data) < window:
            return data
        return pd.Series(data).rolling(window=window, min_periods=1).mean().values

    def _calculate_convergence(self, scores, window=100):
        """Calculate convergence measure (variance over time)"""
        convergence = []
        for i in range(window, len(scores), window):
            variance = np.var(scores[i-window:i])
            convergence.append(variance)
        return convergence

    def _generate_summary_stats(self):
        """Generate summary statistics text"""
        summary = "MODEL COMPARISON SUMMARY\n"
        summary += "=" * 30 + "\n\n"
        
        for model in ['rainbow', 'martingale']:
            summary += f"{model.upper()} DQN:\n"
            if self.results[model]['rewards']:
                summary += f"  Rewards: μ={np.mean(self.results[model]['rewards']):.4f}, "
                summary += f"σ={np.std(self.results[model]['rewards']):.4f}\n"
            if self.results[model]['stopping_times']:
                summary += f"  Stop Times: μ={np.mean(self.results[model]['stopping_times']):.2f}, "
                summary += f"σ={np.std(self.results[model]['stopping_times']):.2f}\n"
            summary += "\n"
        
        # Performance improvement
        if self.results['rainbow']['rewards'] and self.results['martingale']['rewards']:
            improvement = (np.mean(self.results['martingale']['rewards']) - 
                          np.mean(self.results['rainbow']['rewards'])) / np.mean(self.results['rainbow']['rewards']) * 100
            summary += f"Martingale Improvement: {improvement:.2f}%\n"
        
        return summary

    def save_results(self):
        """Save results to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save as pickle
        with open(f'comparison_results_{timestamp}.pkl', 'wb') as f:
            pickle.dump(self.results, f)
        
        # Save as CSV
        df_results = pd.DataFrame({
            'Model': ['Rainbow'] * len(self.results['rainbow']['rewards']) + 
                     ['Martingale'] * len(self.results['martingale']['rewards']),
            'Rewards': self.results['rainbow']['rewards'] + self.results['martingale']['rewards'],
            'Stopping_Times': self.results['rainbow']['stopping_times'] + self.results['martingale']['stopping_times']
        })
        df_results.to_csv(f'comparison_results_{timestamp}.csv', index=False)
        
        print(f"Results saved as comparison_results_{timestamp}.pkl/csv")


def main():
    """Main comparison function"""
    print("=== Martingale DQN vs Rainbow DQN Comparison ===")
    
    # Parameters
    data_path = '../data/Data.pkl'
    wnd_t = 30
    cycle_T = 9
    name_num = 0  # 0 for BTC, 1 for ETH
    
    # Create comparison object
    comparison = ModelComparison(data_path, wnd_t, cycle_T, name_num)
    
    # Run comparison
    print("Starting training and comparison...")
    comparison.train_and_compare(num_frames=30000, seeds=[777])  # Reduced for faster testing
    
    print("Starting evaluation...")
    comparison.evaluate_models(test_episodes=100)
    
    print("Generating plots...")
    comparison.generate_comparison_plots()
    
    print("Saving results...")
    comparison.save_results()
    
    print("\n=== Comparison Complete ===")


if __name__ == "__main__":
    main()

