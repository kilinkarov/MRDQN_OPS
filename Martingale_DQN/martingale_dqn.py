import datetime
import math
import os
import random
import argparse
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle
import random
from collections import deque
from typing import Deque, Dict, List, Tuple
from IPython.display import clear_output
from torch.nn.utils import clip_grad_norm_
from rl_plotter.logger import Logger
import sys
sys.path.append('../Rainbow_DQN')
sys.path.append('../RL_Environment')
from segment_tree import MinSegmentTree, SumSegmentTree

# Register CryptoEnv programmatically
from CryptoEnv import CryptoEnv
gym.envs.register(
    id='CryptoEnv-v0',
    entry_point='CryptoEnv:CryptoEnv',
)

def sigmoid(inx):
    if inx >= 0:
        return 1.0 / (1 + np.exp(-inx))
    else:
        return np.exp(inx) / (1 + np.exp(inx))


class MartingaleReplayBuffer:
    """Enhanced replay buffer with martingale-based prioritization"""
    def __init__(
            self,
            obs_dim: int,
            size: int,
            batch_size: int = 32,
            n_step: int = 3,
            gamma: float = 0.95,
            martingale_weight: float = 0.3
    ):
        self.obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.next_obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size], dtype=np.float32)
        self.rews_buf = np.zeros([size], dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.martingale_values = np.zeros([size], dtype=np.float32)  # Store martingale values
        self.max_size, self.batch_size = size, batch_size
        self.ptr, self.size, = 0, 0
        self.martingale_weight = martingale_weight

        # for N-step Learning
        self.n_step_buffer = deque(maxlen=n_step)
        self.n_step = n_step
        self.gamma = gamma

    def store(
            self,
            obs: np.ndarray,
            act: np.ndarray,
            rew: float,
            next_obs: np.ndarray,
            done: bool,
            martingale_value: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]:
        transition = (obs, act, rew, next_obs, done, martingale_value)
        self.n_step_buffer.append(transition)

        # single step transition is not ready
        if len(self.n_step_buffer) < self.n_step:
            return ()

        # make a n-step transition
        rew, next_obs, done, martingale_value = self._get_n_step_info(
            self.n_step_buffer, self.gamma
        )
        obs, act = self.n_step_buffer[0][:2]

        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.martingale_values[self.ptr] = martingale_value
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

        return self.n_step_buffer[0]

    def sample_batch(self) -> Dict[str, np.ndarray]:
        # Enhanced sampling with martingale-based weighting
        if self.size < self.batch_size:
            idxs = np.arange(self.size)
        else:
            # Compute sampling probabilities based on martingale values
            martingale_priorities = np.abs(self.martingale_values[:self.size]) + 1e-6
            probabilities = martingale_priorities / np.sum(martingale_priorities)
            idxs = np.random.choice(self.size, size=self.batch_size, replace=False, p=probabilities)

        return dict(
            obs=self.obs_buf[idxs],
            next_obs=self.next_obs_buf[idxs],
            acts=self.acts_buf[idxs],
            rews=self.rews_buf[idxs],
            done=self.done_buf[idxs],
            martingale_values=self.martingale_values[idxs],
            indices=idxs,
        )

    def _get_n_step_info(
            self, n_step_buffer: Deque, gamma: float
    ) -> Tuple[np.int64, np.ndarray, bool, float]:
        """Return n step reward, next_obs, done, and accumulated martingale value."""
        rew, next_obs, done, martingale_value = n_step_buffer[-1][-4:]

        for transition in reversed(list(n_step_buffer)[:-1]):
            r, n_o, d, m_val = transition[-4:]
            rew = r + gamma * rew * (1 - d)
            martingale_value = m_val + gamma * martingale_value * (1 - d)
            next_obs, done = (n_o, d) if d else (next_obs, done)

        return rew, next_obs, done, martingale_value

    def __len__(self) -> int:
        return self.size


class MartingaleNetwork(nn.Module):
    """Neural network with martingale-enhanced architecture"""
    def __init__(
            self,
            in_dim: int,
            out_dim: int,
            atom_size: int,
            support: torch.Tensor,
            martingale_dim: int = 32
    ):
        super(MartingaleNetwork, self).__init__()
        
        self.support = support
        self.out_dim = out_dim
        self.atom_size = atom_size
        self.martingale_dim = martingale_dim

        # Main feature extraction (увеличиваем до 512 как в оригинале)
        self.feature_layer = nn.Sequential(
            nn.Linear(in_dim, 512),  # ✅ 512 hidden units
            nn.ReLU(),
            nn.Linear(512, 512),     # ✅ 512 hidden units
            nn.ReLU(),
        )

        # Martingale-specific layers
        self.martingale_layer = nn.Sequential(
            nn.Linear(512, martingale_dim),  # ✅ Вход от 512
            nn.Tanh(),
            nn.Linear(martingale_dim, martingale_dim),
            nn.Tanh(),
        )

        # Enhanced advantage layer with martingale information
        self.advantage_layer = nn.Sequential(
            nn.Linear(512 + martingale_dim, 512),  # ✅ 512 hidden units
            nn.ReLU(),
            nn.Linear(512, out_dim * atom_size)    # ✅ Выход от 512
        )

        # Enhanced value layer with martingale information
        self.value_layer = nn.Sequential(
            nn.Linear(512 + martingale_dim, 512),  # ✅ 512 hidden units
            nn.ReLU(),
            nn.Linear(512, atom_size)              # ✅ Выход от 512
        )

        # Martingale prediction head
        self.martingale_predictor = nn.Sequential(
            nn.Linear(martingale_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        dist = self.dist(x)
        q = torch.sum(dist * self.support, dim=2)
        return q

    def dist(self, x: torch.Tensor) -> torch.Tensor:
        """Get distribution for atoms with martingale enhancement."""
        feature = self.feature_layer(x)
        martingale_feature = self.martingale_layer(feature)
        
        # Combine features
        combined_feature = torch.cat([feature, martingale_feature], dim=-1)
        
        advantage = self.advantage_layer(combined_feature).view(
            -1, self.out_dim, self.atom_size
        )
        value = self.value_layer(combined_feature).view(-1, 1, self.atom_size)
        
        # Dueling architecture with martingale enhancement
        q_atoms = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        dist = F.softmax(q_atoms, dim=-1)
        dist = dist.clamp(min=1e-3)  # for avoiding nans
        
        return dist

    def get_martingale_value(self, x: torch.Tensor) -> torch.Tensor:
        """Compute martingale value for the given state."""
        feature = self.feature_layer(x)
        martingale_feature = self.martingale_layer(feature)
        return self.martingale_predictor(martingale_feature)


class MartingaleDQNAgent:
    """Martingale DQN Agent for Optimal Stopping Problems"""

    def __init__(
            self,
            env: gym.Env,
            memory_size: int,
            batch_size: int,
            target_update: int,
            gamma: float = 0.95,
            # Categorical DQN parameters
            v_min: float = 0.0,
            v_max: float = 20.0,
            atom_size: int = 51,
            # N-step Learning
            n_step: int = 3,
            # Martingale specific parameters
            martingale_weight: float = 0.3,
            martingale_lr: float = 6.25e-5,
    ):
        obs_dim = env.observation_space.shape[1]
        action_dim = env.action_space.n

        self.env = env
        self.batch_size = batch_size
        self.target_update = target_update
        self.gamma = gamma
        self.martingale_weight = martingale_weight

        # device: Apple Silicon GPU / NVIDIA GPU / CPU
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")  # 🍎 Apple Silicon GPU
            print(f"🚀 Using Apple Silicon GPU (MPS): {self.device}")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")  # NVIDIA GPU
            print(f"🚀 Using NVIDIA GPU (CUDA): {self.device}")
        else:
            self.device = torch.device("cpu")   # CPU fallback
            print(f"⚠️ Using CPU: {self.device}")
        print(f"🔧 Device selected: {self.device}")

        # Enhanced memory with martingale-based prioritization
        self.memory = MartingaleReplayBuffer(
            obs_dim, memory_size, batch_size, n_step=n_step, gamma=gamma,
            martingale_weight=martingale_weight
        )

        # Categorical DQN parameters
        self.v_min = v_min
        self.v_max = v_max
        self.atom_size = atom_size
        self.support = torch.linspace(
            self.v_min, self.v_max, self.atom_size
        ).to(self.device)

        # networks: dqn, dqn_target
        self.dqn = MartingaleNetwork(
            obs_dim, action_dim, self.atom_size, self.support
        ).to(self.device)
        self.dqn_target = MartingaleNetwork(
            obs_dim, action_dim, self.atom_size, self.support
        ).to(self.device)
        self.dqn_target.load_state_dict(self.dqn.state_dict())
        self.dqn_target.eval()

        # optimizer with unified learning rate (как у Rainbow)
        self.optimizer = optim.Adam(self.dqn.parameters(), lr=6.25e-5, eps=1.5e-4)

        # transition to store in memory
        self.transition = list()

        # mode: train / test
        self.is_test = False

        # Martingale tracking
        self.martingale_history = []

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select an action from the input state."""
        state_tensor = torch.FloatTensor(state).to(self.device)
        q_values = self.dqn(state_tensor)
        
        # Enhanced action selection with martingale information
        martingale_value = self.dqn.get_martingale_value(state_tensor).item()
        
        # Adjust exploration based on martingale value
        if not self.is_test:
            epsilon = max(0.01, 0.1 * (1 - abs(martingale_value)))
            if random.random() < epsilon:
                selected_action = random.randint(0, q_values.size(-1) - 1)
            else:
                selected_action = q_values.argmax().item()
        else:
            selected_action = q_values.argmax().item()
        
        selected_action = np.array(selected_action)

        if not self.is_test:
            self.transition = [state, selected_action, martingale_value]

        return selected_action

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.float64, bool]:
        """Take an action and return the response of the env."""
        next_state, reward, done, _ = self.env.step(action)

        if not self.is_test:
            martingale_value = self.transition[2]
            self.transition = self.transition[:2] + [reward, next_state, done, martingale_value]

            # Store transition with martingale value
            transition_stored = self.memory.store(*self.transition)

        return next_state, reward, done

    def update_model(self) -> torch.Tensor:
        """Update the model by gradient descent with martingale enhancement."""
        samples = self.memory.sample_batch()
        
        # Standard DQN loss
        elementwise_loss = self._compute_dqn_loss(samples, self.gamma)
        
        # Martingale regularization loss
        martingale_loss = self._compute_martingale_loss(samples)
        
        # Combined loss
        total_loss = torch.mean(elementwise_loss) + self.martingale_weight * martingale_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        clip_grad_norm_(self.dqn.parameters(), 10.0)
        self.optimizer.step()

        return total_loss.item()

    def _compute_martingale_loss(self, samples: Dict[str, np.ndarray]) -> torch.Tensor:
        """Compute martingale-specific loss to ensure martingale property."""
        device = self.device
        state = torch.FloatTensor(samples["obs"]).to(device)
        next_state = torch.FloatTensor(samples["next_obs"]).to(device)
        reward = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(device)
        done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)
        
        # Current martingale values
        current_martingale = self.dqn.get_martingale_value(state)
        
        # Next martingale values
        with torch.no_grad():
            next_martingale = self.dqn_target.get_martingale_value(next_state)
        
        # Martingale condition: M_t = E[M_{t+1} | F_t] for optimal stopping
        # We want: current_martingale ≈ gamma * next_martingale * (1 - done) + reward_component
        target_martingale = self.gamma * next_martingale * (1 - done) + reward
        
        martingale_loss = F.mse_loss(current_martingale, target_martingale)
        
        return martingale_loss

    def _compute_dqn_loss(self, samples: Dict[str, np.ndarray], gamma: float) -> torch.Tensor:
        """Return categorical dqn loss with martingale enhancement."""
        device = self.device
        state = torch.FloatTensor(samples["obs"]).to(device)
        next_state = torch.FloatTensor(samples["next_obs"]).to(device)
        action = torch.LongTensor(samples["acts"]).to(device)
        reward = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(device)
        done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)

        # Categorical DQN algorithm
        delta_z = float(self.v_max - self.v_min) / (self.atom_size - 1)

        with torch.no_grad():
            # Double DQN
            next_action = self.dqn(next_state).argmax(1)
            next_dist = self.dqn_target.dist(next_state)
            next_dist = next_dist[range(self.batch_size), next_action]

            t_z = reward + (1 - done) * gamma * self.support
            t_z = t_z.clamp(min=self.v_min, max=self.v_max)
            b = (t_z - self.v_min) / delta_z
            l = b.floor().long()
            u = b.ceil().long()

            offset = (
                torch.linspace(
                    0, (self.batch_size - 1) * self.atom_size, self.batch_size
                ).long()
                .unsqueeze(1)
                .expand(self.batch_size, self.atom_size)
                .to(self.device)
            )

            proj_dist = torch.zeros(next_dist.size(), device=self.device)
            proj_dist.view(-1).index_add_(
                0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1)
            )
            proj_dist.view(-1).index_add_(
                0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1)
            ) 

        dist = self.dqn.dist(state)
        log_p = torch.log(dist[range(self.batch_size), action])
        elementwise_loss = -(proj_dist * log_p).sum(1)

        return elementwise_loss

    def train(self, logger, seed, num_frames: int, plotting_interval: int = 200):
        """Train the agent."""
        self.is_test = False

        state = self.env.reset()
        update_cnt = 0
        losses = []
        scores = []
        mean_scores = []
        martingale_losses = []
        score = 0

        for frame_idx in range(1, num_frames + 1):
            action = self.select_action(state)
            next_state, reward, done = self.step(action)

            state = next_state
            score += reward

            # if episode ends
            if done:
                state = self.env.reset()
                scores.append(score)
                if len(scores) >= 10:
                    logger.update(score=scores[-10:], total_steps=frame_idx)
                    mean_scores.append(np.mean(scores[-10:]))
                score = 0

            # if training is ready
            if len(self.memory) >= self.batch_size:
                loss = self.update_model()
                losses.append(loss)
                update_cnt += 1

                # if hard update is needed
                if update_cnt % self.target_update == 0:
                    self._target_hard_update()

            # saving models and plotting
            if frame_idx % plotting_interval == 0:
                self._plot(frame_idx, scores, losses, mean_scores)
            if frame_idx % 2000 == 0:
                print("Step: %d, Mean Score: %.2f" % (frame_idx, np.mean(np.array(mean_scores[-2:]) if len(mean_scores) >= 2 else [0])))
            if frame_idx % 1000 == 0:
                torch.save(self.dqn,
                   './models/%s_Martingale_%d_%d/Seed%d_Step_%dk/%d.pth' % (Expetiment_ID, wnd_t, cycle_T, seed, int(num_frames /1000), int(frame_idx/1000)))

        self.env.close()

    def test(self) -> Tuple[float, float]:
        """Single step test"""
        self.is_test = True
        naive_env = self.env
        state = self.env.reset()
        done = False
        score = 0

        while not done:
            action = self.select_action(state)
            next_state, reward, done = self.step(action)

            state = next_state
            score += reward
        price = 1 / (1 + np.exp(reward))
        self.env.close()
        # reset
        self.env = naive_env
        return price, score

    def _target_hard_update(self):
        """Hard update: target <- local."""
        self.dqn_target.load_state_dict(self.dqn.state_dict())

    def _plot(
            self,
            frame_idx: int,
            scores: List[float],
            losses: List[float],
            mean_scores: List[float]
    ):
        """Plot the training progresses."""
        clear_output(True)
        plt.figure(figsize=(15, 12))
        plt.title('Score v.s Episodes', fontsize=15)
        plt.plot(mean_scores, label='mean')
        plt.xlabel('Episodes', fontsize=14)
        plt.ylabel('Score', fontsize=14)
        plt.xticks(fontsize=13)
        plt.yticks(fontsize=13)
        plt.savefig('./figresults/%sMartingale_Score_Num_%d.jpg' % (Expetiment_ID, num_frames))

        plt.figure(figsize=(16, 9))
        plt.title('Loss v.s Steps', fontsize=15)
        plt.plot(losses)
        plt.xlabel('Steps', fontsize=14)
        plt.ylabel('Loss', fontsize=14)
        plt.xticks(fontsize=13)
        plt.yticks(fontsize=13)
        plt.savefig('./figresults/%sMartingale_Loss_Num_%d.jpg' % (Expetiment_ID, num_frames))


def dateAdd(date, interval=1):
    dt = datetime.datetime.strptime(date, "%Y%m%d")
    dt = dt + datetime.timedelta(interval)
    date1 = dt.strftime("%Y%m%d")
    return date1


def GetPriceList(content, name_num=0):
    """Obtain Price List.
    Args:
        content: .pkl file for crypto data
        name_num: 0 for BTC and 1 for ETH
    """
    price_list = []
    name_list = ['BTCBitcoin_price', 'ETHEthereum_price']
    desired = name_list[name_num]
    cnt = 0
    for name in content:
        if desired in name:
            if cnt == 0:
                start_date = name[0:8]
                cnt += 1
            price_list.append(content[name])
    return price_list, start_date


def seed_torch(seed):
    torch.manual_seed(seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def martingale_train(num_frames):
    seed_list = [888, 777, 999]  # set different random seeds
    for seed in seed_list:
        path = './models/%s_Martingale_%d_%d/Seed%d_Step_%dk/' % (Expetiment_ID, wnd_t, cycle_T, seed, int(num_frames /1000))
        if not os.path.exists(path):
            os.makedirs(path)
        logger = Logger(exp_name="Martingale", env_name=Expetiment_ID, seed=seed)
        np.random.seed(seed)
        random.seed(seed)
        seed_torch(seed)
        env.seed(seed)
        agent = MartingaleDQNAgent(env, memory_size, batch_size, target_update, gamma, v_min=v_min, v_max=v_max, atom_size=atom_size, n_step=n_step)
        agent.train(logger, seed, num_frames, plotting_interval=num_frames)


def martingale_evaluate(path, num=0, stat=136):
    """Evaluate the Martingale DQN agent"""
    F = open(data_path, 'rb')
    content = pickle.load(F)
    total_data, start_date = GetPriceList(content, name_num=num)
    test_data_list = [total_data[-stat-wnd_t:]]
    seed = 777
    np.random.seed(seed)
    random.seed(seed)
    seed_torch(seed)
    env.seed(seed)
    agent = MartingaleDQNAgent(env, memory_size, batch_size, target_update)
    agent.dqn = torch.load(path)
    agent.dqn.eval()

    for test_data in test_data_list:
        test_env = gym.make('CryptoEnv-v0', data=test_data, wnd_t=wnd_t, cycle_T=cycle_T)
        ev_episodes = test_env.prepare_episodes()
        original_episodes = test_env.prepare_original_episodes()
        e = 0
        random_list = []
        p_list = []
        first_day = []
        last_day = []
        avg_day = []
        ratio_first_day = []
        ratio_last_day = []
        ratio_random_day = []
        ratio_average_amount = []
        t_list = []
        visual_price = []
    
        for episode in ev_episodes:
            refer_value = original_episodes[e][0][-2]
            t = 0
            for state in episode:
                remain_t = (cycle_T - t) / cycle_T
                price = original_episodes[e][t][-1]
                position_value = sigmoid(price - refer_value)
                obs = (np.concatenate(([position_value, remain_t], state)))
                obs = obs.reshape(1, wnd_t + 2)

                action = agent.dqn(
                    torch.FloatTensor(obs).to('cuda')
                ).argmax()
                action = action.detach().cpu().numpy()

                if action == 1 or t == cycle_T - 1:
                    p_list.append(original_episodes[e][t][-1])
                    first_day.append(original_episodes[e][0][-1])
                    last_day.append(original_episodes[e][-1][-1])
                    random_t = random.randint(0, cycle_T - 1)
                    random_list.append(original_episodes[e][random_t][-1])
                    period_price_list = [original_episodes[e][i][-1] for i in range(cycle_T)]
                    avg_day.append(np.mean(np.array(period_price_list)))
                    if e % cycle_T == 0:
                        t_list.append(int(wnd_t + cycle_T * int(e // cycle_T) + t - 1))
                        visual_price.append(original_episodes[e][t][-1])
                    break
                t += 1
            e += 1
            p_amount = np.sum(np.array([10000 / i for i in p_list]))
            f_amount = np.sum(np.array([10000 / i for i in first_day]))
            l_amount = np.sum(np.array([10000 / i for i in last_day]))
            r_amount = np.sum(np.array([10000 / i for i in random_list]))
            avg_amount = np.sum(np.array([10000 / i for i in avg_day]))
            ratio_first_day.append((p_amount - f_amount) / f_amount * 100)
            ratio_last_day.append((p_amount - l_amount) / l_amount * 100)
            ratio_random_day.append((p_amount - r_amount) / r_amount * 100)
            ratio_average_amount.append((p_amount - avg_amount) / avg_amount * 100)
        
        result_list = [np.mean(np.array(ratio_first_day)), np.mean(np.array(ratio_last_day)), np.mean(np.array(ratio_random_day)), np.mean(np.array(ratio_average_amount))]
        print("Martingale DQN Results:")
        if len([i for i in result_list if i>0]) >= 3:
            print("Compared with always buy on the first day: %.2f %%" % np.mean(np.array(ratio_first_day)))
            print("Compared with always buy on the last day: %.2f %%" % np.mean(np.array(ratio_last_day)))
            print("Compared with always buy on a random day: %.2f %%" % np.mean(np.array(ratio_random_day)))
            print("Compared with buy on the average price: %.2f %%" % np.mean(np.array(ratio_average_amount)))
        else:
            print("none")


def load_price_data(name_num):
    """load original BTC/ETH price data."""
    data_file = open(data_path, 'rb')
    content = pickle.load(data_file)
    total_data, start_date = GetPriceList(content, name_num=name_num)
    if name_num == 0:
        start_index = 2560  # for BTC
    else:
        start_index = 1800  # for ETH
    useful_data = total_data[start_index:]
    train_data = useful_data[0:int(len(useful_data) * 0.85)]
    test_data = useful_data[-int(len(useful_data) * 0.15):]
    print("Total data date range: %s to %s, %d days" % (start_date, dateAdd(start_date, len(total_data)), len(total_data)))
    print("Train data date range: %s to %s, %d days" % (
    dateAdd(start_date, start_index), dateAdd(start_date, start_index + len(train_data)), len(train_data)))
    print("Test data date range: %s to %s %d days" % (
        dateAdd(start_date, start_index + len(train_data) + 1), dateAdd(start_date, start_index + len(train_data) + 1 + len(test_data)),
        len(test_data)))
    return useful_data, train_data, test_data


if __name__ == '__main__':
    # parameters   
    parser = argparse.ArgumentParser()
    parser.add_argument("--ExpID", type=str, default="BTC", help="Experiment ID")
    parser.add_argument("--frames", type=int, default=300 * 1000, help="number of frames")
    parser.add_argument("--name", type=int, default=0, help="which crypto, 0 for BTC and 1 for ETH")
    parser.add_argument("--wnd", type=int, default=30, help="window size")
    parser.add_argument("--cycle", type=int, default=9, help="investment cycle")
    parser.add_argument("--memory_size", type=int, default=10000, help="memory size for replay buffer")
    parser.add_argument("--batch_size", type=int, default=128, help="batch size for training")
    parser.add_argument("--target_update", type=int, default=100, help="Update the target network every target_update episodes")
    parser.add_argument("--gamma", type=int, default=0.95, help="gamma for Q learning")
    parser.add_argument("--v_min", type=int, default=0, help="v_min for C51")
    parser.add_argument("--v_max", type=int, default=20, help="v_max for C51")
    parser.add_argument("--atom_size", type=int, default=51, help="atom size of distributed RL")
    parser.add_argument("--n_step", type=int, default=3, help="multi-step learning")
    parser.add_argument("--data_path", type=str, default=r'../data/Data.pkl', help="path to the price data")
    parser.add_argument("--mode", type=int, default=0, help="mode 0: training, mode 1: evaluation")
    parser.add_argument("--martingale_weight", type=float, default=0.3, help="weight for martingale loss")

    args = parser.parse_args()
    num_frames = args.frames
    Expetiment_ID = args.ExpID
    wnd_t = args.wnd
    cycle_T = args.cycle
    name_num = args.name
    memory_size = args.memory_size
    batch_size = args.batch_size
    target_update = args.target_update
    gamma = args.gamma
    v_min = args.v_min
    v_max = args.v_max
    atom_size = args.atom_size
    n_step = args.n_step
    data_path = args.data_path
    running_mode = args.mode
    martingale_weight = args.martingale_weight

    # load original price data
    useful_data, train_data, test_data = load_price_data(name_num)

    # Create gym environment    
    env_id = 'CryptoEnv-v0'
    env = gym.make('CryptoEnv-v0', data=train_data, wnd_t=wnd_t, cycle_T=cycle_T)

    # training or evaluation
    if running_mode == 0:
        martingale_train(num_frames)
    elif running_mode == 1:
        for i in range(100, 102):
            path = './models/%s_Martingale_%d_%d/Seed777_Step_%dk/%s.pth' % (Expetiment_ID, wnd_t, cycle_T, int(num_frames/1000), str(i))
            martingale_evaluate(path=path, num=name_num, stat=127)
            print("Currently i = %d" % i)
            print(" ")
    else:
        print("wrong mode")

