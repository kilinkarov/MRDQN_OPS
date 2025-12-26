"""Cryptocurrency trading environment for RL."""

import math

import gymnasium as gym
import numpy as np

from crypto_dqn_ops.utils.helpers import normalization, sigmoid


class CryptoEnv(gym.Env):
    """An OpenAI Gym Style RL Environment for Optimal Stopping Problem with DCA."""

    def __init__(self, data, wnd_t=15, cycle_T=30):
        """Initialize environment.

        Args:
            data: Price data list
            wnd_t: Window size for historical data
            cycle_T: Investment cycle length
        """
        self.data = data
        self.wnd_t = wnd_t
        self.cycle_T = cycle_T
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(
            low=0, high=np.finfo(np.float32).max, shape=(1, self.wnd_t + 2), dtype=np.float32
        )
        self.max_num_observations, self.original_wnd_lists, self.wnd_lists = self._prepare_data(
            self.data
        )
        self.num_episodes = self.get_total_num_episodes_per_epoch()
        self.begin_time = 0
        arr = np.arange(self.num_episodes)
        np.random.shuffle(arr)
        self.randomIndexes = arr.tolist()
        self.random_index = 0
        self.reset()
        (
            self.original_price_list,
            self.normalized_price_list,
            self.refer_price,
        ) = self.get_price_list()
        self.buy_time = cycle_T - 1

    def _prepare_data(self, data):
        """Prepare data windows.

        Args:
            data: price list

        Returns:
            no. of wnd sequences, all wnd sequences, and all normalized wnd sequences
        """
        original_wnd_lists = []
        if self.wnd_t > len(data):
            raise ValueError(f"data must be longer than wnd_t. The length of data is: {len(data)}")
        max_num_observations = len(data) - self.wnd_t
        if self.cycle_T > max_num_observations:
            raise ValueError(
                f"cycle_T must be longer than max number of obs. "
                f"max_num_observations is: {max_num_observations}"
            )
        original_wnd_lists.append([data[i : i + self.wnd_t] for i in range(max_num_observations)])
        original_wnd_lists = original_wnd_lists[0]
        wnd_lists = [normalization(i) for i in original_wnd_lists]

        return max_num_observations, original_wnd_lists, wnd_lists

    def get_total_num_episodes_per_epoch(self):
        """Get total number of episodes per epoch."""
        total = 0
        total += self.max_num_observations - self.cycle_T
        return total

    def prepare_episodes(self):
        """Prepare all episodes."""
        episodes = []
        for begin_time in range(self.max_num_observations - self.cycle_T):
            episode = self.wnd_lists[begin_time : begin_time + self.cycle_T]
            episodes.append(episode)
        return episodes

    def prepare_original_episodes(self):
        """Prepare original episodes without normalization."""
        episodes = []
        for begin_time in range(self.max_num_observations - self.cycle_T):
            episode = self.original_wnd_lists[begin_time : begin_time + self.cycle_T]
            episodes.append(episode)
        return episodes

    def get_price_list(self):
        """Get price list for current episode."""
        original_observations = self.original_wnd_lists[
            self.begin_time : self.begin_time + self.cycle_T
        ]
        original_price_list = []
        for observation in original_observations:
            original_price_list.append(observation[-1])
        normalized_price_list = normalization(original_price_list)
        refer_value = original_observations[0][-1]
        return original_price_list, normalized_price_list, refer_value

    def reset(self, seed=None, options=None):
        """Reset environment to initial state."""
        if seed is not None:
            np.random.seed(seed)

        self.current_step = 0
        self.buy_time = self.cycle_T - 1
        self.done = False
        self.begin_time = self.randomIndexes[self.random_index]
        self.random_index = (self.random_index + 1) % self.num_episodes
        self.observations = self.wnd_lists[self.begin_time : self.begin_time + self.cycle_T]
        (
            self.original_price_list,
            self.normalized_price_list,
            self.refer_price,
        ) = self.get_price_list()

        price_diff = self.original_price_list[self.current_step] - self.refer_price
        prices_so_far = self.original_price_list[: self.current_step + 1]
        typical_range = (
            max(prices_so_far) - min(prices_so_far)
            if len(prices_so_far) > 1
            else abs(self.refer_price * 0.1)
        )
        if typical_range > 0:
            normalized_diff = price_diff / typical_range
        else:
            normalized_diff = 0
        self.position_value = sigmoid(normalized_diff)

        self.remaining_time = (self.cycle_T - self.current_step) / self.cycle_T
        obs = np.concatenate(([self.position_value, self.remaining_time], self.observations[0]))
        return obs.astype(np.float32), {}

    def step(self, action):
        """Take action in environment."""
        reward = 0
        current_price = self.normalized_price_list[self.current_step]

        if (action == 1 or self.current_step == (self.cycle_T - 1)) and (not self.done):
            self.sell_time = self.current_step
            if current_price == 0:
                current_price += 0.001
            if current_price == 1:
                current_price -= 0.001
            reward = math.log((1 - current_price) / current_price)
            self.done = True
        elif action == 0:
            if current_price == 0:
                current_price += 0.001
            if current_price == 1:
                current_price -= 0.001
            reward = -0.5 * math.log((1 - current_price) / current_price)

        if self.current_step < (self.cycle_T - 1):
            self.current_step += 1

        price_diff = self.original_price_list[self.current_step] - self.refer_price
        prices_so_far = self.original_price_list[: self.current_step + 1]
        typical_range = (
            max(prices_so_far) - min(prices_so_far)
            if len(prices_so_far) > 1
            else abs(self.refer_price * 0.1)
        )
        if typical_range > 0:
            normalized_diff = price_diff / typical_range
        else:
            normalized_diff = 0
        self.position_value = sigmoid(normalized_diff)
        self.remaining_time = (self.cycle_T - self.current_step) / self.cycle_T
        if self.done:
            obs = np.concatenate(([0, 0], [0] * self.wnd_t))
        else:
            obs = np.concatenate(
                ([self.position_value, self.remaining_time], self.observations[self.current_step])
            )
        info = {}
        return obs.astype(np.float32), reward, self.done, info

    def seed(self, seed=None):
        """Set random seed."""
        if seed is not None:
            np.random.seed(seed)
        return [seed]

    def render(self):
        """Render environment."""
        pass

    def close(self):
        """Close environment."""
        pass
