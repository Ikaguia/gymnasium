import matplotlib.pyplot  as plt
import numpy as np
import gymnasium as gym
from gymnasium import ObservationWrapper
from gymnasium.spaces import Box, Discrete, MultiDiscrete
import os
import glob
import json
from datetime import datetime

def plot(training_rewards, epsilons):
	fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
	if training_rewards:
		ax1.plot(training_rewards)
		ax1.title.set_text('Reward for epsodes')
		plt.ylabel('Training total reward')
	if epsilons:
		ax2.plot(epsilons)
		ax2.title.set_text("Epsilon for episodes")
		plt.ylabel('Epsilon')
	plt.xlabel('Episode')
	plt.show()

class ClampObservation(ObservationWrapper):
	def __init__(self, env, low, high):
		"""
		Args:
			env: Gymnasium environment.
			low: Lower bounds per dimension.
			high: Upper bounds per dimension.
		"""
		super().__init__(env)
		self.low = np.array(low)
		self.high = np.array(high)
		
		assert isinstance(env.observation_space, Box), "Only works with Box spaces"
		self.observation_space = Box(
			low=np.maximum(env.observation_space.low, self.low),
			high=np.minimum(env.observation_space.high, self.high),
			dtype=env.observation_space.dtype
		)

	def observation(self, observation):
		return np.clip(observation, self.low, self.high)

class DiscretizeObservation(ObservationWrapper):
	def __init__(self, env, bins, low, high):
		"""
		Args:
			env: Gymnasium environment.
			bins: List of bin counts per dimension.
			low: Lower bounds per dimension.
			high: Upper bounds per dimension.
		"""
		super().__init__(env)
		self.bins = np.array(bins)
		self.low = np.array(low)
		self.high = np.array(high)

		assert len(self.bins) == self.observation_space.shape[0], \
			"Number of bins must match observation dimensions."

		# Output space is MultiDiscrete (tuple of discrete values per dimension)
		self.observation_space = MultiDiscrete(self.bins)

	def observation(self, observation):
		ratios = (observation - self.low) / (self.high - self.low)
		ratios = np.clip(ratios, 0, 0.999)  # avoid edge case where ratio == 1
		discrete_obs = (ratios * self.bins).astype(int)
		return discrete_obs

class FlatDiscretizeObservation(ObservationWrapper):
	def __init__(self, env, bins, low, high):
		"""
		Args:
			env: Gymnasium environment.
			bins: List of bin counts per dimension.
			low: Lower bounds per dimension.
			high: Upper bounds per dimension.
		"""
		super().__init__(env)

		self.bins = np.array(bins)
		self.low = np.array(low)
		self.high = np.array(high)

		assert len(self.bins) == self.observation_space.shape[0], \
			"Number of bins must match observation dimensions."

		self.nvec = self.bins
		self.total_states = np.prod(self.bins)

		# Now the observation space is a simple Discrete space
		self.observation_space = Discrete(self.total_states)

	def observation(self, observation):
		# Discretize
		ratios = (observation - self.low) / (self.high - self.low)
		ratios = np.clip(ratios, 0, 0.999)
		discrete_obs = (ratios * self.bins).astype(int)

		# Flatten to a single integer index
		flat_index = np.ravel_multi_index(discrete_obs, self.bins)

		return flat_index
