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
	ax1.plot(training_rewards)
	ax1.title.set_text('Reward for epsodes')
	plt.ylabel('Training total reward')

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

def save_qtable(
	Q,
	params: dict,
	results: dict,
	save_dir="checkpoints",
	prefix="qtable",
	keep_best=5,
	log_filename="log.json",
	metric="avg_score"
):
	os.makedirs(save_dir, exist_ok=True)
	log_path = os.path.join(save_dir, log_filename)

	# ----- Load existing JSON log -----
	log_data = {}
	if os.path.exists(log_path):
		with open(log_path, "r") as f:
			log_data = json.load(f)

	# ----- Save latest -----
	latest_filename = f"{prefix}_latest.npy"
	np.save(os.path.join(save_dir, latest_filename), Q)

	current_entry = {
		"params": params,
		"results": results,
		"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
	}
	log_data[latest_filename] = current_entry

	print(f"✅ Saved latest Q-table to {os.path.join(save_dir, latest_filename)}")

	# ----- Evaluate best -----
	# Gather current bests
	best_entries = [
		(name, entry)
		for name, entry in log_data.items()
		if name.startswith(f"{prefix}_best")
	]

	# Add current as a candidate
	best_entries.append((latest_filename, current_entry))

	# Sort best entries by metric descending
	def extract_metric(entry): return entry["results"].get(metric, float("-inf"))
	best_entries_sorted = sorted(
		best_entries,
		key=lambda x: extract_metric(x[1]),
		reverse=True
	)[:keep_best]

	# ----- Save best Q-tables -----
	for idx, (name, entry) in reversed(list(enumerate(best_entries_sorted))):
		best_filename = f"{prefix}_best{idx + 1}.npy"
		src = os.path.join(save_dir, name)
		dst = os.path.join(save_dir, best_filename)
		# Save Q-table file
		if name == latest_filename:
			np.save(dst, Q)
			print(f"🏆 Saved latest Q-table to {dst}")
		elif name != best_filename:
			if os.path.exists(src):
				if os.path.exists(dst): os.remove(dst)
				os.rename(src, dst)
				print(f"🔄 Shifted {src} → {dst}")

		# Update log entry
		log_data[best_filename] = entry

		# Clean up old filename if renamed
		if name != latest_filename and name != best_filename and name in log_data: del log_data[name]

	# ----- Save updated JSON log -----
	with open(log_path, "w") as f:
		json.dump(log_data, f, indent='\t', sort_keys=True)

	print(f"📝 Updated log at {log_path}")

def load_qtable(
	save_dir="checkpoints",
	filename="qtable_latest.npy",
	log_filename="log.json"
):
	file_path = os.path.join(save_dir, filename)
	log_path = os.path.join(save_dir, log_filename)

	if not os.path.exists(file_path):
		print(f"⚠️ Q-table file {file_path} not found.")
		return None, {}, {}

	if not os.path.exists(log_path):
		print(f"⚠️ Log file {log_path} not found.")
		return np.load(file_path), {}, {}

	# Load Q-table
	Q = np.load(file_path)
	print(f"✅ Loaded Q-table from {file_path}")

	# Load log data
	with open(log_path, "r") as f:
		log_data = json.load(f)

	entry = log_data.get(filename)
	if entry is None:
		print(f"⚠️ No log entry found for {filename}")
		return Q, {}, {}

	params = entry.get("params", {})
	results = entry.get("results", {})

	return Q, params, results

