import tensorflow as tf
import numpy as np
import gymnasium as gym
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

HYPERPARAMETERS = {
	"gamma": 0.99,		# Discount factor
	"lr_actor": 0.001,	# Actor learning rate
	"lr_critic": 0.001,	# Critic learning rate
	"clip_ratio": 0.2,	# PPO clip ratio
	"epochs": 10,		# Number of optimization epochs
	"batch_size": 64,	# Batch size for optimization
	"entropy_bonus": 0.05,
	"max_episodes": 1000,
	"max_steps_per_episode": 1000,
	"converged_loss_range": 20,		# How many episodes in a row have their loss within the threshold of each other for early termination. 0 for never terminate early.
	"converged_loss_threshold": 10,	# Theshold for terminating training early
	"normalize_state": True,
}

# Running mean and std tracker
class RunningNormalizer:
	def __init__(self, size, epsilon=1e-8):
		self.mean = tf.Variable(tf.zeros(size), trainable=False)
		self.var = tf.Variable(tf.ones(size), trainable=False)
		self.count = tf.Variable(epsilon, trainable=False)

	def update(self, x):
		batch_mean = tf.reduce_mean(x, axis=0)
		batch_var = tf.math.reduce_variance(x, axis=0)
		batch_count = tf.cast(tf.shape(x)[0], tf.float32)

		total_count = self.count + batch_count
		delta = batch_mean - self.mean
		new_mean = self.mean + delta * (batch_count / total_count)
		m_a = self.var * self.count
		m_b = batch_var * batch_count
		M2 = m_a + m_b + tf.square(delta) * self.count * batch_count / total_count
		new_var = M2 / total_count

		self.mean.assign(new_mean)
		self.var.assign(new_var)
		self.count.assign(total_count)

	def normalize(self, x):
		x_norm = (x - self.mean) / tf.sqrt(self.var + 1e-8)
		return tf.clip_by_value(x_norm, -5.0, 5.0)  # Clipping after normalization

	def save(self, filepath):
		# Convert variables to numpy arrays and save
		np.savez(filepath,
				 mean=self.mean.numpy(),
				 var=self.var.numpy(),
				 count=self.count.numpy())

	def load(self, filepath):
		if not os.path.exists(filepath):
			raise FileNotFoundError(f"No normalizer file found at {filepath}")
		data = np.load(filepath)
		self.mean.assign(data['mean'])
		self.var.assign(data['var'])
		self.count.assign(data['count'])

# Actor and Critic networks for continuous action spaces
class ActorCritic(tf.keras.Model):
	def __init__(self, state_size, action_size, hyperparameters={}):
		super(ActorCritic, self).__init__()
		self.state_size = state_size
		self.action_size = action_size
		self.hyperparameters = {
			**HYPERPARAMETERS,
			**hyperparameters,
		}
		self.actor_dense = tf.keras.Sequential([
			tf.keras.layers.Dense(128, activation='relu'),
			tf.keras.layers.Dense(64, activation='relu'),
			tf.keras.layers.Dense(action_size)
		])
		self.log_std = tf.Variable(initial_value=-0.5 * tf.ones((action_size,)), trainable=True)
		self.critic_dense = tf.keras.Sequential([
			tf.keras.layers.Dense(128, activation='relu'),
			tf.keras.layers.Dense(64, activation='relu'),
			tf.keras.layers.Dense(1)
		])
		self.normalizer = RunningNormalizer(state_size)

	def call(self, state):
		mean = self.actor_dense(state)
		std = tf.exp(self.log_std)
		value = self.critic_dense(state)
		return mean, std, value

def gaussian_log_prob(mean, std, action):
	pre_sum = -0.5 * (((action - mean) / (std + 1e-8))**2 + 2 * tf.math.log(std + 1e-8) + tf.math.log(2 * np.pi))
	return tf.reduce_sum(pre_sum, axis=1)

def gaussian_entropy(std):
	return tf.reduce_sum(0.5 * tf.math.log(2 * np.pi * np.e * std**2), axis=-1)

# PPO algorithm

@tf.function
def compute_ppo_loss(model, old_means, old_stds, old_values, states, actions, returns):
	means, stds, values = model(states)
	log_probs = gaussian_log_prob(means, stds, actions)
	old_log_probs = gaussian_log_prob(old_means, old_stds, actions)
	ratio = tf.exp(log_probs - old_log_probs)
	advantages = returns - tf.squeeze(old_values)
	advantages = (advantages - tf.reduce_mean(advantages)) / (tf.math.reduce_std(advantages) + 1e-8)
	clipped_ratio = tf.clip_by_value(ratio, 1 - model.hyperparameters["clip_ratio"], 1 + model.hyperparameters["clip_ratio"])
	policy_loss = -tf.reduce_mean(tf.minimum(ratio * advantages, clipped_ratio * advantages))
	value_loss = tf.reduce_mean(tf.square(tf.squeeze(values) - returns))
	entropy = tf.reduce_mean(gaussian_entropy(stds))
	total_loss = policy_loss + 0.5 * value_loss - model.hyperparameters["entropy_bonus"] * entropy
	return means, stds, values, policy_loss, value_loss, total_loss

def ppo_loss(model, optimizer_actor, optimizer_critic, old_means, old_stds, old_values, states, actions, returns):
	for _ in range(model.hyperparameters["epochs"]):
		with tf.GradientTape(persistent=True) as tape:
			means, stds, values, actor_loss, critic_loss, total_loss = compute_ppo_loss(model, old_means, old_stds, old_values, states, actions, returns)
		actor_grads = tape.gradient(actor_loss, model.actor_dense.trainable_variables + [ model.log_std ])
		critic_grads = tape.gradient(critic_loss, model.critic_dense.trainable_variables)
		optimizer_actor.apply_gradients(zip(actor_grads, model.actor_dense.trainable_variables + [ model.log_std ]))
		optimizer_critic.apply_gradients(zip(critic_grads, model.critic_dense.trainable_variables))
		del tape

	return total_loss

def init(state_size, action_size, hyperparameters={}):
	model = ActorCritic(state_size, action_size, hyperparameters)
	return model

def train(env, model, silent=False, partial_save=0):
	optimizer_actor = tf.keras.optimizers.Adam(learning_rate=model.hyperparameters["lr_actor"])
	optimizer_critic = tf.keras.optimizers.Adam(learning_rate=model.hyperparameters["lr_critic"])

	results = {"best_score": -float("inf"), "avg_score": 0}
	loss_history = []
	episode_rewards = []  # Track total reward per episode

	for episode in range(model.hyperparameters["max_episodes"]):
		raw_states, states, actions, rewards, values = [], [], [], [], []
		state, _ = env.reset()
		episode_reward = 0  # Initialize total reward for this episode

		for step in range(model.hyperparameters["max_steps_per_episode"]):
			last_step = step == (model.hyperparameters["max_steps_per_episode"] - 1)
			state_tensor = tf.expand_dims(tf.convert_to_tensor(state, dtype=tf.float32), 0)
			raw_states.append(state_tensor)  # Collect raw (unnormalized) state

			if model.hyperparameters["normalize_state"]:
				# Use previously accumulated normalizer statistics — do not update here
				norm_state = model.normalizer.normalize(state_tensor)
				mean, std, value = model(norm_state)
				states.append(norm_state)
			else:
				mean, std, value = model(state_tensor)
				states.append(state_tensor)

			action = tf.random.normal(shape=(model.action_size,), mean=tf.squeeze(mean), stddev=tf.squeeze(std))
			action_clipped = tf.clip_by_value(action, env.action_space.low[0], env.action_space.high[0])
			next_state, reward, done, truncated, _ = env.step(action_clipped.numpy())

			actions.append(tf.expand_dims(action, 0))
			rewards.append(reward)
			values.append(value)

			state = next_state
			episode_reward += reward  # Accumulate reward

			if done or truncated or last_step:
				returns_batch = []
				discounted_sum = 0
				for r in rewards[::-1]:
					discounted_sum = r + model.hyperparameters["gamma"] * discounted_sum
					returns_batch.append(discounted_sum)
				returns_batch.reverse()

				states = tf.concat(states, axis=0)
				actions = tf.concat(actions, axis=0)
				values = tf.concat(values, axis=0)
				returns_batch = tf.convert_to_tensor(returns_batch, dtype=tf.float32)

				# Update normalizer only after the full episode ends
				if model.hyperparameters["normalize_state"]:
					raw_batch = tf.concat(raw_states, axis=0)
					model.normalizer.update(raw_batch)
					states = model.normalizer.normalize(raw_batch)

				old_means, old_stds, _ = model(states)
				loss = ppo_loss(model, optimizer_actor, optimizer_critic, old_means, old_stds, values, states, actions, returns_batch)
				loss_history.append(float(loss))
				episode_rewards.append(episode_reward)

				if not silent:
					print(f"Episode: {(episode + 1):4d}, Reward: {episode_reward:7.2f}, Loss: {loss:9.4f}")

				# Check if last X losses are within Y units of each other for early termination of training
				if model.hyperparameters["converged_loss_range"] > 0 and len(loss_history) >= model.hyperparameters["converged_loss_range"]:
					recent = loss_history[-model.hyperparameters["converged_loss_range"]:]
					if max(recent) - min(recent) < model.hyperparameters["converged_loss_threshold"]:
						print("Stopping early due to converged loss.")
						results["early_stop"] = episode + 1
						model.hyperparameters["max_episodes"] = episode + 1
				break

		if partial_save != 0 and (episode + 1) % partial_save == 0:
			save_model(model, prefix=f"ppo_model_partial_{(episode + 1)}")

		if "early_stop" in results: break

	return {
		"avg_score": np.mean(episode_rewards) if episode_rewards else 0,
		"best_score": np.max(episode_rewards) if episode_rewards else -float("inf"),
		"rewards": episode_rewards,
	}

def simulate(env, model, max_steps=None, params={}):
	state, _ = env.reset(**params)
	for step in range(max_steps or model.hyperparameters["max_steps_per_episode"]):
		env.render()
		state_tensor = tf.expand_dims(tf.convert_to_tensor(state, dtype=tf.float32), 0)

		# Normalize state using frozen normalizer stats (do NOT update during simulation)
		if model.hyperparameters["normalize_state"]:
			norm_state = model.normalizer.normalize(state_tensor)
			mean, std, _ = model(norm_state)
		else:
			mean, std, _ = model(state_tensor)

		# Sample action from the policy distribution
		action = tf.random.normal(shape=(model.action_size,), mean=tf.squeeze(mean), stddev=tf.squeeze(std))
		action_clipped = tf.clip_by_value(action, env.action_space.low[0], env.action_space.high[0])
		state, _, done, truncated, _ = env.step(action_clipped.numpy())
		if done or truncated: return { "step": step, "reason": "done" if done else "truncated" }
	else: return { "step": step, "reason": "max_steps" }

def save_model(model, save_dir="checkpoints", prefix="ppo_model", results={}, metric="avg_score"):
	# TODO: Save hyperparameters and results, use metric
	if not os.path.exists(save_dir): os.mkdir(save_dir)
	prefix = prefix[:-11] if prefix.endswith(".weights.h5") else prefix
	weights_filename = prefix + ".weights.h5"
	normalizer_filename = prefix + ".normalizer.npz"
	model.save_weights(os.path.join(save_dir, weights_filename))
	model.normalizer.save(os.path.join(save_dir, normalizer_filename))

def load_model(model, save_dir="checkpoints", filename="ppo_model"):
	# Ensure model is "built" by passing dummy input
	dummy_input = tf.random.uniform((1, model.state_size))
	model(dummy_input)  # This builds the model layers

	# TODO: Load hyperparameters and results
	prefix = prefix[:-11] if filename.endswith(".weights.h5") else filename
	weights_filename = prefix + ".weights.h5"
	normalizer_filename = prefix + ".normalizer.npz"
	model.load_weights(os.path.join(save_dir, weights_filename))
	model.normalizer.load(os.path.join(save_dir, normalizer_filename))
	# return results

