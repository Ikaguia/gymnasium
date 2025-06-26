import tensorflow as tf
import numpy as np
import gymnasium as gym
import os

# Actor and Critic networks
class ActorCritic(tf.keras.Model):
	def __init__(self, state_size, action_size, hyperparameters={}):
		super(ActorCritic, self).__init__()
		self.state_size = state_size
		self.action_size = action_size
		self.hyperparameters = hyperparameters
		self.dense1 = tf.keras.layers.Dense(64, activation='relu')
		self.policy_logits = tf.keras.layers.Dense(action_size)
		self.dense2 = tf.keras.layers.Dense(64, activation='relu')
		self.value = tf.keras.layers.Dense(1)

	def call(self, state):
		x = self.dense1(state)
		logits = self.policy_logits(x)
		value = self.value(x)
		return logits, value

# PPO algorithm
def ppo_loss(model, optimizer, old_logits, old_values, advantages, states, actions, returns):
	def compute_loss(logits, values, actions, returns):
		actions_onehot = tf.one_hot(actions, model.action_size, dtype=tf.float32)
		policy = tf.nn.softmax(logits)
		action_probs = tf.reduce_sum(actions_onehot * policy, axis=1)
		old_policy = tf.nn.softmax(old_logits)
		old_action_probs = tf.reduce_sum(actions_onehot * old_policy, axis=1)

		# Policy loss
		ratio = tf.exp(tf.math.log(action_probs + 1e-10) - tf.math.log(old_action_probs + 1e-10))
		clipped_ratio = tf.clip_by_value(ratio, 1 - model.hyperparameters["clip_ratio"], 1 + model.hyperparameters["clip_ratio"])
		policy_loss = -tf.reduce_mean(tf.minimum(ratio * advantages, clipped_ratio * advantages))

		# Value loss
		value_loss = tf.reduce_mean(tf.square(values - returns))

		# Entropy bonus (optional)
		entropy_bonus = tf.reduce_mean(policy * tf.math.log(policy + 1e-10))

		total_loss = policy_loss + 0.5 * value_loss - 0.01 * entropy_bonus  # Entropy regularization
		return total_loss

	def get_advantages(returns, values):
		advantages = returns - values
		return (advantages - tf.reduce_mean(advantages)) / (tf.math.reduce_std(advantages) + 1e-8)

	def train_step(states, actions, returns, old_logits, old_values):
		with tf.GradientTape() as tape:
			logits, values = model(states)
			loss = compute_loss(logits, values, actions, returns)
		gradients = tape.gradient(loss, model.trainable_variables)
		optimizer.apply_gradients(zip(gradients, model.trainable_variables))
		return loss

	advantages = get_advantages(returns, old_values)
	for _ in range(model.hyperparameters["epochs"]):
		loss = train_step(states, actions, returns, old_logits, old_values)
	return loss

def init(state_size, action_size, hyperparameters={}):
	model = ActorCritic(state_size, action_size, hyperparameters)
	model.build(input_shape=(None, state_size))
	return model

def train(
	env,
	model,
):
	optimizer = tf.keras.optimizers.Adam(learning_rate=model.hyperparameters["lr_actor"])

	results = { "best_score": -float("inf"), "avg_score": 0 }

	for episode in range(model.hyperparameters["max_episodes"]):
		states, actions, rewards, values, returns = [], [], [], [], []
		state,_ = env.reset()
		for step in range(model.hyperparameters["max_steps_per_episode"]):
			state = tf.expand_dims(tf.convert_to_tensor(state), 0)
			logits, value = model(state)

			# Sample action from the policy distribution
			action = tf.random.categorical(logits, 1)[0, 0].numpy()
			next_state, reward, done, truncated, _ = env.step(action)

			states.append(state)
			actions.append(action)
			rewards.append(reward)
			values.append(value)

			state = next_state

			if done or truncated:
				returns_batch = []
				discounted_sum = 0
				for r in rewards[::-1]:
					discounted_sum = r + model.hyperparameters["gamma"] * discounted_sum
					returns_batch.append(discounted_sum)
				returns_batch.reverse()

				states = tf.concat(states, axis=0)
				actions = np.array(actions, dtype=np.int32)
				values = tf.concat(values, axis=0)
				returns_batch = tf.convert_to_tensor(returns_batch)
				old_logits, _ = model(states)

				loss = ppo_loss(model, optimizer, old_logits, values, returns_batch - np.array(values), states, actions, returns_batch)
				print(f"Episode: {episode + 1}, Loss: {loss.numpy()}")
				results["best_score"] = max(results["best_score"], loss)
				results["avg_score"] = results["avg_score"] + loss
				break

	results["avg_score"] = results["avg_score"] / model.hyperparameters["max_episodes"]

	return results

def simulate(env, model, max_steps=None, params={}):
	state, _ = env.reset(**params)
	for step in range(max_steps or model.hyperparameters["max_steps_per_episode"]):
	    env.render()

	    state = tf.expand_dims(tf.convert_to_tensor(state), 0)
	    logits, value = model(state)

	    # Sample action from the policy distribution
	    action = tf.random.categorical(logits, 1)[0, 0].numpy()
	    next_state, reward, done, truncated, _ = env.step(action)

	    state = next_state

	    if done or truncated: return { "step": step, "reason": "done" if done else "truncated" }
	else:
	    return { "step": step, "reason": "max_steps" }

def save_model(model, save_dir="checkpoints", prefix="ppo_model", hyperparameters={}, results={}, metric="avg_score"):
	if not os.path.exists(save_dir): os.mkdir(save_dir)
	# TODO: Save hyperparameters and results, use metric
	filename = prefix + ("" if prefix.endswith(".weights.h5") else ".weights.h5")
	model.save_weights(os.path.join(save_dir, filename))

def load_model(model, save_dir="checkpoints", filename="ppo_model"):
	# TODO: Load hyperparameters and results
	filename = filename + ("" if filename.endswith(".weights.h5") else ".weights.h5")
	model.load_weights(os.path.join(save_dir, filename))
	# return results

