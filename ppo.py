import tensorflow as tf
import numpy as np
import gymnasium as gym
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Actor and Critic networks
class ActorCritic(tf.keras.Model):
	def __init__(self, state_size, action_size, hyperparameters={}):
		super(ActorCritic, self).__init__()
		self.state_size = state_size
		self.action_size = action_size
		self.hyperparameters = hyperparameters
		self.actor_dense = tf.keras.Sequential([
			tf.keras.layers.Dense(64, activation='relu'),
			tf.keras.layers.Dense(action_size)
		])
		self.critic_dense = tf.keras.Sequential([
			tf.keras.layers.Dense(64, activation='relu'),
			tf.keras.layers.Dense(1)
		])

	def call(self, state):
		logits = self.actor_dense(state)
		value = self.critic_dense(state)
		return logits, value

# PPO algorithm
def ppo_loss(model, optimizer_actor, optimizer_critic, old_logits, old_values, states, actions, returns):
	def compute_loss(logits, values, actions, returns, old_logits, old_values):
		actions_onehot = tf.one_hot(actions, model.action_size, dtype=tf.float32)
		policy = tf.nn.softmax(logits)
		action_probs = tf.reduce_sum(actions_onehot * policy, axis=1)
		old_policy = tf.nn.softmax(old_logits)
		old_action_probs = tf.reduce_sum(actions_onehot * old_policy, axis=1)

		# Policy loss
		ratio = tf.exp(tf.math.log(action_probs + 1e-10) - tf.math.log(old_action_probs + 1e-10))
		clipped_ratio = tf.clip_by_value(ratio, 1 - model.hyperparameters["clip_ratio"], 1 + model.hyperparameters["clip_ratio"])
		advantages = returns - tf.squeeze(old_values)
		advantages = (advantages - tf.reduce_mean(advantages)) / (tf.math.reduce_std(advantages) + 1e-8)
		policy_loss = -tf.reduce_mean(tf.minimum(ratio * advantages, clipped_ratio * advantages))

		# Value loss
		value_loss = tf.reduce_mean(tf.square(tf.squeeze(values) - returns))

		# Entropy bonus (optional)
		entropy_bonus = tf.reduce_mean(policy * tf.math.log(policy + 1e-10))

		total_loss = policy_loss + 0.5 * value_loss - 0.01 * entropy_bonus  # Entropy regularization
		return policy_loss, value_loss, total_loss

	for _ in range(model.hyperparameters["epochs"]):
		with tf.GradientTape(persistent=True) as tape:
			logits, values = model(states)
			actor_loss, critic_loss, total_loss = compute_loss(logits, values, actions, returns, old_logits, old_values)
		actor_grads = tape.gradient(actor_loss, model.actor_dense.trainable_variables)
		critic_grads = tape.gradient(critic_loss, model.critic_dense.trainable_variables)
		optimizer_actor.apply_gradients(zip(actor_grads, model.actor_dense.trainable_variables))
		optimizer_critic.apply_gradients(zip(critic_grads, model.critic_dense.trainable_variables))
		del tape

	return total_loss

def init(state_size, action_size, hyperparameters={}):
	model = ActorCritic(state_size, action_size, hyperparameters)
	return model

def train(env, model):
	optimizer_actor = tf.keras.optimizers.Adam(learning_rate=model.hyperparameters["lr_actor"])
	optimizer_critic = tf.keras.optimizers.Adam(learning_rate=model.hyperparameters["lr_critic"])

	results = {"best_score": -float("inf"), "avg_score": 0}

	for episode in range(model.hyperparameters["max_episodes"]):
		states, actions, rewards, values = [], [], [], []
		state, _ = env.reset()
		for step in range(model.hyperparameters["max_steps_per_episode"]):
			state = tf.expand_dims(tf.convert_to_tensor(state, dtype=tf.float32), 0)
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
				returns_batch = tf.convert_to_tensor(returns_batch, dtype=tf.float32)
				old_logits, _ = model(states)

				loss = ppo_loss(model, optimizer_actor, optimizer_critic, old_logits, values, states, actions, returns_batch)
				print(f"Episode: {episode + 1}, Loss: {loss}")
				results["best_score"] = max(results["best_score"], loss)
				results["avg_score"] += loss
				break

	results["avg_score"] /= model.hyperparameters["max_episodes"]
	return results

def simulate(env, model, max_steps=None, params={}):
	state, _ = env.reset(**params)
	for step in range(max_steps or model.hyperparameters["max_steps_per_episode"]):
		env.render()

		state = tf.expand_dims(tf.convert_to_tensor(state, dtype=tf.float32), 0)
		logits, _ = model(state)

		# Sample action from the policy distribution
		action = tf.random.categorical(logits, 1)[0, 0].numpy()
		state, _, done, truncated, _ = env.step(action)
		if done or truncated: return { "step": step, "reason": "done" if done else "truncated" }
	else: return { "step": step, "reason": "max_steps" }

def save_model(model, save_dir="checkpoints", prefix="ppo_model", hyperparameters={}, results={}, metric="avg_score"):
	if not os.path.exists(save_dir): os.mkdir(save_dir)
	# TODO: Save hyperparameters and results, use metric
	filename = prefix + ("" if prefix.endswith(".weights.h5") else ".weights.h5")
	model.save_weights(os.path.join(save_dir, filename))

def load_model(model, save_dir="checkpoints", filename="ppo_model"):
	# Ensure model is "built" by passing dummy input
	dummy_input = tf.random.uniform((1, model.state_size))
	model(dummy_input)  # This builds the model layers
	# TODO: Load hyperparameters and results
	filename = filename + ("" if filename.endswith(".weights.h5") else ".weights.h5")
	model.load_weights(os.path.join(save_dir, filename))
	# return results

