import itertools
import json
import numpy as np
import gymnasium as gym
import q_learning as ql
import utils

DISCRETIZATION_ARGS = {
	# position, velocity, angle, angular velocity
	'bins': [  12,   24,     24,   24],
	'low':  [-4.8, -3.0, -0.418, -5.0],
	'high': [ 4.8,  3.0,  0.418,  5.0],
}


# Parameters to tune
param_grid = {
	'train_episodes': [8000],				# @param {type:"integer"}
	'decay': [0.001, 0.002, 0.003],			# @param {type:"slider", min: 0.0, max:1.0, step: 0.001}
	'max_steps': [500],						# @param {type:"integer"}
	'eval_episodes': [1000],				# @param {type:"integer"}
	'min_epsilon': [0.01, 0.05],			# @param {type:"number"}
	'max_epsilon': [1.00],					# @param {type:"number"}
	'alpha': [0.05, 0.07, 0.1],				# @param {type:"slider", min: 0.0, max:1.0, step: 0.01}
	'discount_factor': [0.95, 0.99],		# @param {type:"slider", min: 0.0, max:1.0, step: 0.01}
}

def run_training(params):
	env = gym.make('CartPole-v1', render_mode="rgb_array")
	env = utils.FlatDiscretizeObservation(env, **DISCRETIZATION_ARGS)

	_, training_rewards, _ = ql.q_learning(
		env,
		**params,
		debug = False,
		silent = True
	)
	return np.average(training_rewards[-10:]), np.max(training_rewards[-10:])

def main():
	results = []

	# Generate all combinations of parameters
	for combo in itertools.product(*param_grid.values()):
		# Build param dict
		params = dict(zip(param_grid.keys(), combo))
		print(f"Training with params: {params}")

		avg_reward, max_reward = run_training(params)

		# Log the result
		results.append({
			"params": params,
			"results": {
				"avg_reward": avg_reward,
				"max_reward": max_reward
			}
		})

		print(f"→ Avg Reward: {avg_reward:.2f}, Max Reward: {max_reward:.2f}")

	# Save all results to JSON file
	with open("tuning_results.json", "w") as f:
		json.dump(results, f, indent=4)

	print("All tuning results saved to tuning_results.json")

if __name__ == "__main__":
	main()
