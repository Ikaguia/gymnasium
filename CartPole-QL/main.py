import argparse, sys, os
import gymnasium as gym
import numpy as np

# Import files from parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import utils
import q_learning as ql

# ---------------------------
# Constants
# ---------------------------

DISCRETIZATION_ARGS = {
	# position, velocity, angle, angular velocity
	'bins': [  12,   24,     24,   24],
	'low':  [-4.8, -3.0, -0.418, -5.0],
	'high': [ 4.8,  3.0,  0.418,  5.0],
}

Q_LEARN_ARGS = {
	'train_episodes': 8000,	# @param {type:"integer"}
	'decay': 0.0015,		# @param {type:"slider", min: 0.0, max:1.0, step: 0.001}
	'max_steps': 500,		# @param {type:"integer"}
	'eval_episodes': 1000,	# @param {type:"integer"}
	'min_epsilon': 0.01,	# @param {type:"number"}
	'max_epsilon': 1.00,	# @param {type:"number"}
	'alpha': 0.07,			# @param {type:"slider", min: 0.0, max:1.0, step: 0.01}
	'discount_factor': 0.99,# @param {type:"slider", min: 0.0, max:1.0, step: 0.01}
}

SIMULATION_SEED = 123

# ---------------------------
# Command-line interface
# ---------------------------

parser = argparse.ArgumentParser(description="Q-learning with save/load")
parser.add_argument("--load", nargs="?", const="latest", type=str, help="Load Q-table (default: latest if no file specified)")
parser.add_argument("--repeat", default=1, type=int, help="How many times should training repeat (default: 1)")
parser.add_argument("--silent", action="store_true", help="Disable printing of learning rewards")
parser.add_argument("--no_plot", action="store_true", help="Disable plotting of learning values")
parser.add_argument("--no_graphic", action="store_true", help="Disable graphic display for the final simulation")
parser.add_argument("--no_save", action="store_true", help="Don't save Q values to file")

args = parser.parse_args()

# ---------------------------
# Initialize Q
# ---------------------------

Q = None

# ---------------------------
# Load Q from file
# ---------------------------

if args.load:
	if args.load == "latest": filename = "qtable_latest"
	elif args.load.startswith("best"): filename = f"qtable_{args.load}{'1' if args.load == 'best' else ''}"
	if not filename.endswith(".npy"): filename += ".npy"
	loaded, params, results = ql.load_qtable(filename=filename)
	if loaded is not None:
		Q = loaded
		if "DISCRETIZATION_ARGS" in params: DISCRETIZATION_ARGS = params["DISCRETIZATION_ARGS"]
		if "Q_LEARN_ARGS" in params: Q_LEARN_ARGS = params["Q_LEARN_ARGS"]
	else: exit()

# ---------------------------
# Train Q
# ---------------------------

else:
	for repeat in range(args.repeat):
		if args.repeat > 1: print(f"Training ({repeat + 1})...")
		else: print("Training...")

		env = gym.make('CartPole-v1', render_mode="rgb_array")
		env = utils.FlatDiscretizeObservation(env, **DISCRETIZATION_ARGS)

		Q, training_rewards, epsilons = ql.q_learning(
			env,
			**Q_LEARN_ARGS,
			debug = False,
			silent = args.silent
		)
		results = { "avg_score": np.average(training_rewards[-10:]), "max_score": np.max(training_rewards[-10:]) }
		print(f"{results=}")

		if not args.no_plot: utils.plot(training_rewards, epsilons)

		# ---------------------------
		# Save Q to file
		# ---------------------------

		if not args.no_save:
			print("Saving...")
			ql.save_qtable(
				Q,
				params = {
					'Q_LEARN_ARGS': Q_LEARN_ARGS,
					'DISCRETIZATION_ARGS': DISCRETIZATION_ARGS,
				},
				results = results,
				metric = "avg_score",
			)

# ---------------------------
# Simulate using Q
# ---------------------------

print("Simulating...")
env = gym.make('CartPole-v1', render_mode=("rgb_array" if args.no_graphic else "human"))
env = utils.FlatDiscretizeObservation(env, **DISCRETIZATION_ARGS)

end = ql.simulate_using_Q(env, Q, max_steps=500, params={ "seed": SIMULATION_SEED })
print(f'{end=}')

