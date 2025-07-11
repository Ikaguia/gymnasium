import argparse, sys, os
import gymnasium as gym
import numpy as np

# Import files from parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import utils
import ppo_continuous as ppo

# ---------------------------
# Constants
# ---------------------------

HYPERPARAMETERS = {
	"gamma": 0.99,			# Discount factor
	"lr_actor": 0.0003,		# Actor learning rate
	"lr_critic": 0.0005,	# Critic learning rate
	"clip_ratio": 0.2,		# PPO clip ratio
	"epochs": 20,			# Number of optimization epochs
	"batch_size": 1024,		# Batch size for optimization
	"max_episodes": 5000,
	"max_steps_per_episode": 2000,
	"converged_loss_range": 20,		# How many episodes in a row have their loss within the threshold of each other for early termination. 0 for never terminate early.
	"converged_loss_threshold": 10,	# Theshold for terminating training early
	"normalize_state": True,
}

SIMULATION_SEED = 123

# ---------------------------
# Command-line interface
# ---------------------------

parser = argparse.ArgumentParser(description="PPO with save/load")
parser.add_argument("--load", nargs="?", const="latest", type=str, help="Load PPO Model (default: latest if no file specified)")
parser.add_argument("--repeat", default=1, type=int, help="How many times should training repeat (default: 1)")
parser.add_argument("--silent", action="store_true", help="Disable printing of learning rewards")
parser.add_argument("--no_plot", action="store_true", help="Disable plotting of learning values")
parser.add_argument("--no_graphic", action="store_true", help="Disable graphic display for the final simulation")
parser.add_argument("--no_save", action="store_true", help="Don't save model to file")

args = parser.parse_args()

# ---------------------------
# Initialize
# ---------------------------

env = gym.make("InvertedPendulum-v5", render_mode="rgb_array")
model = ppo.init(state_size=env.observation_space.shape[0], action_size=env.action_space.shape[0], hyperparameters=HYPERPARAMETERS)

# ---------------------------
# Load model weights from file
# ---------------------------

if args.load:
	filename = args.load
	if filename == "latest": filename = "ppo_model_latest"
	elif filename.startswith("best"): filename = f"ppo_model_{filename}{'1' if filename == 'best' else ''}"
	if not filename.endswith(".weights.h5"): filename += ".weights.h5"
	ppo.load_model(model, filename=filename)

# ---------------------------
# Train model
# ---------------------------

else:
	for repeat in range(args.repeat):
		if args.repeat > 1: print(f"Training ({repeat + 1})...")
		else: print("Training...")

		env = gym.make('InvertedPendulum-v5', render_mode="rgb_array")

		model = ppo.init(state_size=env.observation_space.shape[0], action_size=env.action_space.shape[0], hyperparameters=HYPERPARAMETERS)
		results = ppo.train(env, model, silent=args.silent)
		print(f"{results=}")
		if not args.no_plot: utils.plot(results["rewards"], None)

		# ---------------------------
		# Save model weights to file
		# ---------------------------

		if not args.no_save:
			print("Saving...")
			ppo.save_model(model, results=results, metric="avg_score")

# ---------------------------
# Simulate using model
# ---------------------------

print("Simulating...")
env = gym.make('InvertedPendulum-v5', render_mode=("rgb_array" if args.no_graphic else "human"))
results = ppo.simulate(env, model, max_steps=500, params={ "seed": SIMULATION_SEED })
print(f'{results=}')

env.close()