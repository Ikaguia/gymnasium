import numpy as np
import os, json

def q_learning(
    env,
    train_episodes = 1000,  # @param {type:"integer"}
    decay = 0.01,           # @param {type:"slider", min: 0.0, max:1.0, step: 0.001}
    max_steps = 100,        # @param {type:"integer"}
    eval_episodes = 10,     # @param {type:"integer"}
    min_epsilon = 0.05,     # @param {type:"number"}
    max_epsilon = 1.00,     # @param {type:"number"}
    alpha = 0.05,           # @param {type:"slider", min: 0.0, max:1.0, step: 0.01}
    discount_factor = 0.0,  # @param {type:"slider", min: 0.0, max:1.0, step: 0.01}
    debug = False,
    silent = False,
):
    Q = np.zeros((env.observation_space.n, env.action_space.n))

    epsilon = 1
    training_rewards = []
    epsilons = []

    for episode in range(train_episodes):
        state, _ = env.reset()
        total_training_rewards = 0

        for step in range(max_steps):
            roulette = np.random.uniform(0, 1)

            action = (np.argmax(Q[state,:]) if roulette > epsilon  # exploit
                      else env.action_space.sample())  # explore

            new_state, reward, done, truncated, info = env.step(action)

            if debug and reward != 0.1: print(reward)

            Q[state, action] = (
                (1-alpha) * Q[state, action] +
                alpha  * (reward + discount_factor * Q[new_state, :].max())
            )

            total_training_rewards += reward
            state = new_state

            if done or truncated:
                break

        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay*episode)

        training_rewards.append(total_training_rewards)
        epsilons.append(epsilon)

        if episode % eval_episodes == 0 and not silent:
            try:
                print(f"Episode {episode:>3}: Accumulated reward: {np.mean(training_rewards[-10:]): 10.2f}")
            except:
                pass

    return Q, training_rewards, epsilons

def simulate_using_Q(env, Q, max_steps=100, debug=False, params={}):
    observation, _ = env.reset(**params)

    end = {}

    for step in range(max_steps):

        env.render()

        action = np.argmax(Q[observation,:])
        if debug: print(Q[observation,:])

        observation, reward, done, truncated, info = env.step(action)

        if done or truncated:
            end = { "step": step, "reason": "done" if done else "truncated" }
            break
    else:
        end = { "step": step, "reason": "max_steps" }

    env.close()

    return end

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

