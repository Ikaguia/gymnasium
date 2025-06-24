import numpy as np

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
