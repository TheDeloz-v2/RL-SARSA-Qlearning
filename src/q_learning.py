import numpy as np
import random

def q_learning(
    env,
    episodes: int,
    alpha: float,
    gamma: float,
    epsilon: float,
    seed: int = None,
    epsilon_min: float = 0.01,
    epsilon_decay: float = 0.99
):
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
        env.reset(seed=seed)

    # Q-table initialization
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    rewards = []

    for ep in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            # e-greedy policy
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state, :])

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated  # which one is True ends the episode

            # Q-Learning update rule
            best_next_action = np.argmax(Q[next_state, :])
            target = reward + gamma * Q[next_state, best_next_action]
            error = target - Q[state, action]
            Q[state, action] += alpha * error

            state = next_state
            total_reward += reward

        # Decay epsilon
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        rewards.append(total_reward)

    return Q, rewards
