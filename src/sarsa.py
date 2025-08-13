import numpy as np
from typing import Tuple, List

def epsilon_greedy(
    Q: np.ndarray,
    state: int,
    epsilon: float,
    rng: np.random.Generator
) -> int:
    """
    Epsilon-greedy policy.

    Args:
        Q (np.ndarray): Q-table
        state (int): Current state
        epsilon (float): Exploration rate
        rng (np.random.Generator): Random number generator

    Returns:
        int: Action
    """
    if rng.random() < epsilon:
        return rng.integers(low=0, high=Q.shape[1])
    return int(np.argmax(Q[state]))

def sarsa(
    env: None,
    episodes: int,
    alpha: float,
    gamma: float,
    epsilon: float,
    seed: int = 777,
    epsilon_min: float = 0.01,
    epsilon_decay: float = 0.999
) -> Tuple[np.ndarray, List[float]]:
    """
    SARSA algorithm.

    Args:
        env (None): Environment
        episodes (int): Number of episodes
        alpha (float): Learning rate
        gamma (float): Discount factor
        epsilon (float): Exploration rate
        seed (int): Random seed
        epsilon_min (float): Minimum exploration rate
        epsilon_decay (float): Exploration rate decay

    Returns:
        Tuple[np.ndarray, List[float]]: Q-table and rewards per episode
    """
    rng = np.random.default_rng(seed)
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    Q = np.zeros((n_states, n_actions), dtype=np.float64)

    rewards_per_episode: List[float] = []
    eps = float(epsilon)

    for ep in range(episodes):
        state, _ = env.reset(seed=seed + ep)
        action = epsilon_greedy(Q, state, eps, rng)

        done = False
        ep_return = 0.0

        while not done:
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_return += reward

            # Choose next action with epsilon-greedy
            if not done:
                next_action = epsilon_greedy(Q, next_state, eps, rng)
                target = reward + gamma * Q[next_state, next_action]
            else:
                next_action = None
                target = reward

            # SARSA update
            td_error = target - Q[state, action]
            Q[state, action] += alpha * td_error

            # Advance
            state = next_state
            action = action if next_action is None else next_action

        rewards_per_episode.append(ep_return)

        # epsilon decay kind of smoothy
        eps = max(epsilon_min, eps * epsilon_decay)

    return Q, rewards_per_episode