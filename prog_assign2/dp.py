from typing import Tuple

import numpy as np
from env import EnvWithModel
from policy import Policy


def value_prediction(env: EnvWithModel, pi: Policy, initV: np.array, theta: float) -> Tuple[np.array, np.array]:
    """
    inp:
        env: environment with model information, i.e. you know transition dynamics and reward function
        pi: policy
        initV: initial V(s); numpy array shape of [nS,]
        theta: exit criteria
    return:
        V: $v_\pi$ function; numpy array shape of [nS]
        Q: $q_\pi$ function; numpy array shape of [nS,nA]
    """
    nS, nA, gamma, TD, R = env.spec.nS, env.spec.nA, env.spec.gamma, env.TD, env.R
    V = np.copy(initV)  # Initialize V with initV

    while True:
        delta = 0  # For checking convergence
        for s in range(nS):
            v = V[s]  # Save the current value of V(s)
            V[s] = sum(pi.action_prob(s, a) * sum(env.TD[s, a, s_prime] * (env.R[s, a, s_prime] + gamma * V[s_prime])
                                                  for s_prime in range(nS)) for a in range(nA))
            delta = max(delta, abs(v - V[s]))  # Update delta

        if delta < theta:  # Check for convergence
            break

    # Compute Q using the converged V
    Q = np.zeros((nS, nA))
    for s in range(nS):
        for a in range(nA):
            Q[s, a] = sum(env.TD[s, a, s_prime] * (env.R[s, a, s_prime] + gamma * V[s_prime]) for s_prime in range(nS))

    return V, Q


class GreedyPolicy(Policy):
    def __init__(self, pi):
        self.pi = pi

    def action_prob(self, state, action=None):
        return 1. if self.pi[state] == action else 0.

    def action(self, state):
        """
        Returns the preferred action for the given state.
        """
        return self.pi[state]


def value_iteration(env: EnvWithModel, initV: np.array, theta: float) -> Tuple[np.array, Policy]:
    """
    inp:
        env: environment with model information, i.e. you know transition dynamics and reward function
        initV: initial V(s); numpy array shape of [nS,]
        theta: exit criteria
    return:
        value: optimal value function; numpy array shape of [nS]
        policy: optimal deterministic policy; instance of Policy class
    """

    #####################
    # TODO: Implement Value Iteration Algorithm (Hint: Sutton Book p.83)
    #####################

    V = np.copy(initV)  # Initialize V with the initial value function
    nS = env.spec.nS  # Number of states
    nA = env.spec.nA  # Number of actions
    gamma = env.spec.gamma  # Discount factor

    while True:
        delta = 0  # For checking convergence
        # Update each state's value
        for s in range(nS):
            v = V[s]  # Store the current value of V[s]
            # Use a temporary variable to store the max value for state s
            max_value = max([sum([env.TD[s, a, s_prime] * (env.R[s, a, s_prime] + gamma * V[s_prime])
                                  for s_prime in range(nS)]) for a in range(nA)])
            V[s] = max_value  # Update the value function with the max value
            delta = max(delta, abs(v - V[s]))  # Update delta

        if delta < theta:  # Check for convergence
            break

    # Derive the optimal policy from the value function
    optimal_policy = np.zeros(nS, dtype=int)
    for s in range(nS):
        # Find the action that maximizes the expected return
        action_values = np.array([sum([env.TD[s, a, s_prime] * (env.R[s, a, s_prime] + gamma * V[s_prime])
                                       for s_prime in range(nS)]) for a in range(nA)])
        optimal_policy[s] = np.argmax(action_values)

    # Create a Greedy policy from the optimal actions
    pi = GreedyPolicy(optimal_policy)
    return V, pi
