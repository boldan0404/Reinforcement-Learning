from typing import Iterable, Tuple

import numpy as np
from env import EnvSpec
from policy import Policy
from collections import defaultdict


def off_policy_mc_prediction_ordinary_importance_sampling(
        env_spec: EnvSpec,
        trajs: Iterable[Iterable[Tuple[int, int, int, int]]],
        bpi: Policy,
        pi: Policy,
        initQ: np.array
) -> np.array:
    """
    input:
        env_spec: environment spec
        trajs: N trajectories generated using
            list in which each element is a tuple representing (s_t,a_t,r_{t+1},s_{t+1})
        bpi: behavior policy used to generate trajectories
        pi: evaluation target policy
        initQ: initial Q values; np array shape of [nS,nA]
    ret:
        Q: $q_pi$ function; numpy array shape of [nS,nA]
    """

    #####################
    # TODO: Implement Off Policy Monte-Carlo prediction algorithm using ordinary importance
    # sampling (Hint: Sutton Book p. 109, every-visit implementation is fine)
    #####################
    Q = np.copy(initQ)
    C = np.zeros(env_spec.nS, env_spec.nA)

    # Loop over each trajectory
    for traj in trajs:
        traj_list = list(traj)
        G = 0  # Cumulative return
        W = 1  # Importance sampling ratio
        for t in range(len(traj_list) - 1, -1, -1):  # Start from the end of the episode
            s = traj_list[t][0]
            a = traj_list[t][1]
            r = traj_list[t][2]
            G = env_spec.gamma * G + r  # Update the return
            C[s,a] += 1  # Update the cumulative sum of the weights
            Q[s, a] += (W /C[s,a]) * (G - Q[s, a])  # Incremental update of Q
            if pi.action(s) != a:  # If the action taken is not the action prescribed by the target policy
                break  # Break the loop and move to the next episode
            W *= pi.action_prob(s, a) / bpi.action_prob(s, a) if bpi.action_prob(s,
                                                                                 a) > 0 else 0
    return Q


def off_policy_mc_prediction_weighted_importance_sampling(
        env_spec: EnvSpec,
        trajs: Iterable[Iterable[Tuple[int, int, int, int]]],
        bpi: Policy,
        pi: Policy,
        initQ: np.array

) -> np.array:
    """
    input:
        env_spec: environment spec
        trajs: N trajectories generated using behavior policy bpi
            list in which each element is a tuple representing (s_t,a_t,r_{t+1},s_{t+1})
        bpi: behavior policy used to generate trajectories
        pi: evaluation target policy
        initQ: initial Q values; np array shape of [nS,nA]
    ret:
        Q: $q_pi$ function; numpy array shape of [nS,nA]
    """

    #####################
    # TODO: Implement Off Policy Monte-Carlo prediction algorithm using weighted importance
    # sampling (Hint: Sutton Book p. 110, every-visit implementation is fine)
    #####################
    Q = np.copy(initQ)
    C = np.zeros(env_spec.nS,env_spec.nA)

    # Loop over each trajectory
    for traj in trajs:
        traj_list = list(traj)
        G = 0  # Cumulative return
        W = 1  # Importance sampling ratio
        for t in range(len(traj_list)-1,-1,-1):  # Start from the end of the episode
            s= traj_list[t][0]
            a = traj_list[t][1]
            r = traj_list[t][2]
            G = env_spec.gamma * G + r # Update the return
            C[s, a] += W  # Update the cumulative sum of the weights
            Q[s, a] += (W / C[(s, a)]) * (G - Q[s, a])  # Incremental update of Q
            if pi.action(s) != a:  # If the action taken is not the action prescribed by the target policy
                break  # Break the loop and move to the next episode
            W *= pi.action_prob(s, a) / bpi.action_prob(s, a) if bpi.action_prob(s,
                                                                                 a) > 0 else 0

    return Q
