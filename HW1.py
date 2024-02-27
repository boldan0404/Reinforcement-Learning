import sys
import numpy as np


class NonStationaryBandit:
    def __init__(self, k=10, bandit_seed=0):
        self.rg = np.random.RandomState(seed=bandit_seed)
        self.q_star = np.zeros(k)

    def reset(self, episode_seed=None):  # based on seed, different episode will be generated
        if episode_seed is None:
            episode_seed = int(self.rg.randint(0, 100000000))

        self.episode_rg = np.random.RandomState(seed=episode_seed)
        # TODO Question: do i need to modify this part of code to set q to zero

    def best(self):  # this determines the best action to take
        # TODO: your code goes here
        return np.argmax(self.q_star)

    def step(self, a):
        # TODO: your code goes here
        # hint: you can use self.episode_rg to generate noise

        # TODO should I use rg or episode_rg
        # Update all q_star values with independent random walks
        self.q_star += self.episode_rg.normal(0, 0.01, size=self.q_star.shape)
        # Reward is q_star(a) + some noise
        reward = self.q_star[a] + self.episode_rg.normal(0, 1)
        return reward


class ActionValue:
    def __init__(self, k, epsilon):
        self.k = k
        self.epsilon = epsilon
        self.q = np.zeros(k)

    def reset(self):
        # no code needed, but implement this method in the child classes
        raise NotImplementedError

    def update(self, a, r):
        # no code needed, but implement this method in the child classes
        raise NotImplementedError

    def epsilon_greedy_policy(self):
        # TODO: your code goes here
        if np.random.rand() > self.epsilon:  # Exploit
            return np.argmax(self.q)
        else:
            return np.random.choice(self.k)  # Explore


class SampleAverage(ActionValue):
    def __init__(self, k, epsilon):
        super().__init__(k, epsilon)
        # TODO: your code goes here
        # initialize all the action count to 0
        self.n = np.zeros(k)

    def reset(self):
        # TODO: your code goes here
        self.q = np.zeros(self.k)
        self.n = np.zeros(self.k)

    def update(self, a, r):
        # TODO: your code goes here
        self.n[a] += 1
        # TODO question: is the new estimate q funcnction is correct
        self.q[a] += (r - self.q[a]) / self.n[a]


class ConstantStepSize(ActionValue):
    def __init__(self, alpha, k, epsilon):
        super().__init__(k, epsilon)
        # TODO: your code goes here
        self.alpha = alpha

    def reset(self):
        # TODO: your code goes here
        self.q = np.zeros(self.k)

    def update(self, a, r):
        # TODO: your code goes here
        self.q[a] += self.alpha * (r - self.q[a])


def experiment(bandit, algorithm, steps, episode_seed=None):
    bandit.reset(episode_seed)
    algorithm.reset()

    rs = []
    best_action_taken = []

    # TODO: implement the experiment loop
    for step in range(steps):
        action = algorithm.epsilon_greedy_policy()
        reward = bandit.step(action)
        algorithm.update(action, reward)

        rs.append(reward)
        # Check if the action taken was the best action
        optimal_action = bandit.best()  # Determine the current best action
        is_optimal = action == optimal_action  # Check if the chosen action was the best
        best_action_taken.append(int(is_optimal))  # Store 1 if true, 0 otherwise

    return np.array(rs), np.array(best_action_taken)


if __name__ == "__main__":
    N_bandit_runs = 300
    N_steps_for_each_bandit = 10000

    sample_average = SampleAverage(k=10, epsilon=0.1)
    constant = ConstantStepSize(k=10, epsilon=0.1, alpha=0.1)

    outputs = []

    for algo in [sample_average, constant]:
        # TODO: run multiple experiments (where N = N_bandit_runs)
        # you will need to compute the average reward across all experiments
        # you will also compute the percentage of times the best action is taken
        all_rewards = np.zeros(N_steps_for_each_bandit)
        all_optimal_actions = np.zeros(N_steps_for_each_bandit)

        for run in range(N_bandit_runs):
            bandit = NonStationaryBandit(k=10, bandit_seed=run)
            rewards, optimal_actions = experiment(bandit, algo, N_steps_for_each_bandit, run)

            all_rewards += rewards
            all_optimal_actions += optimal_actions

        average_rewards = all_rewards / N_bandit_runs
        ratio_optimal_actions = all_optimal_actions / N_bandit_runs

        outputs.append(average_rewards)
        outputs.append(ratio_optimal_actions)

    np.savetxt(sys.argv[1], outputs)
