from typing import Iterable
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class PiApproximationWithNN(nn.Module):
    def __init__(self, state_dims, num_actions, alpha):
        """
        state_dims: the number of dimensions of state space
        action_dims: the number of possible actions
        alpha: learning rate
        """
        super(PiApproximationWithNN, self).__init__()

        # TODO: implement the rest here
        # Define network layers
        self.fc1 = nn.Linear(state_dims, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, num_actions)
        
        # Define optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=alpha, betas=(0.9, 0.999))



    def forward(self, states, return_prob=False):
        # TODO: implement this method

        # Note: You will want to return either probabilities or an action
        # Depending on the return_prob parameter
        # This is to make this function compatible with both the
        # update function below (which needs probabilities)
        # and because in test cases we will call pi(state) and 
        # expect an action as output.

        # Check if the input is a NumPy array and convert it to a tensor if necessary
        if isinstance(states, np.ndarray):
            states = torch.from_numpy(states).float()
        
        # Ensure states have the correct shape (add a batch dimension if missing)
        if states.dim() == 1:
            states = states.unsqueeze(0)

        x = F.relu(self.fc1(states))
        x = F.relu(self.fc2(x))
        action_probs = F.softmax(self.fc3(x), dim=-1)

        if return_prob:
            return action_probs
        else:
            # Sample an action and convert to a Python scalar
            action = torch.multinomial(action_probs, num_samples=1).squeeze().item()
            return action

    def update(self, states, actions_taken, gamma_t, delta):
        """
        states: states
        actions_taken: actions_taken
        gamma_t: gamma^t
        delta: G-v(S_t,w)
        """
        # TODO: implement this method
        
        self.optimizer.zero_grad()
        probs = self.forward(states, return_prob=True)

        # Convert actions_taken to a tensor if it's an integer
        if isinstance(actions_taken, int):
            actions_taken = torch.tensor([actions_taken], dtype=torch.int64)

        action_log_probs = torch.log(probs)
        action_taken_log_probs = action_log_probs.gather(1, actions_taken.unsqueeze(1)).squeeze(1)

        loss = -torch.sum(gamma_t * delta * action_taken_log_probs)
        loss.backward()
        self.optimizer.step()

class Baseline(object):
    """
    The dumbest baseline; a constant for every state
    There is no need to change this class.
    """
    def __init__(self,b):
        self.b = b
        
    def __call__(self, states):
        return self.forward(states)
        
    def forward(self, states) -> float:
        return self.b

    def update(self, states, G):
        pass

class VApproximationWithNN(nn.Module):
    def __init__(self, state_dims, alpha):
        """
        state_dims: the number of dimensions of state space
        alpha: learning rate
        """
        super(VApproximationWithNN, self).__init__()
        
        # TODO: implement the rest here
        
         # Define network layers
        self.fc1 = nn.Linear(state_dims, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=alpha, betas=(0.9, 0.999))

    def forward(self, states) -> float:
        # TODO: implement this method

        x = F.relu(self.fc1(states))
        x = F.relu(self.fc2(x))
        state_values = self.fc3(x)
        return state_values


    def update(self, states, G):
        # TODO: implement this method

        self.optimizer.zero_grad()
        values = self.forward(states)
        loss = F.mse_loss(values, G)
        loss.backward()
        self.optimizer.step()


def REINFORCE(
    env, #open-ai environment
    gamma:float,
    num_episodes:int,
    pi:PiApproximationWithNN,
    V:VApproximationWithNN) -> Iterable[float]:
    """
    implement REINFORCE algorithm with and without baseline.

    input:
        env: target environment; openai gym
        gamma: discount factor
        num_episode: #episodes to iterate
        pi: policy
        V: baseline
    output:
        a list that includes the G_0 for every episodes.
    """
    # TODO: implement this method
    episode_returns = []

    for episode_num in range(num_episodes):
        # Reset environment for the new episode
        state = env.reset()
        done = False
        episode_states, episode_actions, episode_rewards = [], [], []
        # Generate an episode
        while not done:
            state_tensor = torch.from_numpy(state).float().unsqueeze(0)
            action = pi(state_tensor, return_prob=False)
            next_state, reward, done, _ = env.step(action)

            episode_states.append(state_tensor)
            episode_actions.append(action)
            episode_rewards.append(reward)
            state = next_state
        
        # Calculate returns G for each time step
        G = 0
        returns = []
        for t in reversed(range(len(episode_rewards))):
            G = gamma * G + episode_rewards[t]
            returns.insert(0, G)
        
        for t in range(len(episode_states)):
            G_t = returns[t]  # The return following the current state
            state_tensor = episode_states[t]  # The current state tensor

            if isinstance(V, VApproximationWithNN):
                baseline_value = V(state_tensor).item()  # Get the estimated value for the current state
                delta = G_t - baseline_value  # Compute delta for NN baseline
                V.update(state_tensor, torch.tensor([G_t], dtype=torch.float32))  # Update NN baseline
            elif isinstance(V, Baseline):
                baseline_value = V(state_tensor)  # Get the constant baseline value
                delta = G_t - baseline_value  # Compute delta for constant baseline

            gamma_t = gamma ** t
            pi.update(state_tensor, torch.tensor([episode_actions[t]], dtype=torch.int64), gamma_t, delta)

        
        episode_returns.append(returns[0])

    return episode_returns
   
    
    