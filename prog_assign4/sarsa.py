import numpy as np
import math
class StateActionFeatureVectorWithTile():
    def __init__(self,
                 state_low:np.array,
                 state_high:np.array,
                 num_actions:int,
                 num_tilings:int,
                 tile_width:np.array):
        """
        state_low: possible minimum value for each dimension in state
        state_high: possible maimum value for each dimension in state
        num_actions: the number of possible actions
        num_tilings: # tilings
        tile_width: tile width for each dimension
        """
        # TODO: implement here
        self.state_low = state_low
        self.state_high = state_high
        self.num_actions = num_actions
        self.num_tilings = num_tilings
        self.tile_width = tile_width

        # Initialize lists for tiles and offsets
        self.tiles = []
        self.offset = []

        for i in range(len(self.tile_width)):
            self.tiles.append((math.ceil((state_high[i] - state_low[i]) / tile_width[i]) + 1))

        for i in range(self.num_tilings): 
            self.offset.append((state_low - (i / num_tilings) * tile_width)) 

        self.total_tiles = np.prod(self.tiles)
        self.d = self.num_actions * self.num_tilings * self.total_tiles

    def feature_vector_len(self) -> int:
        """
        return dimension of feature_vector: d = num_actions * num_tilings * num_tiles
        """
        # TODO: implement this method
        return self.d

    def __call__(self, s, done, a) -> np.array:
        """
        implement function x: S+ x A -> [0,1]^d
        if done is True, then return 0^d
        """

        if done:
            # Return a zero vector if the episode is done
            return np.zeros(self.d)
        
        # Initialize the feature vector
        feature_vector = np.zeros(self.d)
        
        for tiling_idx in range(self.num_tilings):
            # Calculate the adjusted state based on the offset
            adjusted_state = (np.array(s) - self.state_low + self.offset[tiling_idx]) / self.tile_width
            
            # Find the indices of the active tile in this tiling
            tile_indices = np.floor(adjusted_state).astype(int) % self.tiles
            
            # Convert multi-dimensional tile indices to a single index
            single_index = np.ravel_multi_index(tile_indices, self.tiles)
            
            # Calculate the index in the feature vector for this tiling and action
            index = (tiling_idx * self.total_tiles + single_index) * self.num_actions + a
            feature_vector[index] = 1
        
        return feature_vector

def SarsaLambda(
    env, # openai gym environment
    gamma:float, # discount factor
    lam:float, # decay rate
    alpha:float, # step size
    X:StateActionFeatureVectorWithTile,
    num_episode:int,
) -> np.array:
    """
    Implement True online Sarsa(\lambda)
    """

    def epsilon_greedy_policy(s,done,w,epsilon=.0):
        nA = env.action_space.n
        Q = [np.dot(w, X(s,done,a)) for a in range(nA)]

        if np.random.rand() < epsilon:
            return np.random.randint(nA)
        else:
            return np.argmax(Q)

    w = np.zeros((X.feature_vector_len()))

    #TODO: implement this function
   
    for episode in range (num_episode):
        S = env.reset()
        done = False
        A = epsilon_greedy_policy(S,done,w)
        x = X(S, done, A) 
        z = np.zeros_like(w) 
        Q_old = 0.0  # Old Q-value

        while not done:
            S_prime, R, done, _ = env.step(A)  # Take action A, observe R, S'
            A_prime = epsilon_greedy_policy(S_prime, done, w)  # Next action
            x_prime = X(S_prime, done, A_prime) if A_prime is not None else np.zeros_like(x)
            
            Q = np.dot(w, x)  # Q-value for current state-action pair
            Q_prime = np.dot(w, x_prime) if not done else 0.0  # Q-value for next state-action pair
            
            delta = R + gamma * Q_prime - Q  # Temporal difference error
            
            # Eligibility traces update
            z = gamma * lam * z + (1 - alpha * gamma * lam * np.dot(z, x)) * x
            
            # Weight update
            w += alpha * (delta + Q - Q_old) * z - alpha * (Q - Q_old) * x
            
            Q_old = Q_prime
            x = x_prime
            A = A_prime

    return w
