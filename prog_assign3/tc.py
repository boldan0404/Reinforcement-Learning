import numpy as np
import math
from algo import ValueFunctionWithApproximation

class ValueFunctionWithTile(ValueFunctionWithApproximation):
    def __init__(self,
                 state_low:np.array,
                 state_high:np.array,
                 num_tilings:int,
                 tile_width:np.array):
        """
        state_low: possible minimum value for each dimension in state
        state_high: possible maximum value for each dimension in state
        num_tilings: # tilings
        tile_width: tile width for each dimension
        """
        # TODO: implement this method

        self.offset = []
        self.tiles = [] 
        self.tile_width = tile_width
        self.num_tilings = num_tilings

        for i in range(len(self.tile_width)):
            self.tiles.append((math.ceil((state_high[i] - state_low[i]) / tile_width[i]) + 1))
        for i in range(self.num_tilings): 
            self.offset.append((state_low - (i / num_tilings) * tile_width)) 

        self.weight = np.zeros(np.append(self.num_tilings, self.tiles)) 

    def get_active_tiles(self, s):
        features = np.zeros(np.append(self.num_tilings, self.tiles))
        indices = []
        for j in range(self.num_tilings):
            d = np.floor((s - self.offset[j]) / self.tile_width).astype(int)
            # Ensure indices are within bounds
            d = np.clip(d, 0, np.array(self.tiles) - 1)
            features[j][d[0]][d[1]] = 1
            indices.append([j, d[0], d[1]])
        return features, indices
    
    def __call__(self,s):
        # TODO: implement this method
        features, _ = self.get_active_tiles(s)
        return np.sum(self.weight * features)


    def update(self,alpha,G,s_tau):
        # TODO: implement this method
        features, indices = self.get_active_tiles(s_tau)
        value_estimate = np.sum(self.weight * features)
        delta = G - value_estimate
        for j, x, y in indices:
            self.weight[j][x][y] += alpha * delta
            
        return None
