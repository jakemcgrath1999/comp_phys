#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 14:33:13 2022

@author: jakemcgrath
"""
import numpy as np
import matplotlib.pyplot as plt

class AbelianSandpile:
    """
    An Abelian sandpile model simulation. The sandpile is initialized with a random
    number of grains at each lattice site. Then, a single grain is dropped at a random
    location. The sandpile is then allowed to evolve until it is stable. This process
    is repeated n_step times.

    A single step of the simulation consists of two stages: a random sand grain is 
    dropped onto the lattice at a random location. Then, a set of avalanches occurs
    causing sandgrains to get redistributed to their neighboring locations.
    
    Parameters:
    n (int): The size of the grid
    grid (np.ndarray): The grid of the sandpile
    history (list): A list of the sandpile grids at each timestep
    all_durations (list): A list of the durations of each avalanche
    """

    def __init__(self, n=100, random_state=None):
        self.n = n # size of lattice (default 100)
        np.random.seed(random_state) # set the random seed
        self.grid = np.random.choice([0, 1, 2, 3], size=(n, n)) # define n by n grid of 0s, 1s, 2s, or 3s
        self.history =[self.grid.copy()] # store a copy of the grid
        self.all_durations = list() # list to track the number of duractions at each drop

    def step(self):
        """
        Perform a single step of the sandpile model. Step corresponds a single sandgrain 
        addition and the consequent toppling it causes. 

        Returns: None
        """
        sand_drop = np.zeros((self.n,self.n), dtype = int)  # define a n by n matrix of zeros
        row, col = np.random.choice(self.n, 2)  # select a random place in the matrix 
        sand_drop[row, col] = 1     # add a sand grain to that position
        self.grid += sand_drop     # add the random sand particle to the original grid
        
        counts = 0      # keep track of number of topple events
        while np.amax(self.grid) >= 4:      # condition that piles need less than 4 grains
            i, j = np.where(self.grid >= 4) # find where our tall sand piles are
            self.grid[i[0], j[0]] -= 4      # topple event
            if i[0] < self.n - 1:   # BC: if we're on the grid, otherwise that grian falls off
                self.grid[i[0]+1, j[0]] += 1    # grains spread out
            if i[0] > 0:    # BC: if we're on the grid, otherwise that grian falls off
                self.grid[i[0]-1, j[0]] += 1    # grains spread out
            if j[0] < self.n - 1:   # BC: if we're on the grid, otherwise that grian falls off
                self.grid[i[0], j[0]+1] += 1    # grains spread out
            if j[0] > 0:    # BC: if we're on the grid, otherwise that grian falls off
                self.grid[i[0], j[0]-1] += 1    # grains spread out
            counts += 1     # increase topple event counter
            
        self.all_durations.append(counts)   # update our topple counter history list
        self.history.append(self.grid.copy())   # store new copy of our grid

    @staticmethod
    def check_difference(grid1, grid2):
        """Check the total number of different sites between two grids"""
        return np.sum(grid1 != grid2)

    def simulate(self, n_step):
        """ Simulate the sandpile model for n_step steps """
        for i in range(n_step):
            self.step()
        return self.grid

# Run sandpile simulation
model = AbelianSandpile(n=100, random_state=0)

plt.figure()
plt.imshow(model.grid, cmap='gray')

model.simulate(10000)
plt.figure()
plt.imshow(model.grid, cmap='gray')
plt.title("Final state")

# Compute the pairwise difference between all observed snapshots. This command uses list
# comprehension, a zip generator, and argument unpacking in order to perform this task
# concisely.
all_events =  [model.check_difference(*states) for states in zip(model.history[:-1], model.history[1:])]
# remove transients before the self-organized critical state is reached
all_events = all_events[1000:]
# index each timestep by timepoint
all_events = list(enumerate(all_events))
# remove cases where an avalanche did not occur
all_avalanches = [x for x in all_events if x[1] > 1]
all_avalanche_times = [item[0] for item in all_avalanches]
all_avalanche_sizes= [item[1] for item in all_avalanches]
all_avalanche_durations = [event1 - event0 for event0, event1 in zip(all_avalanche_times[:-1], all_avalanche_times[1:])]

log_bins = np.logspace(np.log10(2), np.log10(np.max(all_avalanche_durations)), 50) # logarithmic bins for histogram
vals, bins = np.histogram(all_avalanche_durations, bins=log_bins)
plt.figure()
plt.loglog(bins[:-1], vals, '.', markersize=10)
plt.title('Avalanche duration distribution')
plt.xlabel('Avalanche duration')
plt.ylabel('Count')

## Visualize activity of the avalanches
# Make an array storing all pairwise differences between the lattice at successive
# timepoints
all_diffs = np.abs(np.diff(np.array(model.history), axis=0))
all_diffs[all_diffs > 0] = 1
all_diffs = all_diffs[np.sum(all_diffs, axis=(1, 2)) > 1] # Filter to only keep big events
most_recent_events = np.sum(all_diffs[-100:], axis=0)
plt.figure(figsize=(5, 5))
plt.imshow(most_recent_events)
plt.title("Avalanch activity in most recent timesteps")