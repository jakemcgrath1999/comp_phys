#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 14:32:21 2022

@author: jakemcgrath
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

class PercolationSimulation:
    """
    A simulation of a 2D directed percolation problem. Given a 2D lattice, blocked sites
    are denoted by 0s, and open sites are denoted by 1s. During a simulation, water is
    poured into the top of the grid, and allowed to percolate to the bottom. If water
    fills a lattice site, it is marked with a 2 in the grid. Water only reaches a site
    if it reaches an open site directly above, or to the immediate left or right 
    of an open site.

    Attributes:
        grid (np.array): the original lattice of blocked (0) and open (1) sites
        grid_filled (np.array): the lattice after water has been poured in
        n (int): number of rows and columns in the lattice
        p (float): probability of a site being blocked in the randomly-sampled lattice
            random_state (int): random seed for the random number generator
        random_state (int): random seed for numpy's random number generator. Used to 
            ensure reproducibility across random simulations. The default value of None
            will use the current state of the random number generator without resetting
            it.
    """

    def __init__(self, n=100, p=0.5, grid=None, random_state=None):
        """
        Initialize a PercolationSimulation object.

        Args:
            n (int): number of rows and columns in the lattice
            p (float): probability of a site being blocked in the randomly-sampled lattice
            random_state (int): random seed for numpy's random number generator. Used to
                ensure reproducibility across random simulations. The default value of None
                will use the current state of the random number generator without resetting
                it.
        """
        self.random_state = random_state # the random seed
        # Initialize a random grid if one is not provided. Otherwise, use the provided grid.
        if grid is None:
            self.n = n
            self.p = p
            self.grid = np.zeros((n, n))
            self._initialize_grid()
        else:
            assert len(np.unique(np.ravel(grid))) <= 2, "Grid must only contain 0s and 1s"
            self.grid = grid.astype(int)
            # override numbers if grid is provided
            self.n = grid.shape[0]
            self.p = 1 - np.mean(grid)

        self.grid_filled = np.copy(self.grid)   # save grid to grid_filled - to be filled in later


    def _initialize_grid(self):
        """
        Sample a random lattice for the percolation simulation...
        Set random seed, then write grid as n by n lattice of blocked or unblocked (0 or 1)
        with probablity p or 1-p.  Send copy of grid to grid_filled
        """
        np.random.seed(self.random_state)   # set the random seed
        # inicialize n by n grid of blocked and open (0s or 1s) with prob p or 1-p
        self.grid = np.random.choice([0,1], size = [self.n, self.n], p = [self.p, 1-self.p])
        self.grid_filled = np.copy(self.grid)   # send copy of grid to grid_filled


    def _flow_recursive(self, i, j):
        """
        The recursive portion of the flow simulation. Notice how grid and grid_filled
        are used to keep track of the global state, even as our recursive calls nest
        deeper and deeper
        """
        # Base cases of the problem, return None and begin to back out of recursion
        if i < 0 or i >= self.n:
            return None
        if j < 0 or j >= self.n:
            return None
        # skip blocked sites, return None and begin to back out of recursion
        if self.grid_filled[i, j] == 0:
            return None
        # skip sites with water, return None and begin to back out of recursion
        if self.grid_filled[i, j] == 2:
            return None
        
        self.grid_filled[i, j] = 2      # otherwise, fill the grid site with water
        self._flow_recursive(i + 1, j)  # recursive call to von Neumann neighborhood
        self._flow_recursive(i - 1, j)
        self._flow_recursive(i, j + 1)
        self._flow_recursive(i, j - 1)


    def _flow(self):
        """
        This method writes to the grid and grid_filled attributes, but it does not
        return anything.
        """
        # start recursive method from top row of lattice... water will then trickle down
        # with the recursive method
        for w in range(self.n):
            self._flow_recursive(0, w)


    def percolate(self):
        """
        Initialize a random lattice and then run a percolation simulation. Report results
        """
        self._flow()    # Run the flow algorithm and report the results
        bottom = self.grid_filled[-1,:]     # get the bottom row of the filled grid
        if np.any(bottom == 2):     # if we see water, then it percolates
            return True
        else:                       # otherwise, it does not
            return False
        

def plot_percolation(mat):
    """
    Plots a percolation matrix, where 0 indicates a blocked site, 1 indicates an empty 
    site, and 2 indicates a filled site
    """
    cvals  = [0, 1, 2]
    colors = [(0, 0, 0), (0.4, 0.4, 0.4), (0.372549, 0.596078, 1)]

    norm = plt.Normalize(min(cvals), max(cvals))
    tuples = list(zip(map(norm,cvals), colors))
    cmap = LinearSegmentedColormap.from_list("", tuples)
    plt.imshow(mat, cmap=cmap, vmin=0, vmax=2)


model = PercolationSimulation(n=20, random_state=0, p=0.1)
print(model.percolate())
plt.figure()
plot_percolation(model.grid_filled)

model = PercolationSimulation(n=20, random_state=0, p=0.4)
print(model.percolate())
plt.figure()
plot_percolation(model.grid_filled)

model = PercolationSimulation(n=20, random_state=0, p=0.9)
print(model.percolate())
plt.figure()
plot_percolation(model.grid_filled)

pvals = np.linspace(0, 1, 25) # control parameter for percolation phase transition
n_reps = 200 # number of times to repeat the simulation for each p value

all_percolations = list()
for p in pvals:
    print("Running replicate simulations for p = {}".format(p), flush=True)
    all_replicates = list()
    for i in range(n_reps):
        # Initialize the model
        model = PercolationSimulation(30, p=p)
        all_replicates.append(model.percolate())
    all_percolations.append(all_replicates)

plt.figure()
plt.plot(pvals, np.mean(np.array(all_percolations), axis=1))
plt.xlabel('Average site occupation probability')
plt.ylabel('Percolation probability')

plt.figure()
plt.plot(pvals, np.std(np.array(all_percolations), axis=1))
plt.xlabel('Average site occupation probability')
plt.ylabel('Standard deviation in site occupation probability')

plt.show()


## Just from curiousity, plot the distribution of cluster sizes at the percolation threshold
## why does it appear to be bimodal?
all_cluster_sizes = list()
p_c = 0.407259
n_reps = 5000
for i in range(n_reps):
    model = PercolationSimulation(100, p=p_c)
    model.percolate()
    cluster_size = np.sum(model.grid_filled == 2)
    all_cluster_sizes.append(cluster_size)

    if i % 500 == 0:
        print("Finished simulation {}".format(i), flush=True)

all_cluster_sizes = np.array(all_cluster_sizes)

plt.figure()
plt.hist(all_cluster_sizes, 50);