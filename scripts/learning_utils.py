
import glob
from math import sqrt, ceil
import matplotlib.pyplot as plt
import numpy as np
import os
import random

def sample(probs):
    """
    :description: given a list of probabilities, randomly select an index into those probabilities
    """
    if len(probs) < 1:
      raise ValueError('Sample received an empty list of probabilities. This should not happen. ')

    target = random.random()
    accum = 0
    for i, prob in enumerate(probs):
        accum += prob
        if accum >= target: return i
    raise ValueError('Invalid probabilities provided to sample method in experiment')

# Function: Weighted Random Choice
# --------------------------------
# Given a dictionary of the form element -> weight, selects an element
# randomly based on distribution proportional to the weights. Weights can sum
# up to be more than 1. 
# source: stanford.cs221.problem_set_6
# may be beneficial to switch to a faster method
def weightedRandomChoice(weightDict):
    weights = []
    elems = []
    for elem in weightDict:
        weights.append(weightDict[elem])
        elems.append(elem)
    total = sum(weights)
    key = random.uniform(0, total)
    runningTotal = 0.0
    chosenIndex = None
    for i in range(len(weights)):
        weight = weights[i]
        runningTotal += weight
        if runningTotal > key:
            chosenIndex = i
            return elems[chosenIndex]
    raise Exception('Should not reach here')

def visualize_grid(Xs, ubound=255.0, padding=1):
    """
    Reshape a 4D tensor of image data to a grid for easy visualization.

    Inputs:
    - Xs: Data of shape (N, H, W, C)
    - ubound: Output grid will have values scaled to the range [0, ubound]
    - padding: The number of blank pixels between elements of the grid
    """
    (N, H, W, C) = Xs.shape
    grid_size = int(ceil(sqrt(N)))
    grid_height = H * grid_size + padding * (grid_size - 1)
    grid_width = W * grid_size + padding * (grid_size - 1)
    grid = np.zeros((grid_height, grid_width, C))
    next_idx = 0
    y0, y1 = 0, H
    for y in xrange(grid_size):
        x0, x1 = 0, W
        for x in xrange(grid_size):
            if next_idx < N:
                img = Xs[next_idx]
                low, high = np.min(img), np.max(img)
                grid[y0:y1, x0:x1] = ubound * (img - low) / (high - low)
                # grid[y0:y1, x0:x1] = Xs[next_idx]
                next_idx += 1
            x0 += W + padding
            x1 += W + padding
        y0 += H + padding
        y1 += H + padding
    # grid_max = np.max(grid)
    # grid_min = np.min(grid)
    # grid = ubound * (grid - grid_min) / (grid_max - grid_min)
    return grid

def get_run_directory(filepath):
    return filepath[:filepath.rindex('/')]

def get_value_array_from_value_image_file(filepath):
    lines = None
    with open(filepath, 'rb') as f:
        lines = f.readlines()
        lines = [line.replace('\n', '').split(' ') for line in lines]
        lines = [[val for val in line if val != ''] for line in lines]
        lines = [[float(val) for val in line] for line in lines]
        lines = np.array(lines)
    return lines

def make_heat_map(filepath, epoch):
    # convert value image to numeric array
    value_array = get_value_array_from_value_image_file(filepath)
    if value_array is None:
        print 'Value image could not be converted to heatmap'
        return

    # determine output filepath
    run_dir = get_run_directory(filepath)
    output_filepath = os.path.join(run_dir, 'heatmaps', 'value_heatmap_{}.png'.format(epoch))

    # create and save heatmap
    heatmap = plt.pcolormesh(value_array)
    plt.colorbar()
    plt.savefig(output_filepath)
    plt.close()

def load_params(filepath):
    params = np.load(filepath)['params']
    return params
        

if __name__ =='__main__':
    make_heat_maps()

