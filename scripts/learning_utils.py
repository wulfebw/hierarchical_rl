
import random

def sample(probs):
    """
    :description: given a list of probabilities, randomly select an index into those probabilities
    """
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