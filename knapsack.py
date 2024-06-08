# A Dynamic Programming based Python Program for 0-1 Knapsack problem
# Returns the maximum value that can be put in a knapsack of capacity W
import numpy as np
from ortools.algorithms.python import knapsack_solver as kps

def knapsack_ortools(values, weights, capacity):
    osolver = kps.KnapsackSolver(kps.KNAPSACK_DYNAMIC_PROGRAMMING_SOLVER , 'test')
    scale = 1000
    values = np.array(values)
    weights = np.array(weights)
    values = (values * scale).astype(np.int64)
    weights = (weights).astype(np.int64)
    capacity = capacity

    osolver.init(values.tolist(), [weights.tolist()], [capacity])
    computed_value = osolver.solve()
    packed_items = [x for x in range(0, len(weights))
                    if osolver.best_solution_contains(x)]

    return packed_items