import pyswarms as ps
from pyswarms.utils.functions import single_obj as fx
import numpy as np
from math import cos, exp, sin
from pyswarms.utils.plotters import plot_cost_history
import matplotlib.pyplot as plt
options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}

def compute_endurance(swarm_list):
    return -1 * (exp(-2 * (swarm_list[1] - sin(swarm_list[0])) ** 2) + sin(swarm_list[2] * swarm_list[3]) + cos(swarm_list[4] * swarm_list[5]))

def f(swarm):
    n_particles = swarm.shape[0]
    j = [compute_endurance(swarm[i]) for i in range(n_particles)]
    return np.array(j)


x_max = np.ones(6)
x_min = np.zeros(6)
my_bounds = (x_min, x_max)

optimizer = ps.single.GlobalBestPSO(n_particles=10, dimensions=6, options=options, bounds=my_bounds)
optimizer.optimize(f, iters=1000)
# Optimization finished | best cost: 2.0171312361938734, best pos: [1.00724141 1.00129715]
# best cost: 0.8995290942156188, best pos: [0.04116386 0.98076449 0.11148487 0.44813072 0.89189742 0.92503493]
#  best cost: -2.830355296351997, best pos: [0.25373547 0.2435184  0.99866873 0.99131776 0.32542352 0.32410701]
# Obtain cost history from optimizer instance
cost_history = optimizer.cost_history

# Plot!
plot_cost_history(cost_history)
plt.show()
