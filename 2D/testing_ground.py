from node_diffusion import diffuse, stokes_einstein
import matplotlib.pyplot as plt
import numpy as np


# Number of time points, in seconds
nt = 60*60*4
# delta t is just one
dt = 1
# Number of cells per direction
Xs, Ys = 11, 1
# Using moss cell measurements
cell_um = 100
dx2, dy2 = cell_um**2, cell_um**2
# Chem radius in meters
r = 3.5e-10

# Important model parameters
D = stokes_einstein(r) * 1e+12
q = 1

D_eff = (D*q*cell_um)/(D+q*cell_um)


# Set IC and add pulse in centre
ic = np.zeros((Ys, Xs))
ic[Ys//2, Xs//2] = 1


cur_state = ic


fig, ax = plt.subplots(1)

for i in range(nt):
    cur_state = diffuse(cur_state, dx2, D, q, 0)


ax.plot(cur_state.ravel())
ax.set_title('{0}'.format(cur_state.sum()))

plt.show()


cur_state
