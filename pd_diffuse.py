from matplotlib.colors import LogNorm
from pylab import figure, cm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


def diffuse_vectorise(un, g, b, dt, dx2, a):
    return (un[1:-1] + a * dt/dx2 * (un[0:-2] - 2 * un[1:-1] + un[2:])) + b


def diffuse_point(c1, c2, c3, a, g, b):
    return Q(a*(c1 - 2*c2 + c3)) * g + b


def Q(A):
    return A * 1


def bs():
    return [cells.shape[0]//2]


def stokes_einstein(x): return ((1.38e-23 * 298.15)/(6*np.pi * 8.9e-4 * x))


def C(dx, nt, a, dt, g, b, c, num_iter, bs):
    dx2 = dx**2
    start = num_iter
    cells = c.copy()
    for n in range(start, nt):
        for idx in range(cells.shape[0]):
            u = cells[idx].copy()
            un = u.copy()
            u[1:-1] = diffuse_vectorise(un, g, b, dt, dx2, a)

            if idx < cells.shape[0]-1:
                u[-1] = diffuse_point(cells[idx+1][0],
                                      u[-1],
                                      u[-2], a, g, b)

            if idx > 0:
                u[0] = diffuse_point(u[1],
                                     u[0],
                                     cells[idx-1][-1], a, g, b)

            cells[idx] = u
    return cells


N = 11  # -5 +5
Xs = 100  # number of positions, per cell
cell_mm = 0.1  # big cells
dx = Xs/cell_mm  # difference in x
dx = 0.1
t = 60*60*2  # time in seconds
dt = 1
g = 1
b = 0
cells = np.zeros((N, Xs))
cells[cells.shape[0]//2] = 1


a = stokes_einstein(3.5e-10) * 1e+6  # mm per second ^2

cells = C(dx, t, a, dt, g, b, cells, 0)


fig, axes = plt.subplots(2, sharex=True)


# Plotting below here

x_labels_locations = np.linspace(0, N*Xs, num=11)
x_labels_locations = [50+(i*100) for i in range(11)]

x_labels = ['C{0}'.format(n) for n in range(11)]


axes[0].plot(cells.ravel())
axes[0].set_yscale('log')

pcm = axes[1].pcolormesh(np.expand_dims(cells.ravel(), axis=0), norm=LogNorm(
    vmin=cells[cells > 0].min(), vmax=cells.max()))

plt.sca(axes[1])
plt.xticks(x_labels_locations, x_labels)
plt.colorbar(pcm, orientation="horizontal", pad=0.2)
plt.tight_layout()
plt.show()
