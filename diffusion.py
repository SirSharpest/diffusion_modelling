import numpy as np
import matplotlib
matplotlib.use("Qt5Agg")


def diffuse_1D_exact_solution(x, t, D):
    return (1/np.sqrt(4*np.pi*D*t)
            * np.exp(- ((np.square(x))/(4*D*t))))


def diffuse_1D_discrete_solution(dx, nt, D, dt, ic, num_iter):
    dx2 = dx**2
    u = ic.copy()
    start = num_iter
    for n in range(start, nt):
        un = u.copy()
        u[1:-1] = un[1:-1] + D * dt/dx2*(un[0:-2] - 2 * un[1:-1] + un[2:])
    return un
