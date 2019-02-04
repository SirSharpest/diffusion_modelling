import numpy as np


def um_to_m(x): return x*1e-6


def um2_to_m2(x): return x*1e-12


def diff(x): return ((1.38e-23 * 298.15)/(6*np.pi * 8.9e-4 * x))


def nt_to_sec(x, dt): return x*(dt*(1/dt))


def diffuse_1D(nx, dx, nt, D, dt, prevState=None, prevIter=None, slow=False):
    dx2 = dx**2
    u = np.zeros(nx)
    u = prevState.copy()
    start = prevIter

    for n in range(start, nt):
        un = u.copy()  # Update previous values

        if not slow:
            u[1:-1] = un[1:-1] + D * dt/dx2*(un[0:-2] - 2 * un[1:-1] + un[2:])
        else:
            for i in range(1, nx-1):
                u[i] = un[i] + D * ((un[i+1] - 2 * un[i] + un[i-1])/dx2)
    return un


def diffuse_1D_better(nt, u, D, dt, dx2, mid_s, mid_e, production, burst=False):
    for i in range(nt):
        un = u.copy()
        un[mid_s:mid_e] = un[mid_s:mid_e] + production
        u[1:-1] = un[1:-1] + D * dt * ((un[2:] - (2 * un[1:-1]) + un[:-2])/dx2)
        if burst:
            if i == 60*60-1:
                u[mid_s:mid_e] = 1
                burst = False
                print('pop')
    return u
