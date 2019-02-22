from json import dump as jsonsave
from matplotlib.colors import LogNorm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()


def diffuse_vectorise(un, g, b, dt, dx2, dy2, a):
    """
    Takes a state, rate of decay, production, delta time, delta space and
    flux of molecule. Uses these data to compute next time state

    """
    return (un[1:-1, 1:-1] + a *
            (((un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[:-2, 1:-1]))/dx2 +
             ((un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, :-2]) / dy2))) *\
        g + b


def diffuse_1d_point(c1, c2, c3, a, g, b, dt, dx2, q=1):
    """
    Takes 3 concentration points and diffuses between them
    uses a diffusiveness of PD to estimate passage through
    """

    return Q((c2 + a * dt/dx2 * (c1 - 2 * c2 + c3)) * g + b)


def Q(A, q=1):
    return A * q


def bs(c, th):
    """
    **** NEEDS TESTED ****
    Provides a method for highlighting cells which
    qualify for a beta production
    """
    b = []
    for i in range(c.shape[0]):
        if c[i].sum() > th:
            b.append(i)
    return b


def stokes_einstein(x): return ((1.38e-23 * 298.15) / (6 * np.pi * 8.9e-4 * x))


def handle_top_wall(cells, idx, idy, g, b, dt, dx2, dy2, a, q=1):

    x_matrix = np.zeros(cells.shape[-1])

    for x in range(1, cells.shape[-1] - 1):
        c1 = cells[idy, idx, 0, x]
        c2 = cells[idy, idx, 0, x+1]
        c3 = cells[idy, idx, 0, x-1]
        x_matrix[x] = diffuse_1d_point(c2, c1, c3, a, g, b, dt, dx2, q=1)

    return Q(cells[idy, idx, 0] + a * (((cells[idy-1, idx, -1] - 2 * cells[idy, idx, 0] + cells[idy, idx, 1]) / dy2) +
                                       (x_matrix)), q=q) * g + b


def handle_bottom_wall(cells, idx, idy, g, b, dt, dx2, dy2, a, q=1):
    x_matrix = np.zeros(cells.shape[-1])

    for x in range(1, cells.shape[-1] - 1):
        c1 = cells[idy, idx, -1, x]
        c2 = cells[idy, idx, -1, x+1]
        c3 = cells[idy, idx, -1, x-1]
        x_matrix[x] = diffuse_1d_point(c2, c1, c3, a, g, b, dt, dx2, q=q)

    return Q(cells[idy, idx, -1] + a * (((cells[idy+1, idx, 0] - 2 * cells[idy, idx, -1] + cells[idy, idx, -2]) / dy2) +
                                        (x_matrix)), q=q) * g + b


def handle_left_wall(cells, idx, idy, g, b, dt, dx2, dy2, a, q=1):
    y_matrix = np.zeros(cells.shape[-2])

    for x in range(1, cells.shape[-2] - 1):
        c1 = cells[idy, idx, x, 0]
        c2 = cells[idy, idx, x+1, 0]
        c3 = cells[idy, idx, x-1, 0]
        y_matrix[x] = diffuse_1d_point(c2, c1, c3, a, g, b, dt, dy2, q=1)

    return Q(cells[idy, idx, :, 0] + a * (((cells[idy, idx-1, :, -1] - 2 * cells[idy, idx, :, 0] + cells[idy, idx, :, 1]) / dx2) +
                                          (y_matrix)), q=q) * g + b


def handle_right_wall(cells, idx, idy, g, b, dt, dx2, dy2, a, q=1):
    y_matrix = np.zeros(cells.shape[-2])

    for x in range(1, cells.shape[-2] - 1):
        c1 = cells[idy, idx, x, -1]
        c2 = cells[idy, idx, x+1, -1]
        c3 = cells[idy, idx, x-1, -1]
        y_matrix[x] = diffuse_1d_point(c2, c1, c3, a, g, b, dt, dy2, q=1)

    return Q(cells[idy, idx, :, -1] + a * (((cells[idy, idx+1, :, 0] - 2 * cells[idy, idx, :, -1] + cells[idy, idx, :, -2]) / dx2) +
                                           (y_matrix)), q=q) * g + b


def C(dx, dy, nt, a, dt, g, b, c, num_iter, bs, th, astimeseries=False, q=1):
    # TODO: Upgrade to 2D
    """
    takes a delta x, number of timepoints to compute, alpha diffusivenes of
    molecule delta time, gamma decay, beta production, current concentration
    matrix, number of iterations completed already and a function to derive
    cells which will produce the beta term
    """
    dx2 = dx**2
    dy2 = dy**2
    start = num_iter
    cells = c.copy()
    for n in range(start, nt):
        production = bs(cells, th)  # TODO: Test
        for idy in range(cells.shape[0]):  # Cells in Y axis
            for idx in range(cells.shape[1]):  # Cells in X axis
                tl, tr, tt, tb = (None, None, None, None)
                bb = 0
                if idx in production:
                    bb = b
                u = cells[idy, idx].copy()
                un = u.copy()  # TODO: Possibly don't need so many copies
                u[1:-1, 1:-1] = diffuse_vectorise(un, g, bb, dt, dx2, dy2, a)

                # # TODO: MAJOR REWORK
                if idx > 0:
                    tl = handle_left_wall(cells,
                                          idx, idy, g,
                                          bb, dt,
                                          dx2, dy2, a)
                if idx < cells.shape[0]-1:
                    tr = handle_right_wall(cells,
                                           idx, idy, g,
                                           bb, dt,
                                           dx2, dy2, a)

                if idy > 0:
                    tt = handle_top_wall(cells,
                                         idx, idy,
                                         g, bb, dt,
                                         dx2, dy2, a)
                if idy < cells.shape[0]-1:
                    tb = handle_bottom_wall(cells,
                                            idx, idy, g,
                                            bb, dt,
                                            dx2, dy2, a)
                if tl is not None:
                    u[:, 0] = tl
                if tr is not None:
                    u[:, -1] = tr
                if tt is not None:
                    u[0, :] = tt
                if tb is not None:
                    u[-1, :] = tb
                cells[idy, idx] = u
        if astimeseries:
            yield cells.copy()
    yield cells


Xs = 10  # number of x positions, per cell
Ys = 10  # number of x positions, per cell
Nx = 5  # -5 +5
Ny = 5
cell_mm = 0.05  # big cells


def make_cell_states(q=1, t=60*60, r=3.5e-10):
    dx = 1
    dy = 1
    dt = 30
    g = 1
    cells = np.zeros((Nx, Ny, Xs, Ys))
    b = 0
    cells[Nx//2, Ny//2] = 1

    th = 1
    a = stokes_einstein(r) * 1e+6  # mm per second ^2
    return C(dx, dy, t, a, dt, g, b, cells, 0, bs, th, astimeseries=True, q=q)


def make_data_for_analysis(t=60*60, chem_space=3.5e-10, average=False):
    chemical_data = {}
    time = t
    num_res = 13
    for c in chem_space:
        tmp_data = {}
        for pd in np.around(np.linspace(0, 0.2, num=20), 3):
            tmp_data['pd_{0}'.format(pd)] = {}
            for k, v in enumerate(make_cell_states(q=pd, t=time, r=c)):
                s1, s2, s3, s4 = v.shape
                tmp_data['pd_{0}'.format(pd)][k] = np.around(
                    np.mean(v.reshape(s1, s2, s3*s4), axis=2), num_res)
        chemical_data[c] = tmp_data
    return chemical_data


if __name__ == '__main__':
    average = False
    ts = 2
    m_dat = {}
    for ch in np.linspace(1e-10, 4e-10, num=20):
        dat = make_data_for_analysis(t=ts, chem_space=[ch], average=average)
        for i in dat.keys():
            for p in dat[i].keys():
                for a in dat[i][p].keys():
                    dat[i][p][a] = dat[i][p][a].tolist()
                    m_dat['{0}_{1}_{2}_{3}'.format(ch, i, p, a)] = dat[i][p][a]
                # jsonsave(dat[i][p], open('./json/data_{0}_{1}.json'.format(i, p), 'w', encoding='utf-8'),
                #          separators=(',', ':'), sort_keys=True, indent=4)

    jsonsave(m_dat, open('master_data_2secs.json', 'w', encoding='utf-8'),
             separators=(',', ':'), sort_keys=True, indent=4)
