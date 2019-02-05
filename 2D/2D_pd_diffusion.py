from matplotlib.colors import LogNorm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm


def diffuse_vectorise(un, g, b, dt, dx2, dy2, a):
    """
    Takes a state, rate of decay, production, delta time, delta space and
    flux of molecule. Uses these data to compute next time state

    """
    return (un[1:-1, 1:-1] + a *
            (((un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[:-2, 1:-1]) * dt / dx2) +
             ((un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, :-2]) * dt / dy2))) *\
        g + b


def diffuse_2d_point(c1, c2, c3, c4, c5, dx2, dy2, dt, q=1):
    """
    Where:
    C1 is Cell_i,j

    C2 is Cell_-1,j
    C3 is Cell_+1,j

    C4 is Cell_i,+1
    C5 is Cell_i,-1
    """
    def Q(A, q=q):
        return A * q
    Q(c1 + (a * ((c2 - 2 * c1 + c3) * dt / dx2) + ((c4 - 2 * c1 + c5) * dt / dx2)))


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

    return (cells[idy, idx, -1] + a * (((cells[idy+1, idx, 0] - 2 * cells[idy, idx, -1] + cells[idy, idx, -2]) / dy2) +
                                 ((x_matrix) / dx2))) * g + b

def handle_left_wall(cells, idx, g, bb, dt, dx2, dy2, a):
    return 0


def handle_right_wall(cells, idx, g, bb, dt, dx2, dy2, a):
    return 0


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
                # if idx > 0:
                #     tl = handle_left_wall(cells,
                #                           idx, g,
                #                           bb, dt,
                #                           dx2, dy2, a)
                # if idx < cells.shape[0]-1:
                #     tr = handle_right_wall(cells,
                #                            idx, g,
                #                            bb, dt,
                #                            dx2, dy2, a)
                
                if idy >0:
                    tt = handle_top_wall(cells,
                                         idx, idy, 
                                         g, bb, dt,
                                         dx2, dy2, a)
                if idy < cells.shape[0]-1:
                    tb = handle_bottom_wall(cells,
                                             idx, idy, g,
                                             bb, dt,
                                             dx2, dy2, a)

                # # TODO: HANDLE CORNER PROBLEM
                # if tl:
                #     u[0, :] = tl
                # if tr:
                #     u[-1, :] = tr
                if tt is not None:
                     u[0, :] = tt
                if tb is not None:
                     u[-1, :] = tb
                cells[idy,idx] = u
        if astimeseries:
            yield cells.copy()
    yield cells


Xs = 10  # number of x positions, per cell
Ys = 10  # number of x positions, per cell
Nx = 5  # -5 +5
Ny = 5
cell_mm = 0.05  # big cells


def make_cell_states(q=1, t=60*60, r=3.5e-10):
    # TODO: Upgrade to 2D
    dx = 0.1
    dy = 0.1
    dt = 1
    g = 1
    cells = np.zeros((Nx, Ny, Xs, Ys))
    b = 0
    cells[4, 3][2:6,2:6] = 1 # confirmation for internal diffusion working
    cells[1,1] = 1
    cells[0,0][0,0] = 1
    th = 1
    a = stokes_einstein(r) * 1e+6  # mm per second ^2
    return C(dx, dy, t, a, dt, g, b, cells, 0, bs, th, astimeseries=True, q=q)


def plot_final_state(cells, N, Xs):
    # TODO: Upgrade to 2D
    sns.set()
    fig, axes = plt.subplots(2)
    x_labels_locations = np.linspace(0, N, num=11)

    x_labels = ['C{0}'.format(n) for n in range(11)]

    axes[0].plot(cells)
    axes[0].set_ylim(1e-12, 10)
    axes[0].set_yscale('log')
    axes[0].set_xlim(0, 10)

    pcm = axes[1].pcolormesh(np.expand_dims(cells,
                                            axis=0),
                             norm=LogNorm(vmin=1e-12,
                                          vmax=1))

    plt.sca(axes[1])
    plt.xticks(x_labels_locations, x_labels)
    plt.colorbar(pcm, orientation="horizontal", pad=0.2)
    plt.tight_layout()
    plt.show()


def make_data_for_analysis(t=60*60, average=False):
    # TODO: Upgrade to 2D
    sa_data = {}
    cal_data = {}
    time = t
    num_res = 13
    for pd in [np.around(1-(0+(10*i)/100), 1) for i in range(0, 1)]:

        if average:
            sa_data['pd_{0}'.format(pd)] = {
                k: np.around(np.median(v, axis=1), num_res).tolist()
                for k, v in enumerate(make_cell_states(q=pd,
                                                       t=time))}

            cal_data['pd_{0}'.format(pd)] = {
                k: np.around(np.median(v, axis=1), num_res).tolist()
                for k, v in enumerate(make_cell_states(q=pd,
                                                       t=time,
                                                       r=1.94e-10))}

        else:
            sa_data['pd_{0}'.format(pd)] = {
                k: np.around(v, num_res)
                for k, v in enumerate(make_cell_states(q=pd,
                                                       t=time))}

            cal_data['pd_{0}'.format(pd)] = {
                k: np.around(v, num_res)
                for k, v in enumerate(make_cell_states(q=pd,
                                                       t=time,
                                                       r=1.94e-10))}

    data = {'sa': sa_data, 'cal': cal_data}

    return data


def plot_data(data, chem, pd, tp):
    fig, axes = plt.subplots(data[chem][pd][tp].shape[0], data[chem][pd][tp].shape[1])
    for y in range(len(data[chem][pd][tp])):
        for x in range(len(data[chem][pd][tp])):
            axes[y,x].pcolormesh(data[chem][pd][tp][y,x], norm=LogNorm(vmin=1e-4, vmax=1), cmap='plasma')
            axes[y,x].set_xticks([])   
            axes[y,x].set_yticks([])   
            axes[y,x].invert_yaxis()
        fig.suptitle('Seconds: {0}'.format(tp))


if __name__ == '__main__':
    average = False
    ts = 60*10
    dat = make_data_for_analysis(t=ts, average=average)
    
    for i in np.linspace(1, ts, num=4, dtype=int):
        plot_data(dat, 'sa', 'pd_1.0', i)        
    # TEsting funcs
    get_middle = lambda x,y,z: dat['sa']['pd_1.0'][x][y,z]
    