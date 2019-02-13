from matplotlib.colors import LogNorm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def diffuse_vectorise(un, g, b, dt, dx2, a):
    """
    Takes a state, rate of decay, production, delta time, delta space and
    flux of molecule. Uses these data to compute next time state

    """
    return (un[1:-1] + a * dt/dx2 * (un[0:-2] - 2 * un[1:-1] + un[2:])) * g + b


def diffuse_point(c1, c2, c3, a, g, b, dt, dx2, q=1):
    """
    Takes 3 concentration points and diffuses between them
    uses a diffusiveness of PD to estimate passage through
    """
    def Q(A, q=q):
        return A * q

    return Q((c2 + a * dt/dx2 * (c1 - 2*c2 + c3)) * g + b)


def bs(c, th):
    """
    Provides a method for highlighting cells which
    qualify for a beta production
    """
    b = []
    for i in range(c.shape[0]):
        if c[i].sum() > th:
            b.append(i)
    return b


def stokes_einstein(x): return ((1.38e-23 * 298.15)/(6*np.pi * 8.9e-4 * x))


def C(dx, nt, a, dt, g, b, c, num_iter, bs, th, astimeseries=False, q=1):
    """
    takes a delta x, number of timepoints to compute, alpha diffusiveness of
    molecule delta time, gamma decay, beta production, current concentration
    matrix, number of iterations completed already and a function to derive
    cells which will produce the beta term
    """
    dx2 = dx**2
    start = num_iter
    cells = c.copy()
    for n in range(start, nt):
        production = bs(cells, th)
        for idx in range(cells.shape[0]):
            bb = 0
            if idx in production:
                bb = b
            u = cells[idx].copy()
            un = u.copy()
            u[1:-1] = diffuse_vectorise(un, g, bb, dt, dx2, a)

            if idx < cells.shape[0]-1:
                tmp_edge1 = diffuse_point(cells[idx+1][0],
                                          u[-1],
                                          u[-2], a, g, bb, dt, dx2, q=q)
            if idx > 0:
                u[0] = diffuse_point(u[1],
                                     u[0],
                                     cells[idx-1][-1], a, g, bb, dt, dx2, q=q)

            # Delay updated other edge until all is computed
            if idx < cells.shape[0]-1:
                u[-1] = tmp_edge1
            cells[idx] = u
        if astimeseries:
            yield cells.copy()
    yield cells


Xs = 10  # number of positions, per cell
N = 5  # -5 +5
cell_mm = 1  # big cells


def make_cell_states(q=1, t=60*60, r=3.5e-10):
    dx = cell_mm/Xs
    dt = 1
    g = 1
    cells = np.zeros((N, Xs))
    b = 0
    cells[cells.shape[0]//2] = 1
    th = 1
    a = stokes_einstein(r) * 1e+5  # mm per second ^2
    return C(dx, t, a, dt, g, b, cells, 0, bs, th, astimeseries=True, q=q)


def plot_final_state(cells, N, Xs):
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


def do_plot():
    states = make_cell_states()
    states = np.array([s for s in states])
    plot_final_state(states[-1], 11, 100)


def make_compare_figures():
    pass


def make_data_for_analysis(t=60*60, average=False):
    sa_data = {}
    cal_data = {}
    time = t
    num_res = 12
    for pd in [np.around(1-(0+(10*i)/100), 1) for i in range(0, 10)]:

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


if __name__ == '__main__':
    average = False
    ts = 60*60*14
    for s in [0.9]:
        fig, ax = plt.subplots()
        data = [i for i in make_cell_states(q=1, t=ts)]
        d14 = data[-1]
        d0 = data[0]
        print('Computed data')
        ax.plot(d0.ravel())
        ax.plot(d14.ravel())
