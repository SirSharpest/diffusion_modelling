from matplotlib.colors import LogNorm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def diffuse_vectorise(un, g, b, dt, dx2, a):
    """
    Takes a state, rate of decay, production, delta time, delta space and
    flux of molecule. Uses these data to compute next time state

    """
    return (un[1:-1] + a  * (un[0:-2] - 2 * un[1:-1] + un[2:])/dx2) * g + b


def diffuse_point(c1, c2, c3, a, g, b, dt, dx2, q=1):
    """
    Takes 3 concentration points and diffuses between them
    uses a diffusiveness of PD to estimate passage through
    """
    def Q(A, q=q):
        return A *q

    return (c2 + a * (Q(c1) - 2*c2 + c3)/dx2) * g + b


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
                u[0] = diffuse_point(cells[idx-1][-1],
                                     u[0],       
                                     u[1],
                                     a, g, bb, dt, dx2, q=q)

            # Delay updated other edge until all is computed
            if idx < cells.shape[0]-1:
                u[-1] = tmp_edge1
            cells[idx] = u
        if astimeseries:
            yield cells.copy()
    yield cells



Xs = 10  # number of positions, per cell
N = 5  # -5 +5
cell_size_in_m = 0.00001


def make_cell_states(q=1, t=60*60, r=3.5e-10):
    dx = cell_size_in_m/Xs
    dt = 1
    g = 1
    cells = np.zeros((N, Xs))
    b = 0
    cells[cells.shape[0]//2] = 1
    th = 1
    a = (stokes_einstein(r) )   # mm per second ^2    
    return C(dx, t, a, dt, g, b, cells, 0, bs, th, astimeseries=True, q=q)


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

def analytical(x, t, D): return (1/np.sqrt(4*np.pi*D*t)
                        * np.exp(- ((np.square(x))/(4*D*t))))

if __name__ == '__main__':
    average = False
    ts = 60
    a = (stokes_einstein(3.5e-10)) 
    
    d = [i for i in make_cell_states(q=1, t=ts)][-1][N//2]
    #print(d[5])
#    for s in [0.9]:
#        fig, ax = plt.subplots()
#        data = [i for i in make_cell_states(q=1, t=ts)]
#        d14 = data[-1].ravel()
#        d0 = data[0]
#        print('Computed data')
#        #ax.plot(d0.ravel())
#        ax.plot(d14)
#        ax.set_ylim(0,1)
