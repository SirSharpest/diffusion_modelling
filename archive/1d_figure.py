import numpy as np
import matplotlib.pyplot as plt
from helper_funcs import diff, diffuse_1D, diffuse_1D_better
from numpy import square


def make_fig():
    # Measurements in mm
    sx = 2
    dx = 0.1
    dx2 = square(dx)
    D = diff(3.5e-10) * 1e+6
    dt = 1  # dx2/(2*D*dx2)
    nx = int(sx/dx)

    nt = (60*60*4)
    nts = [str(n) for n in np.linspace(0, nt, 10, dtype=int)]  # 12 hrs

    # Calc initial state
    u = np.zeros(nx)
    production = 0
    mid_s = int(nx/2-5)
    mid_e = int(nx/2+5)
    u[mid_s:mid_e] = 1
    states = [u.copy()]

    for idx, n in enumerate(nts):
        if idx is 0:
            continue
        u = states[-1].copy()
        u = diffuse_1D_better(nt, u, D, dt, dx2, mid_s,
                              mid_e, production, False)
        states.append(u.copy())

        print(idx)
        print(u.sum())

    fig, axes = plt.subplots(len(nts), 1, sharex=True, sharey=True,
                             dpi=100, figsize=(10, 4))

    x = np.linspace(0, sx, nx)
    y = states[-1]
    extent = [x[0]-(x[1]-x[0])/2., x[-1]+(x[1]-x[0])/2., 0, 1]

    vmax = 0
    for n in states:
        if n.max() > vmax:
            vmax = n.max()

    heat = True
    if heat is True:
        for idx, d in enumerate(nts):
            axes[idx].imshow(states[idx][np.newaxis, :],
                             cmap="plasma", aspect="auto",
                             extent=extent, vmin=0, vmax=vmax)
            title_time = "{0}hrs : {1}minutes".format(
                int(d)//(60*60), (int(d)//60) % 60)
            axes[idx].set_title(title_time)

            for v in range(0, sx*10):
                axes[idx].axvline((v/10)+0.05, color='k', linestyle='solid')
                print(v)
            axes[idx].set_xlim(0, sx)

        plt.tight_layout()
        plt.show()
    else:
        for idx, d in enumerate(nts):
            axes[idx].plot(states[idx])
            title_time = "{0}hrs : {1}minutes".format(
                int(d)//(60*60), (int(d)//60) % 60)
            axes[idx].set_title(title_time)

            # for v in range(0, sx, 100):
            #     axes[idx//3, idx % 3].axvline(v, color='k', linestyle='solid')
            # axes[idx//3, idx % 3].set_xlim(0, sx)

        plt.tight_layout()
        plt.show()


make_fig()
