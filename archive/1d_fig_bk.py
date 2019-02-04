import numpy as np
import matplotlib.pyplot as plt
from helper_funcs import diff, diffuse_1D, diffuse_1D_better
from numpy import square

# Measurements in mm
sx = 7
dx = 0.1
dx2 = square(dx)
D = diff(3.5e-10) * 1e+6
dt = 1  # dx2/(2*D*dx2)
nx = int(sx/dx)

nt = (60*60)
nts = [str(n) for n in np.linspace(0, nt, 6, dtype=int)]  # 12 hrs

# Calc initial state
u = np.zeros(nx)
production = D*2
mid_s = int(nx/2-5)
mid_e = int(nx/2+5)
u[mid_s:mid_e] = production
states = [u.copy()]


for idx, n in enumerate(nts):
    if idx is 0:
        continue
    u = states[-1].copy()
    u = diffuse_1D_better(nt, u, D, dt, dx2, mid_s, mid_e, production, True)
    states.append(u.copy())

    print(idx)
    print(u.sum())

fig, axes = plt.subplots(2, 3, sharex=True, sharey=True,
                         dpi=100, figsize=(10, 2))


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
        axes[idx//3, idx % 3].imshow(states[idx][np.newaxis, :],
                                     cmap="plasma", aspect="auto",
                                     extent=extent, vmin=0, vmax=vmax)
        title_time = "{0}hrs : {1}minutes".format(
            int(d)//(60*60), (int(d)//60) % 60)
        axes[idx//3, idx % 3].set_title(title_time)

        for v in range(0, sx):
            axes[idx//3, idx % 3].axvline(v, color='k', linestyle='solid')
        axes[idx//3, idx % 3].set_xlim(0, sx)

    plt.tight_layout()
    plt.show()
else:
    for idx, d in enumerate(nts):
        axes[idx//3, idx % 3].plot(states[idx])
        title_time = "{0}hrs : {1}minutes".format(
            int(d)//(60*60), (int(d)//60) % 60)
        axes[idx//3, idx % 3].set_title(title_time)

        # for v in range(0, sx, 100):
        #     axes[idx//3, idx % 3].axvline(v, color='k', linestyle='solid')
        # axes[idx//3, idx % 3].set_xlim(0, sx)

    plt.tight_layout()
    plt.show()
