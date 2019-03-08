import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



sns.set()
plt.close('all')


def diffuse_2D(nx, dx, dy, nt, D, dt, prevState=None, prevIter=None):
    dx2 = dx**2
    dy2 = dy**2
    u = np.zeros((nx, nx))
    mid = int(nx/2)

    u[int(mid-(mid/4)):int(mid+(mid/4)),
      int(mid-(mid/4)):int(mid+(mid/4))] = 1

    for n in range(1, nt):
        un = u.copy()  # Update previous values
        u[1:-1, 1:-1] = un[1:-1, 1:-1] + D * \
            (((un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[:-2, 1:-1])/dx2) +
             ((un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, :-2])/dy2))
    return u


nx = 10  # Number of x measurements
dx, dy = 1, 1   # Change in X & Y
nt = 6  # Number of timesteps to make in calculation
dt = 1  # change in time
max_t = 60
D = 0.01  # Diffusion constant in terms of m^2/s
nts = np.around([nt for nt in np.linspace(1, max_t, nt)])
dts = {nt: diffuse_2D(nx, dx, dy, int(nt), D, dt) for nt in nts}

fig, axes = plt.subplots(2, 3, sharex=True, sharey=True)

for idx, d in enumerate(nts):
    axes[idx//3, idx % 3].imshow(dts[d], cmap='gray', vmin=0, vmax=1)
    axes[idx//3, idx % 3].set_axis_off()
    axes[idx//3, idx % 3].set_title('TS: {0}'.format(d))
    print(dts[d].sum())

plt.tight_layout()
plt.show(block=False)
