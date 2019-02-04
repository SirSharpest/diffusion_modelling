import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from helper_funcs import diff, diffuse_1D, diffuse_1D_better
from numpy import square
import matplotlib.gridspec as gridspec
from scipy.signal import medfilt


# Measurements in mm
sx = 5
dx = 0.1
dx2 = square(dx)
D = diff(3.5e-10) * 1e+6
dt = 1  # dx2/(2*D*dx2)
nx = int(sx/dx)

nt = (60*60*4)
# nts = [str(n) for n in np.linspace(0, nt, 6, dtype=int)]  # 12 hrs
nts = ['0', '60', '300', '3600', str(60*60*2), str(60*60*4), str(60*60*12)]

# Calc initial state
u = np.zeros(nx)
production = 0
mid_s = int(nx/2-5)
mid_e = int(nx/2+5)

mid_s = int(5)
mid_e = int(10)

u[mid_s:mid_e] = 0
states = [u.copy()]

c_u = np.zeros(nx)
c_u[mid_s:mid_e] = 1
c_states = [c_u.copy()]


burst = True
for idx, n in enumerate(nts):
    if idx is 0:
        continue
    u = states[-1].copy()
    u = diffuse_1D_better(int(nts[idx]), u, D, dt,
                          dx2, mid_s, mid_e, production, burst)
    states.append(u.copy())


for idx, n in enumerate(nts):
    if idx is 0:
        continue
    u = c_states[-1].copy()
    u = diffuse_1D_better(int(nts[idx]), u, D, dt,
                          dx2, mid_s, mid_e, production, False)
    c_states.append(u.copy())


fig = plt.figure()
axes = []
gs0 = gridspec.GridSpec(len(nts), 1)


for i in range(0, len(nts)):
    gs1 = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=gs0[i], hspace=0.05, wspace=0.05)
    axes.append(plt.subplot(gs1[0]))
    axes.append(plt.subplot(gs1[1]))


x = np.linspace(0, sx, nx)
y = states[-1]
extent = [x[0]-(x[1]-x[0])/2., x[-1]+(x[1]-x[0])/2., 0, 1]

vmax = 0
for n in states:
    if n.max() > vmax:
        vmax = n.max()

num_colors = 255

for idx, _ in enumerate(states):
    states[idx][states[idx] < (1e-13)] = 0
bounds = np.array(states).ravel()
bounds = np.unique(bounds)
bounds.sort()
norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)

frame = 0
heat = True
if heat is True:
    for idx, d in enumerate(nts):
        #states[idx] = medfilt(states[idx], 9)
        frame = axes[idx*2].pcolormesh(states[idx][np.newaxis, :],
                                       cmap="RdBu_r", norm=norm)
        axes[(idx*2)+1].pcolormesh(c_states[idx][np.newaxis, :],
                                   cmap="RdBu_r", norm=norm)
        title_time = "{0}hrs : {1}minutes".format(
            int(d)//(60*60), (int(d)//60) % 60)
        axes[idx*2].set_title(title_time)

        for v in range(0, 50):
            axes[idx*2].axvline((v), color='k', linestyle='solid')
            axes[(idx*2)+1].axvline((v), color='k', linestyle='solid')
            # print(v)
        axes[idx*2].set_xlim(5, 45)
        axes[(idx*2)+1].set_xlim(5, 45)

    fig.colorbar(frame, ax=axes, format='%.0e')
    # plt.tight_layout()
    plt.show()
else:
    for idx, d in enumerate(nts):
        axes[idx*2].plot(states[idx])
        axes[idx*2+1].plot(c_states[idx])
        title_time = "{0}hrs : {1}minutes".format(
            int(d)//(60*60), (int(d)//60) % 60)
        # axes[idx].set_title(title_time)
        axes[idx*2].set_ylim(0, 1)
        # for v in range(0, sx, 100):
        #     axes[idx//3, idx % 3].axvline(v, color='k', linestyle='solid')
        # axes[idx//3, idx % 3].set_xlim(0, sx)

    # plt.tight_layout()
    plt.show()
