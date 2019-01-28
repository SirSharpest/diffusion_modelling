from pd_diffuse import make_cell_states as state_generator
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LogNorm
import seaborn as sns
sns.set()


def animate(i):
    axes[0].cla()
    axes[1].cla()
    axes[0].set_facecolor((0, 0, 0))
    axes[1].set_facecolor((0, 0, 0))

    axes[0].pcolormesh(np.expand_dims(states50[i].ravel(), axis=0),
                       norm=LogNorm(vmin=1e-100, vmax=1))

    axes[1].pcolormesh(np.expand_dims(states100[i].ravel(), axis=0),
                       norm=LogNorm(vmin=1e-100, vmax=1))
    fig.suptitle('FN={0}'.format(i))
    axes[0].set_title('PD at 50%')
    axes[1].set_title('PD at 100%')


fig, axes = plt.subplots(2, figsize=(15, 8), sharex=True, sharey=True)

states100 = np.array([i for i in state_generator(q=1, t=60*60*24)])
states50 = np.array([i for i in state_generator(q=0.5, t=60*60*24)])

t = axes[1].pcolormesh(np.expand_dims(states100[0].ravel(), axis=0),
                       norm=LogNorm(vmin=1e-10, vmax=1e-2))

t = axes[0].pcolormesh(np.expand_dims(states50[0].ravel(), axis=0),
                       norm=LogNorm(vmin=1e-10, vmax=1e-2))
# fig.colorbar(t)

anim = animation.FuncAnimation(
    fig, animate, frames=range(1, states100.shape[0], 50),
    blit=False, interval=200)

# fig.tight_layout()
anim.save('pd_100vs50.mp4')
# plt.show()
