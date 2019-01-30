import imageio
from pd_diffuse import make_cell_states as state_generator
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LogNorm
import seaborn as sns
sns.set()


def make_animation():

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
    # anim.save('pd_100vs50.mp4')
    plt.show()


def make_compare_4by4(ts):
    fig, axes = plt.subplots(2, 2, figsize=(18, 8), sharex=True, sharey=True)

    def make_states(x, y): return np.array(
        [i for i in state_generator(q=x, t=60*60*y)])

    t = max(ts)
    states = {"states25": make_states(.25, t),
              "states50": make_states(.50, t),
              "states75": make_states(.75, t),
              "states100": make_states(1, t)}

    for t in ts:

        t1 = axes[0, 0].pcolormesh(
            np.expand_dims(states['states25'][t*60*60].ravel(),
                           axis=0),
            norm=LogNorm(vmin=1e-12, vmax=1e-2))
        axes[0, 0].set_title('PD: 25')
        t2 = axes[0, 1].pcolormesh(
            np.expand_dims(states['states50'][t*60*60].ravel(),
                           axis=0),
            norm=LogNorm(vmin=1e-12, vmax=1e-2))
        axes[0, 1].set_title('PD: 50')
        t3 = axes[1, 0].pcolormesh(
            np.expand_dims(states['states75'][t*60*60].ravel(),
                           axis=0),
            norm=LogNorm(vmin=1e-12, vmax=1e-2))
        axes[1, 0].set_title('PD: 75')
        t4 = axes[1, 1].pcolormesh(
            np.expand_dims(states['states100'][t*60*60].ravel(),
                           axis=0),
            norm=LogNorm(vmin=1e-12, vmax=1e-2))
        axes[1, 1].set_title('PD: 100')

        for i in range(4):
            x = i//2
            y = i % 2
            axes[x, y].axvline()

        # labels thingy
        N = 11
        Xs = 100
        x_labels_locations = np.linspace(0, N*Xs, num=11)
        x_labels_locations = [50+(i*100) for i in range(11)]
        x_labels = ['C{0}'.format(n) for n in range(11)]

        for i in range(4):
            x = i//2
            y = i % 2

            for j in x_labels_locations:
                axes[x, y].axvline(j+50, color='blue')

        plt.sca(axes[0, 0])
        plt.xticks(x_labels_locations, x_labels)

        # cbar = fig.colorbar(t4, ticks=[1, 0.5, 0],
        #                     ax=axes)
        # cbar.ax.set_yticklabels(['Low', 'Medium', 'High'])

        fig.tight_layout()
        fig.suptitle('Diffusion at {0}hr'.format(t))
        fig.savefig('diffusion{0}hr.png'.format(t))
        # plt.show()


def make_gif():
    files = ['./media/diffusion{0}hr.png'.format(i) for i in range(1, 24)]
    imageio.mimsave('./media/timelapse.gif',
                    [imageio.imread(f) for f in files])
