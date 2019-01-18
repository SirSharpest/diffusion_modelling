import matplotlib.patches as mpatches
from diffusion import diffuse_1D_discrete_solution, diffuse_1D_exact_solution
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
plt.close('all')
"""
Idea: include velocity: https://stackoverflow.com/a/38066090
"""

PLOT = False


def diff(x): return ((1.38e-23 * 298.15)/(6*np.pi * 8.9e-4 * x))


def random_walk(s, n, dx):
    """
    Calculates random walk in 2D
    """
    steps = np.random.choice([-dx, dx], (s, 2, n))
    # change start position to be randomised by +- 1mm max
    steps[0, 0, :] = np.random.random((1, n)) - np.random.random((1, n))
    steps[0, 1, :] = np.random.random((1, n)) - np.random.random((1, n))
    steps[0, 1, :] = steps[0, 1, :]/10
    return np.cumsum(steps, axis=0)


likelihood_of_binding = 0.5
distance = 2  # given in mm
max_x_offset = distance
max_y_offset = 0.5
num_of_chitin_molecules = int(1e4)
D = diff(3.5e-10) * 1e+6  # mm per second ^2
delta_x = 0.01  # currently assumes no velocity
cell_size = 0.1


def make_walks():

    walks = random_walk(60*60, num_of_chitin_molecules, delta_x)
    walks = walks.astype(float)

    plt.figure(figsize=(20, 4))
    plt.grid(False)
    y_breaches = []
    for i in range(walks.shape[-1]):

        y_breach = np.argmax(abs(walks[1:, 1, i]) >= max_y_offset)
        x_breach = np.argmax(abs(walks[1:, 0, i]) >= max_x_offset)

        # give y a chance to not bind
        if y_breach != 0:
            while(np.random.rand() > likelihood_of_binding
                  and abs(walks[y_breach, 1, i]) <= max_y_offset*1.1):
                tmp_y_breach = np.argmax(
                    abs(walks[y_breach+1:, 1, i]) >= max_y_offset)
                if tmp_y_breach == 0:
                    break
                else:
                    y_breach = tmp_y_breach

        if y_breach == 0 or x_breach == 0:
            cut_point = max(y_breach, x_breach)
        else:
            cut_point = min(y_breach, x_breach)

        if cut_point > 0:
            walks[cut_point:, 0, i] = np.nan
            walks[cut_point:, 1, i] = np.nan

        y_breaches.append(walks[y_breach, 1, i])
        # PLOTTING BELOW
        if PLOT:
            plt.scatter(walks[cut_point-1, 0, i],
                        walks[cut_point-1, 1, i],
                        color='b' if cut_point == 0 else 'r', s=10)

            plt.xlim(-max_x_offset, max_x_offset)
            plt.ylim(-max_y_offset*1.1, max_y_offset*1.1)

    if PLOT:
        x_label_loc = np.linspace(-distance,
                                  distance,
                                  num=int(distance/cell_size)*2+1)

        _ = [plt.axvline(x)for x in x_label_loc]
        plt.axvline(-1, c='r')
        plt.axvline(1, c='r')

        x_labels = [r"i{0}".format(int(x)) for x in np.arange(-distance/cell_size,
                                                              distance/cell_size+1)]

        plt.xticks(x_label_loc - cell_size/2, x_labels)

        red_patch = mpatches.Patch(color='red', label='Bound elcitors ')
        blue_patch = mpatches.Patch(color='blue', label='Unbound elicitors')
        plt.legend(handles=[red_patch, blue_patch])
        plt.suptitle('Random Walk diffusion to estimate binding')
        plt.tight_layout()
        plt.savefig('random_binding.png')
    return walks


# START COUNTING HERE
# use np arrays, just resize at the end and only plot start:nan
simulation_times = 200
end_times = np.zeros(num_of_chitin_molecules*simulation_times)
travel_distance = np.zeros(num_of_chitin_molecules*simulation_times)

for i in range(simulation_times):
    print('Generation #{0}'.format(i))
    walks = make_walks()
    for i in range(walks.shape[-1]):
        tmp = np.where(np.isnan(walks[:, :, i]))[0]
        dist = 0
        if tmp.size == 0:
            dist = np.nan
        else:
            end_times[i] = tmp[0]
            dist = walks[tmp[0]-1, 0, i]
            travel_distance[i] = dist


plt.hist(end_times)
plt.title('Distribution for time of molecules to bind to cell membrane; N={0}'.format(
    len(end_times)))
plt.savefig('binding_time.png')
plt.cla()
plt.hist(travel_distance)
plt.title('Distribution for distance of binding; N={0}'.format(len(end_times)))
plt.savefig('binding_distance.png')
