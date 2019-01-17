from diffusion import diffuse_1D_discrete_solution, diffuse_1D_exact_solution
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


def diff(x): return ((1.38e-23 * 298.15)/(6*np.pi * 8.9e-4 * x))


def random_walk(s, n):
    """
    Calculates random walk in 2D
    """
    steps = np.random.choice([-1, 1, 0], (s, 2, n))
    return np.cumsum(steps, axis=0)


likelihood_of_binding = 1
max_x_offset = 10000
max_y_offset = 100
distance = 100  # given in mm
steps = 100
num_of_chitin_molecules = int(1e4)
D = diff(3.5e-10) * 1e+6  # mm per second ^2


walks = random_walk(200000, 10)
walks = walks.astype(float)
b = walks.copy()
for i in range(walks.shape[-1]):

    y_breach = np.argmax(abs(walks[1:, 1, i]) >= max_y_offset)
    x_breach = np.argmax(abs(walks[1:, 0, i]) >= max_x_offset)

    if y_breach == 0 or x_breach == 0:
        cut_point = max(y_breach, x_breach)
    else:
        cut_point = min(y_breach, x_breach)

    if cut_point > 0:
        walks[cut_point:, 0, i] = np.nan
        walks[cut_point:, 1, i] = np.nan

    plt.plot(walks[:, 0, i], walks[:, 1, i])
plt.savefig('random_binding.png')
plt.close('all')
