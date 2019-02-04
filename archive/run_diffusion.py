from diffusion import diffuse_1D_discrete_solution, diffuse_1D_exact_solution
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
plt.close('all')


def um_to_m(x): return x*1e-6


def um2_to_m2(x): return x*1e-12


def diff(x): return ((1.38e-23 * 298.15)/(6*np.pi * 8.9e-4 * x))


def nt_to_sec(x, dt): return x*(dt*(1/dt))


# Measurements in mm
sx = 5
dx = 0.1
D = diff(3.5e-10) * 1e+6
dt = 1  # dx2/(2*D*dx2)
nx = int(sx/dx)

nts = [1, 60, 60*30, 60*60]

# Calc initial state
u = np.zeros(nx)
production = 1
mid_s = int(nx/2-5)
mid_e = int(nx/2+5)

mid_s = int(5)
mid_e = int(10)

u[mid_s:mid_e] = production
states = [u.copy()]

c_u = np.zeros(nx)
c_u[mid_s:mid_e] = 1
c_states = [c_u.copy()]

end_state = diffuse_1D_discrete_solution(dx, nts[3], D, dt, states[0], 0)

for i, nt in enumerate(nts):
    states.append(diffuse_1D_discrete_solution(dx, nt, D, dt, states[-1], 0))


def plot_states(states, nts):
    fig, axes = plt.subplots(len(nts), 1, sharex=True)
    for i in range(len(nts)):
        axes[i].plot(states[i])
        axes[i].set_title('Timesteps (seconds): {0}'.format(nts[i]))
    plt.tight_layout()
    plt.show(block=False)


plot_states(states, nts)
