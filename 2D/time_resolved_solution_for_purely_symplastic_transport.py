import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erf
import seaborn as sns
sns.set()

fig = plt.figure(0, figsize=(5, 5))
fig.clf()
fig, ax = plt.subplots(1, sharex=True, num=0)


def D_eff(D, q, cell_um):
    return (D*q*cell_um)/(D+q*cell_um)


delta = 0.1
beta = 2*delta
#D = D_eff(300, 10, 100)
D = 1

t = iter([100])


def f(k, T):
    return beta/(2*np.sqrt(D*delta)) * \
        (np.exp(-np.sqrt(delta/D)*abs(k)) *
         (1+erf(np.sqrt(delta*T) - (abs(k))/(2*np.sqrt(D*T)))) +
         np.exp(-np.sqrt(delta/D) * abs(k)) *
         (-1+erf(np.sqrt(delta*T) + (abs(k)/(2*np.sqrt(D*T))))))


nums = 50
for ti in t:
    vals = [f(i, ti) for i in range(0, nums)]
    ax.plot(np.arange(0, nums), vals, label='T: {0} D: 1'.format(ti))

D = 10
t = iter([100])
for ti in t:
    vals = [f(i, ti) for i in range(0, nums)]
    ax.plot(np.arange(0, nums), vals, label='T: {0} D: 10'.format(ti))

D = 100
t = iter([100])
for ti in t:
    vals = [f(i, ti) for i in range(0, nums)]
    ax.plot(np.arange(0, nums), vals, label='T: {0} D: 100'.format(ti))

ax.legend()
ax.set_ylabel('f(k,T)')
ax.set_xlabel(r'$\mu m$')
fig.canvas.draw()
# plt.show(block=False)
