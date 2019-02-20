import numpy as np
import matplotlib.pyplot as plt


def C(x, t, D): return (1/np.sqrt(4*np.pi*D*t)
                        * np.exp(((-np.square(x))/(4*D*t))))


def stokes_einstein(x): return ((1.38e-23 * 298.15)/(6*np.pi * 8.9e-4 * x))


D = stokes_einstein(3.5e-10)
X = np.linspace(-0.0001, 0.0001, 100)
t = 1

c = [C(x, t, D) for x in X]


fig, ax = plt.subplots(1)


ax.plot(c)
ax.set_title('TS: {0}'.format('1'))
plt.tight_layout()
plt.show()
