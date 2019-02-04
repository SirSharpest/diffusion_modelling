import numpy as np
import matplotlib.pyplot as plt
from helper_funcs import diff


def C(x, t, D): return (1/np.sqrt(4*np.pi*D*t)
                        * np.exp(- ((np.square(x))/(4*D*t))))


D = diff(3.5e-10) * 1e+6
X = np.linspace(-5, 5, 10)
t = 10

c = [C(x, t, D) for x in X]


fig, ax = plt.subplots(1)


ax.plot(c)
ax.set_title('TS: {0}'.format('1'))
plt.tight_layout()
plt.show()
