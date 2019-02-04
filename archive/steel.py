import numpy  # loading our favorite library
from matplotlib import pyplot  # and the useful plotting library
from numpy import pi


def diff(x): return ((1.38e-23 * 298.15)/(6*pi * 8.9e-4 * x))


nx = 5
dx = 0.1
nt = 60*60  # the number of timesteps we want to calculate
D = diff(3.65e-10) * 1e+6
dt = dx**2 / (2 * D * dx**2)


u = numpy.ones(int(nx/dx))  # a numpy array with nx elements all equal to 1.
u[20:30] = 2  # setting u = 2 between 0.5 and 1 as per our I.C.s

# our placeholder array, un, to advance the solution in time
un = numpy.ones(nx)

for n in range(nt):  # iterate through time
    un = u.copy()  # copy the existing values of u into un
    for i in range(1, 49):
        u[i] = un[i] + D * (un[i+1] - 2 * un[i] + un[i-1])/dx**2

pyplot.plot(u)
