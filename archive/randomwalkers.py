import matplotlib.pyplot as plt
from random import randint
# Define sizes for random walk radius
CONST_MIN = -100
CONST_MAX = 100


class Walkers:

    def __init__(self, N):
        self.centre = [0, 0]
        self.walkers = [Walker(self.centre[0],
                               self.centre[1], n) for n in range(N)]

    def make_colors(self, N):
        return [['r', 'g', 'b'][n % 3] for n in range(N)]

    def get_xs(self):
        return [self.walkers[w].x for w, _ in enumerate(self.walkers)]

    def get_ys(self):
        return [self.walkers[w].y for w, _ in enumerate(self.walkers)]

    def get_positions(self):
        return list(zip(self.get_xs(), self.get_ys()))

    def get_colors(self):
        return [self.walkers[w].c for w, _ in enumerate(self.walkers)]

    def plot_walkers(self, ax):
        ax.scatter(*zip(*self.get_positions()), c=self.get_colors(), s=40)
        ax.set_xlim(CONST_MIN, CONST_MAX)
        ax.set_ylim(CONST_MIN, CONST_MAX)

    def do_random_walk(self, N):
        for n in range(N):
            for w, _ in enumerate(self.walkers):
                self.walkers[w].random_move()


class Walker:
    def __init__(self, inital_x, inital_y, identifier, color='k'):
        self.identifier = identifier
        self.x = 0
        self.y = 0
        self.c = 'k'

    def random_move(self):
        new_x = self.x
        new_y = self.y

        r = randint(0, 4)
        if r == 0:
            pass
        elif r == 1:
            new_y = self.y+1
        elif r == 2:
            new_y = self.y-1
        elif r == 3:
            new_x = self.x-1
        elif r == 4:
            new_x = self.x+1

        if self.check_valid_move(new_x, new_y):
            self.x = new_x
            self.y = new_y

    def check_valid_move(self, x, y):
        if x < CONST_MIN or x > CONST_MAX:
            return False
        if y < CONST_MIN or y > CONST_MAX:
            return False
        return True


walkers = Walkers(30)
fig, ax = plt.subplots(1, 3, figsize=(6, 2), sharex=True, sharey=True)


for idx, i in enumerate([1, 100, 3000]):
    walkers.do_random_walk(i)
    walkers.plot_walkers(ax[idx])


plt.tight_layout()
plt.show()
