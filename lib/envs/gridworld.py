import numpy as np
import sys
import StringIO

from gym import utils
from gym.envs.toy_text import discrete

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3


class GridworldEnv(discrete.DiscreteEnv):
    """
    Grid World environment from Sutton & Barto (2016).

    S is the agent's starting position and G is the goal state.

    The agent can take actions in each direction
    (UP=0, RIGHT=1, DOWN=2, LEFT=3).
    Actions going off the edge leave the agent in its current state.
    Each non-terminating step, the agent receives a random reward of -12 or +10
    with equal probability.
    In the goal state every action yields +5 and ends an episode.
    """

    metadata = {'render.modes': ['human', 'ansi', 'image']}

    def __init__(self, map):
        self.desc = desc = np.asarray(map, dtype='c')
        self.nrow, self.ncol = nrow, ncol = desc.shape

        nA = 4
        nS = nrow * ncol

        isd = np.array(desc == b'S').astype('float64').ravel()
        isd /= isd.sum()

        P = {s: {a: [] for a in range(nA)} for s in range(nS)}

        def to_s(row, col):
            return row * ncol + col

        def inc(row, col, a):
            newrow, newcol = row, col
            if a == 0:  # left
                newcol = max(col - 1, 0)
            elif a == 1:  # down
                newrow = min(row + 1, nrow - 1)
            elif a == 2:  # right
                newcol = min(col + 1, ncol - 1)
            elif a == 3:  # up
                newrow = max(row - 1, 0)
            newletter = desc[newrow, newcol]
            if newletter == b'W':
                return (row, col)
            else:
                return (newrow, newcol)

        # Fill the transitions.
        for row in range(nrow):
            for col in range(ncol):
                s = to_s(row, col)
                for a in range(4):
                    li = P[s][a]
                    letter = desc[row, col]
                    if letter == b'G':
                        li.append((1.0, s, 0, True))
                    else:
                        newrow, newcol = inc(row, col, a)
                        newstate = to_s(newrow, newcol)
                        newletter = desc[newrow, newcol]
                        done = newletter == b'G'
                        rew = float(newletter == b'G')
                        li.append((1.0, newstate, rew, done))

        # We expose the model of the environment for educational purposes
        # This should not be used in any model-free learning algorithm
        self.P = P

        super(GridworldEnv, self).__init__(nS, nA, P, isd)

    def get_image(self):
        image = np.ones(self.desc.shape + (3,))
        image[self.desc == b'W'] = (0, 0, 0)
        image[self.desc == b'S'] = (0.5, 0.5, 0.5)
        image[self.desc == b'G'] = (0.0, 0.8, 0.0)
        row, col = self.s // self.ncol, self.s % self.ncol
        image[row, col] = (0.8, 0.0, 0.0)
        return image

    def _render(self, mode='human', close=False):
        if close:
            return

        outfile = StringIO() if mode == 'ansi' else sys.stdout

        row, col = self.s // self.ncol, self.s % self.ncol
        desc = self.desc.tolist()
        desc = [[c.decode('utf-8') for c in line] for line in desc]
        desc[row][col] = utils.colorize(desc[row][col], "red", highlight=True)
        if self.lastaction is not None:
            outfile.write("  ({})\n".format(["Left","Down","Right","Up"][self.lastaction]))
        else:
            outfile.write("\n")
        outfile.write("\n".join(''.join(line) for line in desc)+"\n")
