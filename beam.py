"""
Contains data structures needed to represent a beam with given
loads etc.
"""
import numpy as np
import matplotlib.pyplot as plt

class Load:
    """
    Base class for PointLoad and Moment.
    """
    def __init__(self, val, x):
        """
        val : value of the force/moment
        x : its position along the beam
        """
        self.val = val
        self.x = x
        return None

class PointLoad(Load):
    """
    Point load with known components.
    """
    def F(self, a, b):
        """
        Resulting force on the interval [a, b].
        """
        if (self.x <= max(a, b)) and (self.x >= min(a, b)):
            return self.val
        else:
            return 0.

    def M(self, x, a, b):
        """
        Returns the resulting moment if the force lies within the
        interval [a, b], w.r.t. point x.
        """
        if (self.x <= max(a, b)) and (self.x >= min(a, b)):
            return self.val * (self.x - x)
        else:
            return 0.

class Moment(Load):
    """
    Discrete moment of value val acting at point x.
    """
    def F(self, a, b):
        """
        Resulting force - always zero.
        """
        return 0.

    def M(self, x, a, b):
        """
        Returns the moment value if it lies within the interval
        [a, b].
        """
        if (self.x <= max(a, b)) and (self.x >= min(a, b)):
            return self.val
        else:
            return 0.

class ConstantContinuousLoad:
    """
    Constant continuous load of value val [N/m] on the interval
    [a, b].
    """
    def __init__(self, val, a, b):
        self.val = val
        self.a, self.b = min(a, b), max(a, b)
        return None

    def F(self, a, b):
        """
        Return the resultant force [N] of the part of the load that
        lies within the interval [a, b].
        """
        return self.val * (min(self.b, b) - max(self.a, a))

    def M(self, x, a, b):
        """
        Returns the resulting moment [N*m] of the part of the load
        that lies within the interval [a, b], w.r.t. point x.
        """
        return self.F(a, b) * (.5*(max(self.a, a) + min(self.b, b)) - x)

class Beam:
    """
    Represents a beam with loads, reactions, profile etc.
    Provides computation of reactions and evaluation of traction 
    and bending moment along the beam.
    """
    def __init__(self, loads, reactions, l, profile=None):
        """
        loads : list of given loads (point loads, moments,...)
        reactions : list of unknown loads
        l : length of the beam (it starts at 0)
        profile : suitable representation of profile (TBD)
        """
        self.l = l
        self.loads = loads
        self.reactions = reactions
        self.profile = profile
        return None

    def get_reactions(self):
        """
        Returns actual values of reactions.
        """
        A = np.array([[
            ri.M(rj.x + jj*.1, 0, self.l)
            for ri in self.reactions] for jj, rj in enumerate(self.reactions)])
        b = np.array([
            sum([fi.M(rj.x + jj*.1, 0, self.l) for fi in self.loads])
            for jj, rj in enumerate(self.reactions)])
        return np.linalg.solve(A, b)

    def traction(self, x):
        """
        Returns traction at x.
        """
        return (
            -sum([fi.F(0, x) for fi in self.loads])
            +sum([ri * self.reactions[ii].F(0, x)
                  for ii, ri in enumerate(self.get_reactions())])
        )

    def moment(self, x):
        """
        Returns bending moment at x.
        """
        return (
            +sum([fi.M(x, 0, x) for fi in self.loads])
            -sum([rv * ri.M(x, 0, x)
                  for ri, rv in zip(self.reactions, self.get_reactions())])
        )

if __name__ == '__main__':
    beam = Beam(
        [ConstantContinuousLoad(1., 0., .5)],
        [PointLoad(1., 0.), PointLoad(1., 1.)],
        1.
    )
    print('reactions:', beam.get_reactions())
    x = np.linspace(0, 1, 51)
    plt.plot(x, [beam.traction(xi) for xi in x], '.-', label='traction')
    plt.plot(x, [beam.moment(xi) for xi in x], '.-', label='bending moment')
    plt.legend(loc='best')
    plt.grid()
    plt.show()
