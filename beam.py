"""
Contains data structures needed to represent a beam with given
loads etc.
"""
import numpy as np
from scipy.integrate import odeint
from scipy.optimize import root
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
        return self.val * max(min(self.b, b) - max(self.a, a), 0)

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
    def __init__(self, loads, reactions, l, E=1., Jz=1.):
        """
        loads : list of given loads (point loads, moments,...)
        reactions : list of unknown loads
        l : length of the beam (it starts at 0)
        profile : suitable representation of profile (TBD)
        """
        self.l = l
        self.loads = loads
        self.reactions = reactions
        self.E = E
        self.Jz = Jz
        return None

    def get_reactions(self):
        """
        Returns actual values of reactions.
        """
        A = np.array([[
            ri.M(rj.x + jj*.1, 0, self.l + .1*len(self.reactions))
            for ri in self.reactions] for jj, rj in enumerate(self.reactions)])
        b = np.array([
            sum([fi.M(rj.x + jj*.1, 0, self.l + .1*len(self.reactions))
                 for fi in self.loads])
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

    def displacement(self, x):
        """
        Returns vertical displacement at the point x.
        """
        def dv(y, x):
            """
            y[0] : vertical displacement
            y[1] : rotation
            Returns a numpy array with 1st and 2nd derivative of
            displacement.
            """
            return np.array([y[1], -self.moment(x) / self.E / self.Jz])

        def disp_ivp(x, x0=np.zeros(2), step=.1):
            """
            Returns displacement and rotation at point x with initial
            conditions x0.
            """
            ii = 1
            if x > 0: ii = 5
            return odeint(dv, x0, np.linspace(0, x, int(np.floor(x / step))+ii))[-1]

        def r(x0):
            """
            Returns residual vector at locations of reactions.
            """
            out = np.zeros(len(self.reactions))
            for ii, re in enumerate(self.reactions):
                jj = 0
                if type(re) == Moment: jj = 1
                out[ii] = disp_ivp(re.x, x0)[jj]
            return out

        out = disp_ivp(x, root(r, np.zeros(2)).x)
        return out

if __name__ == '__main__':
    l = 1.
    beam = Beam(
        [ConstantContinuousLoad(1., 0, .5*l)],
        [PointLoad(1., 0.), PointLoad(1., l)],
        l,
    )
    print('reactions:', beam.get_reactions())
    x = np.linspace(0, l, 51)

    # plot traction and bending moment
    plt.plot(x, [beam.traction(xi) for xi in x], '.-', label='traction')
    plt.plot(x, [beam.moment(xi) for xi in x], '.-', label='bending moment')
    plt.legend(loc='best')
    plt.grid()

    # plot deflection
    v = np.array([beam.displacement(xi) for xi in x]).T
    plt.figure()
    plt.plot(x, v[0], '.-', label='vertical displacement')
    plt.plot(x, v[1], '.-', label='rotation')
    plt.legend(loc='best')
    plt.grid()

    plt.show()
