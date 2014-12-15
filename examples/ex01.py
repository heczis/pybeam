"""
Example usage of functions provided by beam module.

Simply supported beam of total length 1.2m with rectangular
cross-section.

The loads are:
* Constant continuous load q = 50kN/m between coordinates
  a = 0.5m and a+b = 0.8m.
* Two point loads F = -20kN (i.e. in the upwards direction)
  at a = 0.5m and a+b = 0.8m.

Dimensions of the cross-section (height H and width B) are to be determined so that maximum stress is 120MPa. H/B = 2 holds.

Deflection and angle of deflection are to be determined at point
C at the location a+b=0.8m.
"""
import numpy as np
import matplotlib.pyplot as plt
from beam import Beam, ConstantContinuousLoad, PointLoad

# known quantities
q = 50e3
F = 20e3
a, b, c = .5, .3, .4
C = a + b
E = 2e11
sigD = 120e6

### computation of reactions: ###

# loads - list of known loads
loads = [
    ConstantContinuousLoad(q, a, a+b),
    PointLoad(-F, a),
    PointLoad(-F, a+b)
]
# reactions - list of unknown loads
#  All must have magnitude = 1, otherwise the results of
#  Beam.get_reactions would be wrong.
reactions = [PointLoad(1, 0), PointLoad(1, a+b+c)]

# Initialize the beam object.
# For now we don't need the actual value of Jz since we are
# going to compute the actual dimensions based on the values
# of reaction forces.
beam = Beam(loads, reactions,
            l = a+b+c,
            E = E,
            Jz = 1.,
)

Rs = beam.get_reactions()
print('reactions:\n R_A = {:.2e}N\n R_B = {:.2e}N'.format(Rs[0], Rs[1]))

### computation and plots of bending force and moment ###
n = 12*5+1
x = np.linspace(0, a+b+c, n)

plt.plot(x, [beam.force(xi) for xi in x], '.-', label='force')
Mo = np.array([beam.moment(xi) for xi in x])
plt.plot(x, Mo, '.-', label='moment')

plt.legend(loc='best')
plt.grid()

### actual dimensions of the cross-section ###

# sigma = Mmax/Wo = Mmax/(Jz/(H/2)) <= sigD
# Jz >= H/2*Mmax/sigD
#   [obdelnikovy prurez]
#   Jz = 1/12*B*H**3 = (H/B=2) =
#      = 1/24*H**4
# H**4 >= 24*H/2*Mmax/sigD
# H >= (12*Mmax/sigD)**(1/3)

Mmax = max(abs(Mo))
print('Mmax = {:.4}N'.format(Mmax))

H = (12*Mmax/sigD)**(1./3)
print('H = {:.4}m'.format(H))
print('B = {:.4}m'.format(.5*H))

Jz = H**4/24
beam.Jz = Jz

### computation and plots of deflection and angle ###

v = np.array([beam.deflection(xi) for xi in x]).T
v_C = beam.deflection(C)
print('v_C = {:.4}m\nphi_C = {:.4}rad'.format(v_C[0], v_C[1]))

plt.figure()
plt.plot(x, v[0], '.-', label='deflection')
plt.plot(x, v[1], '.-', label='angle')
plt.grid()
plt.legend(loc='best')
plt.show()
