======
pyBeam
======

Python module for structural analysis of statically determinate thin beams.

Available types of supports:

- simple support (deflection fixed),
- only angle of deflection fixed,
- encastre (both deflection and angle of deflection fixed),

Available types of loads:

- force [N]
- moment [N.m]
- constant continuous load [N/m]

The Beam class provides methods for calculating:

- reactions,
- bending force and bending moment along the beam,
- deflection and angle of deflection.

Requirements
------------

For importing beam into other scripts: numpy, scipy

For running the example: matplotlib

The software should work under both python versions 2 and 3, but has not been tested systematically against different versions of its requirements.
