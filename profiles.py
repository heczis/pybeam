"""
Characteristics of different geometrical profiles.
"""
def jz_rectangle(b, h, y=0, z=0):
    """
    Quadratic moment of a rectangular profile.
    b : dimension in the z direction
    h : dimension in the y direction
    y : offset of the z axis in y direction, w.r.t. the center of
        the profile
    z : offset of the y axis in z direction, w.r.t. the center of
        the profile
    """
    return b*h**3 / 12. + b*h*(y**2 + z**2)
