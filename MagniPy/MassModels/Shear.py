import numpy as np

def shear_def(x,y,shear,shear_theta):

    phi = np.arctan2(y, x)
    e1, e2 = shear * np.cos(2 * (phi - shear_theta * np.pi / 180)), shear * np.sin(
        2 * (phi - shear_theta * np.pi / 180))
    shearx = -e1 * x - e2 * y
    sheary = e2 * x - e1 * y

    return shearx,sheary
