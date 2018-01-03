import numpy as np

class Shear:

    def def_angle(self,xgrid,ygrid,shear,shear_theta):

        shear_theta += -90

        phi = np.arctan2(ygrid, xgrid)
        e1, e2 = shear * np.cos(2 * (phi - shear_theta * np.pi / 180)), shear * np.sin(
            2 * (phi - shear_theta * np.pi / 180))
        shearx = -e1 * xgrid - e2 * ygrid
        sheary = e2 * xgrid - e1 * ygrid

        return shearx,sheary
