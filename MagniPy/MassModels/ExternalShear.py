import numpy as np
from MagniPy.util import polar_to_cart

class Shear:

    def def_angle(self, x, y, shear, shear_theta):

        shear_theta += -90

        phi = np.arctan2(y, x)
        e1, e2 = shear * np.cos(2 * (phi - shear_theta * np.pi / 180)), shear * np.sin(
            2 * (phi - shear_theta * np.pi / 180))

        shearx = -e1 * x - e2 * y
        sheary = e2 * x - e1 * y

        return shearx,sheary

    def params(self,shear,shear_theta):

        subkwargs = {}
        subkwargs['shear'] = shear
        subkwargs['shear_theta'] = shear_theta

        lenstronomy_args = {}
        lenstronomy_args['shear'] = shear
        lenstronomy_args['shear_theta'] = shear_theta

        return subkwargs,lenstronomy_args
