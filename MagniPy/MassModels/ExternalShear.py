import numpy as np
from MagniPy.util import polar_to_cart

class Shear:

    def def_angle(self, x, y, shear, shear_theta):

        shear_theta += -90
        e1,e2 = polar_to_cart(shear,shear_theta)
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
