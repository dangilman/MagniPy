from lenstronomy.LensModel.Profiles.spemd import SPEMD
from lenstronomy.LensModel.Profiles.external_shear import ExternalShear

class SIE_shear(object):
    """
    class for singular isothermal ellipsoid (SIS with ellipticity)
    """
    def __init__(self):
        self.spemd = SPEMD()
        self.shear = ExternalShear()

    def function(self, x, y, theta_E, q, phi_G, shearx, sheary, shear_e1, shear_e2, center_x=0, center_y=0):
        """

        :param x:
        :param y:
        :param theta_E:
        :param q:
        :param phi_G:
        :param center_x:
        :param center_y:
        :return:
        """
        gamma = 2

        return self.spemd.function(x, y, theta_E, gamma, q, phi_G, center_x, center_y) + \
               self.shear.function(shearx,sheary,shear_e1,shear_e2)

    def derivatives(self, x, y, theta_E, q, phi_G, shearx, sheary, shear_e1, shear_e2, center_x=0, center_y=0):
        """

        :param x:
        :param y:
        :param theta_E:
        :param q:
        :param phi_G:
        :param center_x:
        :param center_y:
        :return:
        """
        gamma = 2
        return self.spemd.derivatives(x, y, theta_E, gamma, q, phi_G, center_x, center_y) + \
               self.shear.derivatives(shearx, sheary, shear_e1, shear_e2)

    def hessian(self, x, y, theta_E, q, phi_G, shearx, sheary, shear_e1, shear_e2, center_x=0, center_y=0):
        """

        :param x:
        :param y:
        :param theta_E:
        :param q:
        :param phi_G:
        :param center_x:
        :param center_y:
        :return:
        """
        gamma = 2
        return self.spemd.hessian(x, y, theta_E, gamma, q, phi_G, center_x, center_y) + \
            self.shear.hessian(shearx, sheary, shear_e1, shear_e2)