class MassSheet:

    def params(self,kappa_ext=None,mass=None,area=None,sigma_crit=None):

        subkwargs = {}
        otherkwargs = {}

        if mass is None or area is None or sigma_crit is None:
            assert kappa_ext is not None
            subkwargs['kappa_ext'] = kappa_ext
        else:
            subkwargs['kappa_ext'] = mass*area**-1*sigma_crit**-1

        otherkwargs['name'] = 'CONVERGENCE'

        return subkwargs,otherkwargs

    def def_angle(self,x,y,kappa_ext):

        """
        deflection angle

        :param x: x-coordinate
        :param y: y-coordinate
        :param kappa_ext: external convergence
        :return: deflection angles (first order derivatives)
        """

        f_x = kappa_ext * x
        f_y = kappa_ext * y
        return f_x, f_y