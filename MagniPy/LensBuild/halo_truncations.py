class Truncation:

    def __init__(self,truncation_routine=str,params=None):

        """

        :param truncation_routine: the type of truncation function applied for each halo
        Currently implemented:
        1) Isothermal virial 3d: truncates based on the tidal radius corresponding to a 3d position in an
        isothermal external mass distribution
        2) fixed_radius: truncates at a radius xi*r200
        """
        implemented = ['virial3d','fixed_radius']
        assert truncation_routine in implemented,'truncation routine not valid.'

        self.routine = truncation_routine
        self.params = params

    def virial3d(self,M):

        scrit = self.params['sigmacrit']
        Rmain = self.params['Rein']
        r3d = self.params['r3d']

        return (0.5*M*r3d**2*(scrit*Rmain)**-1)**(1*3**-1)

    def fixed_radius(self, fixed_radius):

        return fixed_radius