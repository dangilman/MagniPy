from MagniPy.Workflow.radio_lenses.lens1115 import Lens1115
from MagniPy.util import approx_theta_E
from MagniPy.Workflow.mock_lenses.analog_base import MockLensBase

class Lens1115AnalogFold(MockLensBase):

    reference_lens = Lens1115()

    def __init__(self):

        theta_E = approx_theta_E(self.reference_lens.x, self.reference_lens.y)
        kwargs_reference = [{'theta_E': theta_E, 'center_x': 0., 'center_y': 0., 'e1': 0.1, 'e2': -0.2, 'gamma': 2.},
                            {'gamma1': 0.05, 'gamma2': 0.}]
        source_x, source_y = 0.04, -0.12

        super(Lens1115AnalogFold, self).__init__(source_x, source_y, kwargs_reference)

class Lens1115AnalogCross(MockLensBase):

    reference_lens = Lens1115()

    def __init__(self):

        theta_E = approx_theta_E(self.reference_lens.x, self.reference_lens.y)
        kwargs_reference = [{'theta_E': theta_E, 'center_x': 0., 'center_y': 0., 'e1': -0.1, 'e2': 0.2, 'gamma': 2.},
                            {'gamma1': -0.05, 'gamma2': 0.02}]
        source_x, source_y = 0.04, -0.02

        super(Lens1115AnalogCross, self).__init__(source_x, source_y, kwargs_reference)
        #plt.scatter(self.x_image, self.y_image); plt.show()

class Lens1115AnalogCusp(MockLensBase):

    reference_lens = Lens1115()

    def __init__(self):

        theta_E = approx_theta_E(self.reference_lens.x, self.reference_lens.y)
        kwargs_reference = [{'theta_E': theta_E, 'center_x': 0., 'center_y': 0., 'e1': -0.1, 'e2': 0., 'gamma': 2.},
                            {'gamma1': 0.03, 'gamma2': 0.02}]
        source_x, source_y = 0.09, -0.01

        super(Lens1115AnalogCusp, self).__init__(source_x, source_y, kwargs_reference)
        #plt.scatter(self.x_image, self.y_image); plt.show()
