from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver
from lenstronomy.LensModel.lens_model import LensModel

class MockLensBase(object):

    def __init__(self, source_x, source_y, kwargs):

        self.lensModel = LensModel(['SPEMD', 'SHEAR'])
        self.solver = LensEquationSolver(self.lensModel)
        self.x_image, self.y_image = self.solver.findBrightImage(source_x, source_y, kwargs)