from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LensModel.lens_model_extensions import LensModelExtensions
from MagniPy.util import polar_to_cart

# import the lens equation solver class (finding image plane positions of a source position)
from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver
# import lens model solver with 4 image positions constrains
from lenstronomy.LensModel.Solver.solver4point import Solver4Point

class LenstronomyWrap:

    def __init__(self,xtol=1.49012e-10,min_distance=0.01, search_window=5, precision_limit=10**(-10), num_iter_max=100,
                 multiplane=None,cosmo=None,z_source = None):

        self.xtol = xtol
        self.min_distance = 0.01
        self.search_window = 5
        self.precision_limit = 10**-10
        self.num_iter_max = 100
        self.cosmo = cosmo

        assert z_source is not None
        self.zsrc = z_source

        assert multiplane is not None
        self.multiplane = multiplane

        self.lens_model_list,self.lens_model_params = [],[]

    def update_settings(self,xtol=None,min_distance=None,search_window = None,precision_limit=None,num_iter_max = None):

        if xtol is not None:
            self.xtol = xtol
        if min_distance is not None:
            self.min_distance = min_distance
        if search_window is not None:
            self.search_window = search_window
        if precision_limit is not None:
            self.precision_limit = precision_limit
        if num_iter_max is not None:
            self.num_iter_max = num_iter_max

    def assemble(self,system):

        lens_model_list = []
        lens_model_params = []

        for deflector in system.lens_components:

            lens_model_list.append(deflector.args['lenstronomy_name'])
            lens_model_params.append(deflector.lenstronomy_args)

            if deflector.has_shear:

                lens_model_list.append('SHEAR')

                e1,e2 = polar_to_cart(deflector.args['shear'],deflector.args['shear_theta'])

                lens_model_params.append({'e1':e1,'e2':e2})

        self.lens_model_list, self.lens_model_params = lens_model_list,lens_model_params

        self.redshift_list = system.redshift_list

    def get_lensmodel(self):

        return LensModel(lens_model_list=self.lens_model_list, z_source=self.zsrc,
                              redshift_list=self.redshift_list,
                              cosmo=self.cosmo, multi_plane=self.multiplane)

    def update_lensparams(self, component_index, newkwargs):

        self.lens_model_params[component_index] = newkwargs

    def reset_assemble(self):

        self.lens_model_list, self.lens_model_params = [], []

    def solve_leq(self,xsrc,ysrc):

        lensmodel = LensModel(lens_model_list=self.lens_model_list,z_source=self.zsrc,redshift_list=self.redshift_list,
                              cosmo=self.cosmo,multi_plane=self.multiplane)

        lensEquationSolver = LensEquationSolver(lensModel=lensmodel)

        x_image,y_image =  lensEquationSolver.image_position_from_source(kwargs_lens=self.lens_model_params, sourcePos_x=xsrc, sourcePos_y=ysrc,
                                                                         min_distance=self.min_distance, search_window=self.search_window,
                                                                         precision_limit=self.precision_limit, num_iter_max=self.num_iter_max)

        return x_image,y_image

    def optimize_lensmodel(self,x_image,y_image):

        self.model = self.get_lensmodel()

        solver4Point = Solver4Point(lensModel=self.model, decoupling=True)
        kwargs_fit = solver4Point.constraint_lensmodel(x_pos=x_image, y_pos=y_image, kwargs_list=self.lens_model_params,
                                                       xtol=self.xtol)[0]

        return kwargs_fit









