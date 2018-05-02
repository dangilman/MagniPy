from MagniPy.Solver.LenstronomyWrap.generate_input import LenstronomyWrap
from lenstronomy.LensModel.lens_model_extensions import LensModelExtensions
#from lenstronomy.util.util import array2image
import numpy as np

class MultiLensWrapper:

    def __init__(self,gridsize=int,res=0.0005,source_shape='',multiplane=True,astropy_class=None,z_source=None,source_size=None):

        self.gridsize = gridsize
        self.res = res
        self.source_shape = source_shape
        self.astropy_class = astropy_class
        self.z_source = z_source
        self.source_size = source_size
        self.multiplane = multiplane

    def compute_mag(self,xpos,ypos,lens_system,print_mag=False):

        lenstronomywrap = LenstronomyWrap(multiplane=self.multiplane, cosmo=self.astropy_class,
                                               z_source=self.z_source)

        lenstronomywrap.assemble(lens_system)

        lensModelExtensions = LensModelExtensions(lens_model_list=lenstronomywrap.lens_model_list,
                                                  z_source=self.z_source, redshift_list=lenstronomywrap.redshift_list,
                                                  cosmo=self.astropy_class, multi_plane=lens_system.multiplane)

        fluxes = lensModelExtensions.magnification_finite(x_pos=xpos,
                                                          y_pos=ypos,
                                                          kwargs_lens=lenstronomywrap.lens_model_params,
                                                          source_sigma=self.source_size, window_size=self.gridsize,
                                                          grid_number=self.gridsize * self.res ** -1,
                                                          shape=self.source_shape)

        return np.array(fluxes)

    def rayshoot(self,x,y,lens_system,source_function):

        lenstronomywrap = LenstronomyWrap(multiplane=self.multiplane, cosmo=self.astropy_class,
                                          z_source=self.z_source)
        lenstronomywrap.assemble(lens_system)

        lensModel = lenstronomywrap.get_lensmodel()

        xnew,ynew = lensModel.ray_shooting(x,y,lenstronomywrap.lens_model_params)

        beta = source_function(xnew,ynew)

        return array2image(beta,len(beta)**.5,len(beta)**.5)

