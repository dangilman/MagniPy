from generate_input import LenstronomyWrap
from generate_input import LensModelExtensions
import numpy as np

class MultiLensWrapper:

    def __init__(self,gridsize=int,res=0.0005,source_shape='',astropy_class=None,z_source=None,source_size=None):

        self.gridsize = gridsize
        self.res = res
        self.source_shape = source_shape
        self.astropy_class = astropy_class
        self.z_source = z_source
        self.source_size = source_size

    def compute_mag(self,xpos,ypos,lens_system,print_mag=False):

        lenstronomywrap = LenstronomyWrap(multiplane=lens_system.multiplane, cosmo=self.astropy_class,
                                               z_source=self.z_source)

        lenstronomywrap.assemble(lens_system)

        lensModelExtensions = LensModelExtensions(lens_model_list=lenstronomywrap.lens_model_list,
                                                  z_source=self.z_source, redshift_list=lens_system.redshift_list,
                                                  cosmo=self.astropy_class, multi_plane=lens_system.multiplane)

        fluxes = lensModelExtensions.magnification_finite(x_pos=xpos,
                                                          y_pos=ypos,
                                                          kwargs_lens=lenstronomywrap.lens_model_params,
                                                          source_sigma=self.source_size, window_size=self.gridsize,
                                                          grid_number=self.gridsize * self.res ** -1,
                                                          shape=self.source_shape)

        return np.array(fluxes)
