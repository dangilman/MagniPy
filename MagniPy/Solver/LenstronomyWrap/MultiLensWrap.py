from MagniPy.Solver.LenstronomyWrap.generate_input import LenstronomyWrap
from lenstronomy.LensModel.lens_model_extensions import LensModelExtensions
from lenstronomy.Util.util import array2image,image2array
import matplotlib.pyplot as plt
import numpy as np

class MultiLensWrapper:

    def magnification(self,xpos,ypos,xcoords,ycoords,lensmodel,lens_model_params,source_function,res):

        flux = []

        for i in range(0,len(xpos)):

            x,y = xpos[i]+xcoords,ypos[i]+ycoords

            betax,betay = lensmodel.ray_shooting(x,y,lens_model_params)

            image = source_function(betax,betay)

            flux.append(np.sum(image*res**2))

        return np.array(flux)

    def rayshoot(self,x,y,lens_system,source_function):

        lenstronomywrap = LenstronomyWrap(multiplane=self.multiplane, cosmo=self.astropy_class,
                                          z_source=self.z_source)
        lenstronomywrap.assemble(lens_system)

        lensModel = lenstronomywrap.get_lensmodel()

        xnew,ynew = lensModel.ray_shooting(x,y,lenstronomywrap.lens_model_params)

        beta = source_function(xnew,ynew)

        return array2image(beta,len(beta)**.5,len(beta)**.5)


