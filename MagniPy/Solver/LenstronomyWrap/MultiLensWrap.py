from MagniPy.Solver.LenstronomyWrap.lenstronomy_wrap import LenstronomyWrap
from lenstronomy.LensModel.lens_model_extensions import LensModelExtensions
from lenstronomy.Util.util import array2image,image2array
import matplotlib.pyplot as plt
import numpy as np

class MultiLensWrapper:

    def magnification(self,xpos,ypos,lensmodel,lens_model_params,source_function,res,n=4):

        flux = []

        for i in range(0,n):

            betax,betay = lensmodel.ray_shooting(xpos[i].ravel(),ypos[i].ravel(),lens_model_params)

            image = source_function(betax,betay)

            #plt.imshow(image.reshape(240, 240))
            #plt.show()

            flux.append(np.sum(image*res**2))

        return np.array(flux)

    def rayshoot(self,x,y,lens_system,source_function,astropy,zsrc):

        lenstronomywrap = LenstronomyWrap(cosmo=astropy,
                                          z_source=zsrc)

        lensModel,lens_model_params = lenstronomywrap.get_lensmodel(lens_system)

        xnew,ynew = lensModel.ray_shooting(x.ravel(),y.ravel(),lens_model_params)

        beta = source_function(xnew,ynew)

        return beta


