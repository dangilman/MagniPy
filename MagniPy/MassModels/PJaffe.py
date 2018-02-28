import numpy as np
from MagniPy.LensBuild.Cosmology.cosmology import Cosmo

class PJaffe:

    def __init__(self,z=None,zsrc=None,cosmology=None):

        if cosmology is None:

            self.cosmology = Cosmo(zd=z,zsrc=zsrc,compute=False)

        else:
            self.cosmology = cosmology

        self.sigmacrit = self.cosmology.sigmacrit

    def convergence(self,rcore,rtrunc,r=False):
        if r is False:
            return (rcore**2+self.x**2 + self.y**2)**-.5 - (rtrunc**2+self.x**2 + self.y**2)**-.5
        else:
            return (rcore**2+r**2)**-.5 - (rtrunc**2+r**2)**-.5

    def def_angle(self,x,y,rcore=0,rt=None,b=None,center_x=None,center_y=None):

        x = x - center_x
        y = y - center_y

        r=np.sqrt(x**2+y**2)

        magdef = b*(np.sqrt(1+(rcore*r**-1)**2)-np.sqrt(1+(rt*r**-1)**2)+(rt-rcore)*r**-1)

        return magdef*x*r**-1,magdef*y*r**-1

    def params(self,x=None,y=None,mass=None,rt=None,rc=0,**kwargs):

        subkwargs = {}
        subkwargs['b'] = self.b(mass,rt,rc)
        subkwargs['rt'] = rt
        subkwargs['center_x'] = x
        subkwargs['center_y'] = y

        otherkwargs ={}
        otherkwargs['mass'] = mass
        otherkwargs['name'] = 'PJaffe'

        return subkwargs,otherkwargs

    def b(self,M,rt,rc):

        return M*((rt-rc)*self.sigmacrit*np.pi)**-1




