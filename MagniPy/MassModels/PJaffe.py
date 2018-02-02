import numpy as np
from MagniPy.LensBuild.cosmology import Cosmo

class Pjaffe(Cosmo):

    def __init__(self,z1=0.5,z2=1.5,cosmology=''):

        Cosmo.__init__(self, zd=z1, zsrc=z2)

    def convergence(self,rcore,rtrunc,r=False):
        if r is False:
            return (rcore**2+self.x**2 + self.y**2)**-.5 - (rtrunc**2+self.x**2 + self.y**2)**-.5
        else:
            return (rcore**2+r**2)**-.5 - (rtrunc**2+r**2)**-.5

    def def_angle(self,xgrid,ygrid,rcore=None,rtrunc=None,b=None,x0=None,y0=None):

        x = xgrid - x0
        y = ygrid - y0

        r=np.sqrt(x**2+y**2)

        magdef = b*(np.sqrt(1+(rcore*r**-1)**2)-np.sqrt(1+(rtrunc*r**-1)**2)+(rtrunc-rcore)*r**-1)

        return magdef*x*r**-1,magdef*y*r**-1

    def params(self,M,rt,rc=0,**kwargs):

        subkwargs = {}
        subkwargs['name'] = 'Pjaffe'
        subkwargs['b'] = self.b(M,rt,rc)

        return subkwargs

    def b(self,M,rt,rc):

        return M*((rt-rc)*self.sigmacrit*np.pi)**-1




