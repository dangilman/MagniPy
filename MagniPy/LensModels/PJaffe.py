import numpy as np

class PJaffe_profile:
    def __init__(self,xgrid,ygrid):

        self.x,self.y = xgrid,ygrid

    def convergence(self,rcore,rtrunc,r=False):
        if r is False:
            return (rcore**2+self.x**2 + self.y**2)**-.5 - (rtrunc**2+self.x**2 + self.y**2)**-.5
        else:
            return (rcore**2+r**2)**-.5 - (rtrunc**2+r**2)**-.5

    def def_angle(self,rcore=None,rtrunc=None,b=None,x0=None,y0=None):

        x = self.x - x0
        y = self.y - y0

        r=np.sqrt(x**2+y**2)

        magdef = b*(np.sqrt(1+(rcore*r**-1)**2)-np.sqrt(1+(rtrunc*r**-1)**2)+(rtrunc-rcore)*r**-1)

        return magdef*x*r**-1,magdef*y*r**-1
