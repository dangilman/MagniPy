import numpy as np

class Local:

    def __init__(self,x0,y0,redshift=None,res=0.0004,size=int):

        self.x0,self.y0 = x0,y0
        steps = max(1, 2 * round(size * res ** -1))
        self.xgrid, self.ygrid = np.meshgrid(np.linspace(self.x - size, self.x + size, steps),
                                                 np.linspace(self.y - size, self.y + size, steps))
