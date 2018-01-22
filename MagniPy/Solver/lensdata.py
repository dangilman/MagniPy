import numpy as np

class LensData:

    def __init__(self,x,y,m,t,source):

        self.x = x
        self.y = y
        self.m = m*np.max(m)**-1
        self.t = t
        self.srcx = source[0]
        self.srcy = source[1]

    def flux_ratios(self,index):

        ref = float(self.m[index])

        m = np.array(self.m)*ref**-1

        m = np.delete(m,index)

        return m