import numpy as np

class Data:

    def __init__(self,x,y,m,t,source):

        self.set_pos(x,y)
        if m is not None:
            self.set_mag(m)
        else:
            self.m = None
        if t is not None:
            self.set_t(t)
        else:
            self.t = None
        self.set_src(source[0],source[1])

    def sort_by_pos(self,x,y):

        dr = np.sqrt(np.array(self.x - x)**2 + np.array(self.y - y)**2)

        inds = np.argsort(dr)
        self.x = self.x[inds]
        self.y = self.y[inds]
        if self.m is not None:
            self.m = self.m[inds]
        if self.t is not None:
            self.t = self.t[inds]

    def set_pos(self,x,y):

        self.x,self.y = np.array(x),np.array(y)

    def set_mag(self,mag):

        self.m = np.array(mag)*np.max(mag)**-1

    def set_t(self,t):

        self.t = np.array(t)

    def set_src(self,srcx,srcy):

        self.srcx,self.srcy = srcx,srcy

    def flux_ratios(self,index):

        ref = float(self.m[index])

        m = np.array(self.m)*ref**-1

        m = np.delete(m,index)

        self.fluxratios = m

        return m

    def all(self):

        return [self.x,self.y,self.m,self.t]


