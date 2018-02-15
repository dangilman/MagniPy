import numpy as np

class Data:

    def __init__(self,x,y,m,t,source):

        self.decimals_pos = 4
        self.decimals_mag = 5
        self.decimals_time = 1
        self.decimals_src = 6

        self.set_pos(x,y)
        if m is not None:
            self.set_mag(m)
        else:
            self.m = None
        if t is not None:
            self.set_t(t)
        else:
            self.set_t(None)
        if source is not None:
            self.set_src(source[0],source[1])
        else:
            self.set_source(None,None)


    def sort_by_pos(self,x,y):

        import itertools

        if self.nimg != 4:
            return
        x_self = np.array(list(itertools.permutations(self.x)))
        y_self = np.array(list(itertools.permutations(self.y)))

        indexes = [0,1,2,3]
        index_iterations = list(itertools.permutations(indexes))
        delta_r = []

        for i in range(0,int(len(x_self))):
            dr = 0
            for j in range(0,int(len(x_self[0]))):
                dr += (x_self[i][j] - x[j])**2 + (y_self[i][j] - y[j])**2

            delta_r.append(dr**.5)

        min_indexes = np.array(index_iterations[np.argmin(delta_r)])

        self.set_pos(self.x[min_indexes],self.y[min_indexes])

        if self.m is not None:
            self.set_mag(self.m[min_indexes])
        if self.t is not None:
            self.set_t(self.t[min_indexes])

    def set_pos(self,x,y):

        self.x,self.y = np.round(np.array(x),self.decimals_pos),np.round(np.array(y),self.decimals_pos)
        self.nimg = len(self.x)

    def set_mag(self,mag):

        if mag is None:
            self.mag = None
        else:
            self.m = np.round(np.array(mag)*np.max(mag)**-1,self.decimals_mag)

    def set_t(self,t):

        if t is None:
            self.t = None
        else:
            self.t = np.round(np.array(t),self.decimals_time)

    def set_src(self,srcx,srcy):

        if srcx is None or srcy is None:
            self.srcx,self.srcy = None,None
        else:
            self.srcx,self.srcy = np.round(srcx,self.decimals_src),np.round(srcy,self.decimals_src)

    def compute_flux_ratios(self,index=1):

        ref = float(self.m[index])

        m = np.array(self.m)*ref**-1

        m = np.delete(m,index)

        self.fluxratios = m

    def all(self):

        return [self.x,self.y,self.m,self.t]




