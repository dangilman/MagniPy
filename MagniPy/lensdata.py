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
            self.set_t([0,0,0,0])
        if source is not None:
            self.set_src(source[0],source[1])
        else:
            self.set_src(None,None)


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
            try:
                self.m = np.round(np.array(mag)*np.max(mag)**-1,self.decimals_mag)
            except:
                self.m = np.nan*np.ones_like(self.x)

    def set_t(self,t):

        if t is None:
            self.t = None
        else:
            self.t = np.round(np.array(t),self.decimals_time)

    def set_src(self,srcx,srcy):

        if srcx is None or srcy is None:
            self.srcx,self.srcy = None,None
        else:
            self.srcx,self.srcy = np.round(np.mean(srcx),self.decimals_src),np.round(np.mean(srcy),self.decimals_src)

    def compute_flux_ratios(self,fluxes=None,index=1):

        if fluxes is None:

            ref = float(self.m[index])
            try:
                m = np.array(self.m)*ref**-1
            except:
                print 'reference data has zero flux'
                m = np.ones_like(self.m)*np.nan

            m = np.delete(m,index)

            self.fluxratios = m

            return self.fluxratios

        else:

            ref = float(fluxes[index])

            try:
                fluxes = np.array(fluxes)*ref**-1
            except:
                print 'reference data has zero flux'
                fluxes = np.ones_like(fluxes)*np.nan

            return np.delete(fluxes, index)

    def flux_anomaly(self, other_data=None, index=1, sum_in_quad=False):

        other_mags = other_data.m

        other = self.compute_flux_ratios(fluxes=other_mags,index=index)
        
        self.compute_flux_ratios(index=index)

        if sum_in_quad:

            return (np.sum((self.fluxratios - other)**2*other**-2))**.5

        else:

            return np.absolute((self.fluxratios - other) * other ** -1)



    def all(self):

        return [self.x,self.y,self.m,self.t]




