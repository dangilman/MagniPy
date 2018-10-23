from MagniPy.Analysis.KDE.kde import *

class SingleDensity:

    def __init__(self,pnames=None,samples=None,pranges=None,kde_train_ranges=None,kde_class ='mine',
                 steps=50,scale=5,reweight=True,kernel_function='Sigmoid',bandwidth_scale=1):

        self.reweight = reweight
        self.scale = scale
        self.steps = steps
        self.param_names = pnames
        self.full_samples = samples
        self.prior_weights = samples.weights
        self.kernel_function = kernel_function
        self.bandwidth_scale = bandwidth_scale

        self.full_ranges = pranges

        if kde_train_ranges is None:
            self.kde_train_ranges = pranges
        else:
            self.kde_train_ranges = kde_train_ranges

        self.posteriorsamples = {x: samples.samples[x] for x in pnames}

        self.pranges = {x: pranges[x] for x in pnames}
        self.dimension = len(self.posteriorsamples.keys())
        assert self.dimension in [1, 2], 'Data dimension must be 1 or 2.'

        self.kde = self._set_kde(kde_class)

        self.kde_class = kde_class

        if self.dimension == 1:
            self.X = np.linspace(self.pranges[self.param_names[0]][0],self.pranges[self.param_names[0]][1],self.steps)
            self.data = samples.samples[self.param_names[0]]

        elif self.dimension == 2:
            self.X = np.linspace(self.pranges[self.param_names[0]][0], self.pranges[self.param_names[0]][1], self.steps)
            self.Y = np.linspace(self.pranges[self.param_names[1]][0], self.pranges[self.param_names[1]][1], self.steps)
            self.data = np.vstack([np.array(samples.samples[self.param_names[0]]), samples.samples[self.param_names[1]]]).T

    def _set_kde(self,kde_class):

        if self.dimension==2:
            if kde_class=='scipy':
                return KDE_scipy(dim=2)
            else:
                return KernelDensity(reweight=self.reweight,scale=self.scale,
                                     bandwidth_scale=self.bandwidth_scale,kernel=self.kernel_function)
        else:
            raise Exception('not yet implemented')

    def _trim_density(self, density, x_left, y_left, x_right, y_right):

        density = density[x_left:x_right,y_left:y_right]
        return density

    def _trim(self, ranges, xtrim, ytrim, N, pnames):

        ranx = ranges[pnames[0]][1] - ranges[pnames[0]][0]
        rany = ranges[pnames[1]][1] - ranges[pnames[1]][0]

        dx_left = (xtrim[0] - ranges[pnames[0]][0])*ranx**-1

        dx_right = (ranges[pnames[0]][1] - xtrim[1])*ranx**-1

        dy_top = (ranges[pnames[1]][1] - ytrim[1])*rany**-1
        dy_bottom = (ytrim[0] - ranges[pnames[1]][0])*rany**-1

        start_xlow = N*dx_left
        start_xhigh = N * (1 - dx_right)

        start_ylow = N * dy_bottom
        start_yhigh = N * (1 - dy_top)

        ranges[pnames[0]][0] = xtrim[0]
        ranges[pnames[0]][1] = xtrim[1]
        ranges[pnames[1]][0] = ytrim[0]
        ranges[pnames[1]][1] = ytrim[1]

        return ranges, int(np.round(start_xlow)), int(np.round(start_ylow)), int(np.round(start_xhigh)), \
               int(np.round(start_yhigh))


    def __call__(self, xtrim = None, ytrim = None):
        
        if xtrim is None and ytrim is None:

            param_ranges = self.pranges
            X, Y = self.X, self.Y

        else:

            param_ranges, xstart, ystart, xend, yend = self._trim(self.pranges, xtrim,
                                                                  ytrim, len(self.X), self.param_names)

            X, Y = np.linspace(xstart, xend, self.steps), np.linspace(ystart, yend, self.steps)

        kde, xx, yy = self.kde(self.data, X, Y, pranges_true=[self.kde_train_ranges[self.param_names[0]],
                               self.kde_train_ranges[self.param_names[1]]], prior_weights=self.prior_weights)

        density = np.histogram2d(xx, yy, bins=len(X), normed=True, weights=kde.ravel())[0]

        return density.T, param_ranges

def scotts_factor(n,d=2):

    return n ** (-1. / (d + 4))



