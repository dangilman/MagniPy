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

    def __call__(self, **kwargs):

        if self.dimension==1:

            kde = self.kde(self.data,self.X,**kwargs)[0]

            density = np.histogram(kde,bins=len(self.X),density=True,weights=kde.ravel())[0]

            return density.T,self.X,None

        elif self.dimension==2:

            kde,xx,yy = self.kde(self.data,self.X,self.Y,
                                 pranges_training=[self.kde_train_ranges[self.param_names[0]],self.kde_train_ranges[self.param_names[1]]],
                                 prior_weights=self.prior_weights)

            density = np.histogram2d(xx,yy, bins=len(self.X),normed=True, weights=kde.ravel())[0]

            return density.T,self.X,self.Y

def scotts_factor(n,d=2):

    return n ** (-1. / (d + 4))



